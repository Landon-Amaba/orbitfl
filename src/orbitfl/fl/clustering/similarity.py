"""Clustering by model-update similarity (MCFL-style baseline).

SimilarityReactive discovers clusters from the clients' model updates, following
MCFL:
  - a bootstrap of T_init rounds during which updates are collected,
  - a pairwise similarity that blends the cosine of the latest updates with the
    cosine of the updates accumulated over the bootstrap (weight gamma),
  - average-linkage agglomerative clustering with distance threshold d_th.
With recluster_every_R set, the bootstrap and clustering are repeated every R
rounds. overhead_rounds counts T_init for every completed clustering.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from orbitfl.fl.clustering.base import ClusteringStrategy, FLState

_UPDATE_REFERENCES: tuple[str, str] = ("last_global", "round_start")


class SimilarityReactive(ClusteringStrategy):
    """Clusters discovered from model-update similarity.

    The first T_init calls to membership() collect updates and return the current
    partition, which is random at the start. After T_init rounds a similarity matrix
    is built and all LEOs are clustered by average-linkage agglomerative clustering
    with distance threshold d_th. With recluster_every_R set, this cycle repeats
    every R rounds, and overhead_rounds grows by T_init each time.

    Similarity between clients i and j:
      s_ij = gamma * cos(g_i, g_j) + (1 - gamma) * cos(delta_i, delta_j)
      d_ij = 1 - s_ij
    where g_i is the client's update in the latest round and delta_i its update
    accumulated over the current bootstrap window.

    update_reference selects what g_i and delta_i are measured against:
      "last_global" (default):
          g_i = w_i - last synced global model; delta_i = running sum of g_i.
          A client with no update in a round contributes a zero vector, whose
          cosine with any other vector is 0.
      "round_start" (the definition in the MCFL paper):
          g_i = w_i - the model client i started the round from;
          delta_i = w_i - the model client i started the bootstrap window from.
          A client absent from a round keeps its most recent g_i and delta_i, so
          absence is not read as dissimilarity.
    """

    def __init__(
        self,
        n_leo: int,
        n_clusters: int,
        T_init: int = 10,
        recluster_every_R: int | None = None,
        gamma: float = 0.5,
        d_th: float = 1.0,
        seed: int = 0,
        update_reference: str = "last_global",
    ) -> None:
        """Parameters
        ----------
        n_leo              : int    Number of LEO satellites.
        n_clusters         : int    Number of clusters in the initial random partition.
                                    The number found by clustering is set by d_th.
        T_init             : int    Bootstrap rounds per clustering (MCFL uses 10).
        recluster_every_R  : int | None
                                    Repeat bootstrap and clustering every R rounds.
                                    None: cluster once and keep that partition.
        gamma              : float  Weight on the latest-update cosine; (1 - gamma) on
                                    the accumulated-update cosine.
        d_th               : float  Distance threshold for agglomerative clustering
                                    (MCFL uses 1.0).
        seed               : int    Seed for the initial random partition.
        update_reference   : str    "last_global" or "round_start"; see class docstring.

        Raises
        ------
        ValueError  If update_reference is not a recognised value.
        """
        if update_reference not in _UPDATE_REFERENCES:
            raise ValueError(
                f"update_reference must be one of {_UPDATE_REFERENCES}, got {update_reference!r}"
            )
        self._update_reference = update_reference
        self._n_leo = n_leo
        self._n_clusters = n_clusters
        self._T_init = T_init
        self._R = recluster_every_R
        self._gamma = gamma
        self._d_th = d_th
        self._rng = np.random.default_rng(seed)

        # Length of a flattened model update, set from the first ClientUpdate received
        self._dim: int | None = None

        # Accumulated update vectors, (N_leo, dim) float64; reset at each bootstrap
        self._accumulated: np.ndarray | None = None
        # Update vector from the most recent round: (N_leo, dim)
        self._instantaneous: np.ndarray | None = None

        # Model each client started the current bootstrap window from ("round_start"
        # only): sat_id -> (dim,) float64
        self._window_start: dict[int, np.ndarray] = {}

        # Rounds elapsed in the current bootstrap window
        self._bootstrap_count: int = 0

        # Whether a bootstrap is in progress
        self._in_bootstrap: bool = True

        # Current cluster assignment, (N_leo,) int; random until the first clustering
        self._current: np.ndarray = self._rng.integers(
            0, n_clusters, size=n_leo, dtype=np.int32
        )

        # Round index at which the current cycle started
        self._cycle_start: int = 0

        # Bootstrap rounds spent across all completed clusterings
        self._total_overhead: int = 0

    # --- ClusteringStrategy interface ---

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """Collect updates during a bootstrap; cluster after T_init rounds.

        During bootstrap rounds the current partition is returned, so the FL loop keeps
        training. After T_init rounds the similarity matrix is built and the partition
        is replaced.

        Parameters
        ----------
        round_idx : int      Current round (0-based).
        state     : FLState  Must contain client_updates from the previous round.

        Returns
        -------
        np.ndarray  (N_leo,) int  Cluster id; no -1 entries.
        """
        # Start a bootstrap cycle if one is already running, or if recluster_every_R is
        # set and the next clustering is due.
        if (
            not self._in_bootstrap
            and self._R is not None
            and (round_idx - self._cycle_start) >= self._R
        ):
            self._start_bootstrap(round_idx)

        # Add this round's client updates to the accumulators
        self._accumulate(state)

        if self._in_bootstrap:
            self._bootstrap_count += 1
            if self._bootstrap_count >= self._T_init:
                self._cluster()
                self._total_overhead += self._T_init
                self._in_bootstrap = False
                self._cycle_start = round_idx

        return self._current.copy()

    @property
    def overhead_rounds(self) -> int:
        """Bootstrap rounds spent across all completed clusterings.

        Only cycles that finished (T_init rounds collected and a new partition
        adopted) are counted; a bootstrap in progress is not.
        """
        return self._total_overhead

    # --- Helpers ---

    def _start_bootstrap(self, round_idx: int) -> None:
        """Reset accumulators and enter a new bootstrap window."""
        self._accumulated = None
        self._instantaneous = None
        self._window_start = {}
        self._bootstrap_count = 0
        self._in_bootstrap = True

    def _accumulate(self, state: FLState) -> None:
        """Add the updates in client_updates to the accumulators.

        The update of client i is state_dict_i - global_model, both flattened to 1-D.
        Clients without an update contribute a zero vector.
        """
        updates = state.client_updates
        if not updates:
            return

        # Allocate the accumulators whenever they are missing: on the first call and
        # after _start_bootstrap has reset them for a new cycle.
        if self._accumulated is None or self._instantaneous is None:
            if self._dim is None:
                ref = updates[0].state_dict
                self._dim = int(sum(v.numel() for v in ref.values()))
            self._accumulated = np.zeros((self._n_leo, self._dim), dtype=np.float64)
            self._instantaneous = np.zeros((self._n_leo, self._dim), dtype=np.float64)

        if self._update_reference == "round_start":
            self._accumulate_round_start(state)
            return

        # Flatten the global model once
        global_flat = _flatten_state(state.global_model)

        # Clear the latest-update buffer; it stays zero for clients with no update
        self._instantaneous[:] = 0.0

        for u in updates:
            if u.sat_id < 0 or u.sat_id >= self._n_leo or u.n_samples == 0:
                continue
            delta = _flatten_state(u.state_dict) - global_flat
            self._instantaneous[u.sat_id] = delta
            self._accumulated[u.sat_id] += delta

    def _accumulate_round_start(self, state: FLState) -> None:
        """Record g_i and delta_i relative to each client's own starting models.

        Buffers are not cleared between rounds: a client with no update this round
        keeps its most recent values.

        Parameters
        ----------
        state : FLState  Must carry round_start_models for every update.

        Raises
        ------
        ValueError  If an update has no recorded start model.
        """
        assert self._accumulated is not None and self._instantaneous is not None
        for u in state.client_updates:
            if u.sat_id < 0 or u.sat_id >= self._n_leo or u.n_samples == 0:
                continue
            if u.sat_id not in state.round_start_models:
                raise ValueError(f"No start model recorded for satellite {u.sat_id}.")
            start = _flatten_state(state.round_start_models[u.sat_id])  # (dim,)
            current = _flatten_state(u.state_dict)                       # (dim,)
            first = self._window_start.setdefault(u.sat_id, start)       # (dim,)
            self._instantaneous[u.sat_id] = current - start
            self._accumulated[u.sat_id] = current - first

    def _cluster(self) -> None:
        """Build the similarity matrix and run agglomerative clustering.

        Similarity: s_ij = gamma * cos(g_i, g_j) + (1 - gamma) * cos(delta_i, delta_j)
        Distance:   d_ij = 1 - s_ij, in [0, 2]
        """
        if self._accumulated is None or self._instantaneous is None:
            # No updates received: keep the current partition
            return

        inst_sim = _pairwise_cosine(self._instantaneous)   # (N, N)
        accum_sim = _pairwise_cosine(self._accumulated)    # (N, N)
        sim = self._gamma * inst_sim + (1.0 - self._gamma) * accum_sim  # (N, N)
        dist = np.clip(1.0 - sim, 0.0, 2.0)  # cosine distance in [0, 2]

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self._d_th,
            metric="precomputed",
            linkage="average",
        )
        labels = agg.fit_predict(dist)
        self._current = labels.astype(np.int32)


# --- Module-level helpers ---


def _flatten_state(state_dict: dict) -> np.ndarray:
    """
    Concatenate all tensors in a state_dict into a single float64 1-D array.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]

    Returns
    -------
    np.ndarray  (D,) float64
    """
    return np.concatenate(
        [v.detach().cpu().float().numpy().ravel() for v in state_dict.values()],
        dtype=np.float64,
    )


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarities for rows of X.

    Rows with zero norm are treated as orthogonal to all other rows (cos = 0).

    Parameters
    ----------
    X : (N, D) float64

    Returns
    -------
    np.ndarray  (N, N) float64  in [-1, 1]
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # (N, 1)
    norms = np.where(norms == 0, 1.0, norms)           # avoid divide-by-zero
    X_norm = X / norms                                  # (N, D)
    sim = X_norm @ X_norm.T                             # (N, N)
    return np.clip(sim, -1.0, 1.0)
