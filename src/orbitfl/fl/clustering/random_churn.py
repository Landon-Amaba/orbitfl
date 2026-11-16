"""Random clustering with a matched membership-change rate (control).

RandomChurnClustering re-draws a random partition at the rate at which
satellites change serving MEO in the trace (Trace.churn_rate_per_orbit). It
keeps the amount of membership change while removing any geographic meaning
from the clusters, which separates the effect of geography from the effect of
changing membership.
"""

from __future__ import annotations

import math

import numpy as np

from orbitfl.fl.clustering.base import ClusteringStrategy, FLState
from orbitfl.trace.reader import Trace


class RandomChurnClustering(ClusteringStrategy):
    """Random partition whose membership changes at a matched rate.

    Every round, each satellite independently moves to a different, randomly chosen
    cluster with probability p_switch. If a geographic strategy beats this one, the
    advantage comes from the geographic structure of its clusters and not from how
    often membership changes.

    Cluster ids run 0 .. n_clusters-1; there are no -1 entries.

    p_switch:
      churn rate per orbit  = mean MEO handovers per satellite per LEO orbit (trace)
      LEO period            = from the altitude in the trace
      churn rate per round  = churn_rate_per_orbit * round_duration_s / leo_period_s
      p_switch              = churn rate per round, clamped to [0, 1]
    """

    def __init__(
        self,
        trace: Trace,
        n_leo: int,
        n_clusters: int,
        round_duration_s: int,
        seed: int,
    ) -> None:
        """Parameters
        ----------
        trace            : Trace  Provides churn_rate_per_orbit() and the LEO altitude.
        n_leo            : int    Number of LEO satellites.
        n_clusters       : int    Number of clusters.
        round_duration_s : int    Seconds per FL round.
        seed             : int    Seed for the random partition.
        """
        churn_per_orbit: float = trace.churn_rate_per_orbit()

        # Estimate LEO orbital period from altitude in the trace
        leo_pos_t0 = trace._positions[
            (trace._positions["t_idx"] == 0) & (trace._positions["kind"] == "leo")
        ]
        mean_alt_km: float = float(leo_pos_t0["alt_km"].mean())
        # T = 2pi sqrt(a^3/mu) for a circular orbit
        from orbitfl.geometry.constants import MU_KM3_S2 as _MU
        from orbitfl.geometry.constants import R_EARTH_KM as _RE
        a_km = _RE + mean_alt_km
        leo_period_s: float = 2.0 * math.pi * math.sqrt(a_km**3 / _MU)

        # Convert churn rate per orbit to per round, then clamp
        churn_per_round: float = churn_per_orbit * round_duration_s / leo_period_s
        self._p_switch: float = float(np.clip(churn_per_round, 0.0, 1.0))

        self._n_leo = n_leo
        self._n_clusters = n_clusters
        self._rng = np.random.default_rng(seed)

        # Start from a random partition
        self._current = self._rng.integers(0, n_clusters, size=n_leo, dtype=np.int32)

    # --- ClusteringStrategy interface ---

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """Move each satellite to another cluster with probability p_switch.

        A satellite that switches draws its new cluster uniformly from the other
        n_clusters - 1 clusters.

        Parameters
        ----------
        round_idx : int      Unused; a switch draw is made on every call.
        state     : FLState  Unused.

        Returns
        -------
        np.ndarray  (N_leo,) int  Cluster ids in 0 .. n_clusters-1.
        """
        # Each LEO switches with probability p_switch
        switch_mask = self._rng.random(self._n_leo) < self._p_switch
        n_switch = int(switch_mask.sum())

        if n_switch > 0 and self._n_clusters > 1:
            # Uniform over the other clusters
            offsets = self._rng.integers(1, self._n_clusters, size=n_switch)
            new_clusters = (self._current[switch_mask] + offsets) % self._n_clusters
            self._current[switch_mask] = new_clusters.astype(np.int32)

        return self._current.copy()

    @property
    def p_switch(self) -> float:
        """Per-round per-satellite switching probability."""
        return self._p_switch

    @property
    def overhead_rounds(self) -> int:
        """Zero: a random partition needs no discovery."""
        return 0
