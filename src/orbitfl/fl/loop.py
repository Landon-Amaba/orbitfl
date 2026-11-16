"""Federated training loop.

FederatedRunner(cfg, trace, world, strategy).run() returns a RunResult. Every
strategy runs for the same fixed number of rounds (fl.round_budget); there is
no early stopping. In round r:
  1. members = strategy.membership(r, state)
  2. the participants are the clients that have a cluster and a visible
     aggregator, sampled at participation_rate; the others skip the round
  3. each participant trains local_epochs on its data up to this round,
     starting from its cluster's model
  4. FedAvg within each cluster every round; FedAvg across clusters every
     sync_interval rounds
  5. metrics are logged: AUC and balanced accuracy on a held-out probe set for
     the global model and for the per-cluster (personalized) models,
     communication bytes, membership changes, and the ARI between the clusters
     and the ground regions
A round with no participants is logged and skipped.

Round r covers trace time [r*d, (r+1)*d), with d = round_duration_s.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from orbitfl.config import Config
from orbitfl.fl.aggregation import ClientUpdate, StateDict, fedavg, hierarchical_fedavg
from orbitfl.fl.client import SatelliteClient
from orbitfl.fl.clustering.base import ClusteringStrategy, FLState
from orbitfl.metrics.forecast import auc, balanced_accuracy
from orbitfl.models.forecaster import OccupancyForecaster
from orbitfl.trace.reader import Trace
from orbitfl.utils.device import get_device
from orbitfl.world.occupancy import OccupancyWorld, SatDataset

# Fraction of each satellite's samples held out as the probe set
_PROBE_FRAC: float = 0.20
_PARAM_BYTES: int = 4  # float32


@dataclasses.dataclass
class RunResult:
    """Per-round metrics and metadata produced by FederatedRunner.run().

    Attributes
    ----------
    metrics       : pd.DataFrame  One row per round.
    config        : Config        The config that produced this run.
    strategy_name : str           Name of the clustering strategy.
    churn_rate    : float         MEO handovers per satellite per orbit in the trace.
    """

    metrics: pd.DataFrame
    config: Config
    strategy_name: str
    churn_rate: float


class FederatedRunner:
    """Fixed-budget FL loop for any clustering strategy.

    Usage:
      runner = FederatedRunner(cfg, trace, world, strategy)
      result = runner.run()
    """

    def __init__(
        self,
        cfg: Config,
        trace: Trace,
        world: OccupancyWorld,
        strategy: ClusteringStrategy,
        device: torch.device | None = None,
    ) -> None:
        """Parameters
        ----------
        cfg      : Config             Full experiment config.
        trace    : Trace              Loaded trace.
        world    : OccupancyWorld     Occupancy world for this run's seed.
        strategy : ClusteringStrategy Clustering strategy to run.
        device   : torch.device       Compute device; auto-selected if None.
        """
        self._cfg = cfg
        self._trace = trace
        self._world = world
        self._strategy = strategy
        self._device = device or get_device()

        self._N = trace.n_leo
        self._T = trace.n_timesteps
        self._C = cfg.world.n_channels
        self._W = cfg.model.window

        # Number of trace timesteps in one FL round
        self._spr: int = max(
            1, cfg.fl.round_duration_s // cfg.sim.timestep_s
        )

        # Per-satellite datasets
        delta = cfg.experiment.delta_steps
        self._datasets: list[SatDataset] = [
            world.observation_stream(trace, k, window=self._W, delta_steps=delta)
            for k in range(self._N)
        ]

        # Train / probe split: the last _PROBE_FRAC of each satellite's samples is the probe
        self._train_slices: list[slice] = []
        self._probe_slices: list[slice] = []
        for ds in self._datasets:
            n = len(ds)
            n_probe = max(1, int(n * _PROBE_FRAC))
            n_train = max(0, n - n_probe)
            self._train_slices.append(slice(0, n_train))
            self._probe_slices.append(slice(n_train, n))

        # Global model and clients
        self._global_model = OccupancyForecaster(
            n_channels=self._C,
            window=self._W,
            use_position=cfg.model.use_position,
            hidden=cfg.model.hidden,
        )
        init_state: StateDict = {
            k: v.clone() for k, v in self._global_model.state_dict().items()
        }
        self._clients: list[SatelliteClient] = [
            SatelliteClient(
                sat_id=k,
                model_state=init_state,
                dataset=self._datasets[k],
                cfg=cfg.fl,
                model_cfg=cfg.model,
            )
            for k in range(self._N)
        ]

        # One probe subset per satellite, fixed for the whole run. The global probe
        # loader concatenates them; the personalized evaluation regroups the same subsets
        # by cluster each round. Both balanced accuracies are thus computed on the same
        # samples.
        self._probe_subset_per_sat: list[Subset] = [
            Subset(ds, range(*sl.indices(len(ds))))
            for ds, sl in zip(self._datasets, self._probe_slices)
        ]
        probe_ds = ConcatDataset(
            [s for s in self._probe_subset_per_sat if len(s) > 0]
        )
        self._probe_loader = DataLoader(
            probe_ds, batch_size=512, shuffle=False, num_workers=0
        )

        # For fl.per_sample_eval: the dataset indices of each satellite's probe samples
        # and the FL round each one falls in. The forecast window of sample i ends at
        # trace timestep (i + W - 1), so its round is (i + W - 1) // steps_per_round.
        R = self._cfg.fl.round_budget
        self._probe_idx_per_sat: list[np.ndarray] = []   # (n_probe_k,) dataset indices
        self._probe_round_per_sat: list[np.ndarray] = []  # (n_probe_k,) round per sample
        # Round of every dataset sample, per satellite. cluster_local_train uses it to
        # select the samples collected while a client belonged to its current cluster.
        self._sample_round_per_sat: list[np.ndarray] = []  # (len(ds),) round per sample
        for sl, ds in zip(self._probe_slices, self._datasets):
            n = len(ds)
            all_rounds = np.minimum(
                (np.arange(n, dtype=np.intp) + self._W - 1) // self._spr, R - 1
            ).astype(np.intp)
            self._sample_round_per_sat.append(all_rounds)
            start, stop, _ = sl.indices(n)
            idx = np.arange(start, stop, dtype=np.intp)              # (n_probe_k,)
            self._probe_idx_per_sat.append(idx)
            self._probe_round_per_sat.append(all_rounds[idx])

        # Communication per round: one model up and one model down per client
        self._param_count: int = sum(
            p.numel() for p in self._global_model.parameters()
        )
        self._bytes_per_model: int = self._param_count * _PARAM_BYTES

        # Region label per satellite for the ARI diagnostic (see _compute_region_labels)
        self._region_labels: np.ndarray = self._compute_region_labels()

    # --- Public interface ---

    def run(self) -> RunResult:
        """Run the full fixed-budget FL loop.

        Returns
        -------
        RunResult  Per-round metrics and run metadata.
        """
        rows: list[dict] = []
        rng = np.random.default_rng(self._cfg.seed)

        global_state: StateDict = {
            k: v.clone() for k, v in self._global_model.state_dict().items()
        }
        prev_membership: np.ndarray | None = None
        fl_state = FLState(round_idx=0, global_model=global_state, client_updates=[])

        # cluster_local_train and per_sample_eval need every round's membership before
        # training starts. The schedule is computed once, in round order, and then used
        # as the only source of membership, so a strategy that draws random numbers per
        # call (random_churn) is not advanced twice. This requires a strategy whose
        # membership does not depend on training state; similarity depends on the
        # previous round's updates and cannot be precomputed.
        schedule: np.ndarray | None = None
        if self._cfg.fl.per_sample_eval or self._cfg.fl.cluster_local_train:
            if type(self._strategy).__name__ == "SimilarityReactive":
                raise ValueError(
                    "fl.per_sample_eval / fl.cluster_local_train are unsupported for the "
                    "similarity arm (state-dependent membership cannot be precomputed)."
                )
            schedule = np.stack([
                self._strategy.membership(
                    r, FLState(round_idx=r, global_model=global_state, client_updates=[])
                )
                for r in range(self._cfg.fl.round_budget)
            ])  # (round_budget, N_leo) int

        # Per-cluster models, keyed by cluster id. They persist across rounds between two
        # global syncs, which lets clusters drift apart, and are reset to the global
        # model at every global sync. A cluster without a model (first round after a
        # sync, or a new cluster id) starts from global_state. With sync_interval == 1
        # every client therefore starts every round from the global model, and
        # hierarchical FedAvg gives the same result as flat FedAvg.
        cluster_models: dict[int, StateDict] = {}

        for r in range(self._cfg.fl.round_budget):
            # 1. Cluster membership: from the precomputed schedule if there is one,
            # otherwise from the strategy.
            membership: np.ndarray = (
                schedule[r] if schedule is not None
                else self._strategy.membership(r, fl_state)
            )

            # 2. Participants of this round
            trace_t = r * self._spr  # timestep at the start of this round
            assignment_t = self._trace.assignment_at(
                min(trace_t, self._T - 1)
            )  # (N_leo,) meo_id or -1

            # Eligible clients have a cluster (membership[k] != -1) and a visible aggregator
            # in the trace (assignment_t[k] != -1)
            eligible = np.where(
                (membership >= 0) & (assignment_t >= 0)
            )[0]

            if self._cfg.fl.participation_rate < 1.0 and len(eligible) > 0:
                n_select = max(1, int(len(eligible) * self._cfg.fl.participation_rate))
                eligible = rng.choice(eligible, size=n_select, replace=False)

            # 3. Each participant trains on all of its training data up to the end of this
            # round (cumulative window).
            round_start_sample = 0
            round_end_sample = max(0, (r + 1) * self._spr - self._W + 1)
            round_window = slice(round_start_sample, round_end_sample)

            updates: list[ClientUpdate] = []
            start_models: dict[int, StateDict] = {}  # sat_id -> model it trained from
            for k in eligible:
                client = self._clients[k]
                # Stay inside the training partition
                train_end = self._train_slices[k].stop
                eff_start = round_window.start
                eff_end = min(round_window.stop, train_end)
                if eff_end <= eff_start:
                    continue
                cid = int(membership[k])
                # cluster_local_train: keep only the samples collected while this client
                # belonged to its current cluster, according to the membership schedule.
                sample_indices: np.ndarray | None = None
                if self._cfg.fl.cluster_local_train:
                    rng_idx = np.arange(eff_start, eff_end, dtype=np.intp)
                    sample_rounds = self._sample_round_per_sat[k][rng_idx]
                    sample_indices = rng_idx[schedule[sample_rounds, k] == cid]
                    if sample_indices.size == 0:
                        continue
                # Train from the cluster's model, or from the global model if the cluster has
                # none yet.
                start_state = cluster_models.get(cid, global_state)
                update = client.train_local(
                    start_state,
                    slice(eff_start, eff_end),
                    device=self._device,
                    sample_indices=sample_indices,
                )
                if update.n_samples > 0:
                    updates.append(update)
                    start_models[update.sat_id] = start_state

            # 4. Aggregation: FedAvg within each cluster every round, FedAvg across
            #    clusters every sync_interval rounds.
            did_sync = False
            if updates:
                clusters: dict[int, list[ClientUpdate]] = {}
                for u in updates:
                    cid = int(membership[u.sat_id])
                    clusters.setdefault(cid, []).append(u)

                # Cluster-level FedAvg. The result is kept for the next round, so clients keep
                # training along their cluster's trajectory until the next global sync.
                for cid, ups in clusters.items():
                    cluster_models[cid] = fedavg(ups)

                if (r + 1) % self._cfg.fl.sync_interval == 0:
                    # Global sync: data-size-weighted merge of the clusters that trained this round.
                    # cluster_models is reset only after the metrics are logged, so the personalized
                    # metric can still score this round's cluster models.
                    global_state = hierarchical_fedavg(clusters)
                    did_sync = True

            # 5. Metrics
            n_cluster_changes = (
                int(np.sum(membership != prev_membership))
                if prev_membership is not None
                else 0
            )
            prev_membership = membership.copy()

            comm_up = len(updates) * self._bytes_per_model
            comm_down = len(eligible) * self._bytes_per_model  # model sent down to all

            # The global-model probe and the personalized probe (each satellite scored under
            # its own cluster's model) use the same samples. Probe evaluation dominates the
            # cost of a round, so it runs every eval_interval rounds and on the final round.
            # Other rounds log NaN in the probe columns; communication, participation and ARI
            # are logged every round.
            do_eval = (
                r % self._cfg.fl.eval_interval == 0
                or r == self._cfg.fl.round_budget - 1
            )
            if do_eval:
                probe_auc, probe_ba = self._evaluate_probe(global_state)
                if self._cfg.fl.per_sample_eval:
                    pers_auc, pers_ba = self._evaluate_personalized_per_sample(
                        cluster_models, schedule, global_state
                    )
                else:
                    pers_auc, pers_ba = self._evaluate_personalized(
                        cluster_models, membership, global_state
                    )
                # Flush the progress line so batch logs fill in while the job is running.
                print(
                    f"  [{type(self._strategy).__name__}] "
                    f"round {r + 1}/{self._cfg.fl.round_budget} "
                    f"ba_probe={probe_ba:.4f} ba_personalized={pers_ba:.4f}",
                    flush=True,
                )
            else:
                probe_auc = probe_ba = pers_auc = pers_ba = float("nan")

            # Diagnostic only. Two caveats:
            #  (1) self._region_labels is a snapshot at t = 0, so this measures agreement
            #      with the initial geography, not partition quality over the run (see
            #      _compute_region_labels).
            #  (2) satellites with membership < 0 are counted as cluster 0 instead of being
            #      excluded, which raises the agreement when many satellites have no
            #      aggregator in view.
            ari = float(
                adjusted_rand_score(self._region_labels, np.where(membership < 0, 0, membership))
            )

            rows.append({
                "round": r,
                "n_participating": len(updates),
                "n_eligible": len(eligible),
                "n_cluster_changes": n_cluster_changes,
                "auc_probe": probe_auc,
                "ba_probe": probe_ba,
                "auc_personalized": pers_auc,
                "ba_personalized": pers_ba,
                "comm_bytes_up": comm_up,
                "comm_bytes_down": comm_down,
                "overhead_rounds": self._strategy.overhead_rounds,
                "ari_vs_regions": ari,
                # Number of distinct clusters this round; -1 (unassigned) is not a cluster.
                "n_clusters": int(np.unique(membership[membership >= 0]).size),
            })

            # Reset the cluster models to the merged global model, now that this round's
            # personalized metric has been logged.
            if did_sync:
                cluster_models = {}

            # State passed to the strategy in the next round
            fl_state = FLState(
                round_idx=r + 1,
                global_model=global_state,
                client_updates=updates,
                round_start_models=start_models,
            )

        return RunResult(
            metrics=pd.DataFrame(rows),
            config=self._cfg,
            strategy_name=type(self._strategy).__name__,
            churn_rate=self._trace.churn_rate_per_orbit(),
        )

    # --- Helpers ---

    def _predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run the model currently loaded in self._global_model over a loader.

        The caller must have called load_state_dict / eval / to() beforehand.

        Returns
        -------
        (y_true, y_pred) : tuple[np.ndarray, np.ndarray]  Flattened targets and
            predicted probabilities. Both empty if the loader yields nothing.
        """
        y_true_list: list = []
        y_pred_list: list = []
        with torch.no_grad():
            for obs_win, pos_enc, target in loader:
                obs_win = obs_win.to(self._device)
                pos_enc = pos_enc.to(self._device)
                target_np = target.numpy()
                logits = self._global_model(
                    obs_win, pos_enc if self._cfg.model.use_position else None
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true_list.append(target_np)
                y_pred_list.append(probs)
        if not y_true_list:
            return np.array([]), np.array([])
        return np.concatenate(y_true_list).ravel(), np.concatenate(y_pred_list).ravel()

    def _evaluate_probe(self, global_state: StateDict) -> tuple[float, float]:
        """Evaluate the global model on the probe set. Returns (auc, balanced_accuracy)."""
        self._global_model.load_state_dict(global_state)
        self._global_model.to(self._device)
        self._global_model.eval()
        y_true, y_pred = self._predict(self._probe_loader)
        return auc(y_true, y_pred), balanced_accuracy(y_true, (y_pred >= 0.5).astype(float))

    def _evaluate_personalized(
        self,
        cluster_models: dict[int, "StateDict"],
        membership: np.ndarray,
        global_state: StateDict,
    ) -> tuple[float, float]:
        """Personalized (per-cluster) probe evaluation.

        Each satellite's probe samples are scored under its own cluster's current
        model, and the predictions are pooled across satellites. The merged global
        model hides differences between cluster models; this metric shows them, so it
        is the one that can register a benefit from clustering.

        Satellites whose cluster has no model yet, or whose membership is -1, are
        scored under the global model. The probe samples are the same as in
        _evaluate_probe, so the two balanced accuracies are directly comparable.

        Parameters
        ----------
        cluster_models : dict[int, StateDict]  Current per-cluster models.
        membership     : (N_leo,) int          This round's cluster id per satellite.
        global_state   : StateDict             Model used for clusters without a model.

        Returns
        -------
        (auc, balanced_accuracy) : tuple[float, float]  The global probe result if
            there are no probe samples.
        """
        # Group satellites by cluster id.
        groups: dict[int, list[int]] = {}
        for k in range(self._N):
            groups.setdefault(int(membership[k]), []).append(k)

        self._global_model.to(self._device)
        self._global_model.eval()

        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for cid, sats in groups.items():
            subsets = [
                self._probe_subset_per_sat[k]
                for k in sats
                if len(self._probe_subset_per_sat[k]) > 0
            ]
            if not subsets:
                continue
            state = cluster_models.get(cid, global_state)
            self._global_model.load_state_dict(state)
            loader = DataLoader(
                ConcatDataset(subsets), batch_size=512, shuffle=False, num_workers=0
            )
            y_true, y_pred = self._predict(loader)
            if y_true.size:
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

        if not y_true_all:
            return self._evaluate_probe(global_state)

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        return auc(y_true, y_pred), balanced_accuracy(y_true, (y_pred >= 0.5).astype(float))

    def _evaluate_personalized_per_sample(
        self,
        cluster_models: dict[int, "StateDict"],
        schedule: np.ndarray,
        global_state: StateDict,
    ) -> tuple[float, float]:
        """Personalized evaluation with per-sample routing (fl.per_sample_eval).

        Each probe sample is scored under the model of the cluster its satellite
        belonged to in that sample's round (schedule[round, sat]), instead of one model
        per satellite. With region clusters a satellite's samples are each scored under
        the model of the region they were collected in. With static clusters every
        sample of a satellite goes to the same cluster and the result equals the
        per-satellite metric. Satellites cross all regions during a run, so scoring a
        satellite under a single cluster model would average out any region-specific
        behaviour.

        The probe samples are the same as in _evaluate_probe. Returns the global probe
        result if there are no probe samples.

        Parameters
        ----------
        cluster_models : dict[int, StateDict]  Current per-cluster models.
        schedule       : (round_budget, N_leo) int  Per-round membership schedule.
        global_state   : StateDict             Model used for clusters without a model.

        Returns
        -------
        (auc, balanced_accuracy) : tuple[float, float]
        """
        # Group probe samples by the cluster they are routed to: cid -> per-satellite Subsets.
        groups: dict[int, list[Subset]] = {}
        for k in range(self._N):
            idx = self._probe_idx_per_sat[k]           # (n_probe_k,) dataset indices
            if idx.size == 0:
                continue
            cids = schedule[self._probe_round_per_sat[k], k]  # (n_probe_k,) cluster id
            ds = self._datasets[k]
            for cid in np.unique(cids):
                sel = idx[cids == cid]
                if sel.size:
                    groups.setdefault(int(cid), []).append(Subset(ds, sel.tolist()))

        self._global_model.to(self._device)
        self._global_model.eval()

        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for cid, subsets in groups.items():
            state = cluster_models.get(cid, global_state)
            self._global_model.load_state_dict(state)
            loader = DataLoader(
                ConcatDataset(subsets), batch_size=512, shuffle=False, num_workers=0
            )
            y_true, y_pred = self._predict(loader)
            if y_true.size:
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

        if not y_true_all:
            return self._evaluate_probe(global_state)

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        return auc(y_true, y_pred), balanced_accuracy(y_true, (y_pred >= 0.5).astype(float))

    def _compute_region_labels(self) -> np.ndarray:
        """Region label per LEO satellite, from the sub-satellite points at t = 0.

        Used for the ari_vs_regions diagnostic. The labels are a snapshot at t = 0,
        whereas region-based clustering assigns each satellite to the region currently
        under it. The two agree exactly at round 0, so ARI = 1.0 there by construction,
        and they diverge as the satellites move, agreeing again when the constellation
        returns to its initial phase. On the 180-satellite trace the per-round mean ARI
        of region clustering is about 0.04, against about 0.00 for plane and random
        churn: the mean does not separate the strategies, only the round-0 value and
        the range do. ari_vs_regions therefore shows that region clusters are
        geographic at the moment they are formed. It is not a measure of partition
        quality over a run; that would need the region under each satellite at every
        round.

        Returns
        -------
        np.ndarray  (N_leo,) int  Region id per satellite, evaluated at t = 0 only.
        """
        from orbitfl.world.regions import region_at

        region_map = self._world._region_map
        labels = np.zeros(self._N, dtype=np.int32)
        for k in range(self._N):
            subpts = self._trace.subpoints(k)
            if len(subpts) == 0:
                continue
            lat, lon = float(subpts[0, 0]), float(subpts[0, 1])
            labels[k] = region_at(lat, lon, region_map)
        return labels
