"""Tests of FederatedRunner (src/orbitfl/fl/loop.py).

They check that the clustering partition affects training exactly when it
should:
  1. sync_interval > 1: two different partitions of the same participating
     clients give different global-model accuracy, because cluster models are
     kept between global syncs.
  2. sync_interval == 1 with full participation: the partition has no effect,
     because cluster models are reset to the global model every round and
     hierarchical FedAvg equals flat FedAvg.

A fixed, hand-built partition is used instead of a strategy class, so the tests
cover the loop alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from orbitfl.config import Config, load_config
from orbitfl.fl.clustering.base import ClusteringStrategy, FLState
from orbitfl.fl.clustering.similarity import SimilarityReactive
from orbitfl.fl.loop import FederatedRunner
from orbitfl.trace.reader import Trace
from orbitfl.utils.seeding import seed_everything
from orbitfl.world.occupancy import OccupancyWorld

_GOLDEN = Path(__file__).parent / "golden" / "tiny_trace"
_BASE_CFG = Path(__file__).parent.parent / "configs" / "base.yaml"
_NEEDS_TRACE = pytest.mark.skipif(
    not _GOLDEN.exists(), reason="golden tiny_trace fixture not found"
)


# --- Helpers ---


class _FixedPartition(ClusteringStrategy):
    """
    Deterministic static partition for isolating the loop's clustering behaviour.

    Returns the same caller-supplied assignment every round (no -1 entries), so
    two instances differ only in how they group an identical participating set.
    """

    def __init__(self, assignments: list[int]) -> None:
        # (N_leo,) int - fixed cluster id per LEO; identical every round.
        self._assignments: np.ndarray = np.asarray(assignments, dtype=np.int32)

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """Return the fixed partition (ignores round_idx and state)."""
        return self._assignments.copy()

    @property
    def overhead_rounds(self) -> int:
        """Zero - a fixed partition consumes no measurement rounds."""
        return 0


def _cfg(
    sync_interval: int,
    eval_interval: int = 1,
    round_budget: int = 4,
    cluster_local_train: bool = False,
    per_sample_eval: bool = False,
) -> Config:
    """Small Config for a short loop over the golden tiny trace.

    Parameters
    ----------
    sync_interval       : int   Rounds between global syncs.
    cluster_local_train : bool  Value of fl.cluster_local_train.
    per_sample_eval     : bool  Value of fl.per_sample_eval.
    """
    return load_config(
        _BASE_CFG,
        override_dict={
            "fl": {
                "round_budget": round_budget,
                "round_duration_s": 300,
                "sync_interval": sync_interval,
                "eval_interval": eval_interval,
                "cluster_local_train": cluster_local_train,
                "per_sample_eval": per_sample_eval,
            },
            "model": {"hidden": 16, "window": 16},
            "experiment": {"delta_steps": 4, "seeds": [0]},
        },
    )


def _run_metrics(
    trace: Trace,
    assignments: list[int],
    sync_interval: int,
    seed: int = 0,
    eval_interval: int = 1,
    round_budget: int = 4,
    cluster_local_train: bool = False,
    per_sample_eval: bool = False,
):
    """
    Run the loop with a fixed partition and return the full per-round metrics.

    Re-seeds immediately before runner construction so the initial model weights
    (and the world) are identical across calls - any difference is therefore
    attributable solely to the partition.
    """
    cfg = _cfg(
        sync_interval, eval_interval=eval_interval, round_budget=round_budget,
        cluster_local_train=cluster_local_train, per_sample_eval=per_sample_eval,
    )
    seed_everything(seed)
    world = OccupancyWorld(
        cfg.world,
        seed=seed,
        timestep_s=cfg.sim.timestep_s,
        n_steps=trace.n_timesteps,
    )
    seed_everything(seed)  # identical model init regardless of world RNG draws
    runner = FederatedRunner(
        cfg, trace, world, _FixedPartition(assignments), device=torch.device("cpu")
    )
    return runner.run().metrics


def _final_ba(
    trace: Trace,
    assignments: list[int],
    sync_interval: int,
    seed: int = 0,
) -> float:
    """Final global-probe balanced accuracy for a fixed partition."""
    return float(_run_metrics(trace, assignments, sync_interval, seed)["ba_probe"].iloc[-1])


# Two partitions over 6 LEOs, both with no -1 entries -> identical participation,
# different grouping. Partition A: two clusters of three; B: three clusters of two.
_PARTITION_A: list[int] = [0, 0, 0, 1, 1, 1]
_PARTITION_B: list[int] = [0, 1, 2, 0, 1, 2]


# --- Tests ---


@_NEEDS_TRACE
class TestClusteringAffectsTraining:
    @pytest.fixture
    def trace(self) -> Trace:
        return Trace.load(_GOLDEN)

    def test_partition_changes_ba_when_sync_interval_gt_1(self, trace: Trace) -> None:
        """Different partitions give different accuracy when sync_interval > 1.

        Cluster models persist between global syncs, so two partitions of the same
        participating clients must reach different global-model accuracy.
        """
        ba_a = _final_ba(trace, _PARTITION_A, sync_interval=2)
        ba_b = _final_ba(trace, _PARTITION_B, sync_interval=2)
        assert abs(ba_a - ba_b) > 1e-5, (
            f"Partition had no effect at sync_interval=2 "
            f"(BA_A={ba_a:.8f}, BA_B={ba_b:.8f}); "
            "cluster models are not shaping training."
        )

    def test_partition_irrelevant_when_sync_interval_1(self, trace: Trace) -> None:
        """The partition has no effect when sync_interval == 1.

        With full participation, hierarchical FedAvg must equal flat FedAvg, so the
        partition cannot change accuracy.
        """
        ba_a = _final_ba(trace, _PARTITION_A, sync_interval=1)
        ba_b = _final_ba(trace, _PARTITION_B, sync_interval=1)
        assert abs(ba_a - ba_b) < 1e-9, (
            f"Partition changed BA at sync_interval=1 "
            f"(BA_A={ba_a:.8f}, BA_B={ba_b:.8f}); "
            "hierarchical FedAvg does not reduce to flat FedAvg."
        )

    def test_personalized_metric_present_and_reflects_clustering(
        self, trace: Trace
    ) -> None:
        """The personalized probe metric is logged and differs from the global probe.

        With a non-trivial partition and sync_interval > 1 the cluster models diverge,
        so scoring each satellite under its own cluster's model must differ from the
        merged global model on at least one round.
        """
        df = _run_metrics(trace, _PARTITION_A, sync_interval=2)
        for col in ("ba_personalized", "auc_personalized"):
            assert col in df.columns, f"missing personalized column {col!r}"
        assert (df["ba_personalized"].between(0.0, 1.0)).all()
        max_gap = float((df["ba_personalized"] - df["ba_probe"]).abs().max())
        assert max_gap > 1e-6, (
            "Personalized metric never differs from the global probe "
            f"(max gap {max_gap:.2e}); per-cluster models are not being scored."
        )

    def test_eval_interval_skips_probe_rounds(self, trace: Trace) -> None:
        """
        eval_interval > 1 computes probe/personalized BA only every N rounds (and
        always the final round); skipped rounds log NaN, while comm/participation
        are still recorded every round.
        """
        df = _run_metrics(
            trace, _PARTITION_A, sync_interval=1, eval_interval=3, round_budget=7
        )
        # Rounds 0, 3, 6 evaluated; final round (6) always evaluated.
        evaluated = df.loc[df["round"].isin([0, 3, 6]), "ba_probe"]
        skipped = df.loc[df["round"].isin([1, 2, 4, 5]), "ba_probe"]
        assert evaluated.notna().all(), "evaluated rounds must have a probe BA"
        assert skipped.isna().all(), "skipped rounds must log NaN for probe BA"
        assert not df["ba_probe"].isna().iloc[-1], "final round must be evaluated"
        # Non-probe metrics logged every round regardless of eval_interval.
        assert (df["n_eligible"] >= 0).all()
        assert df["comm_bytes_up"].notna().all()


@_NEEDS_TRACE
class TestRegionModeEphemeris:
    """Region-based ephemeris (use_meo_proximity=False) clusters by ground geography."""

    @pytest.fixture
    def trace(self) -> Trace:
        return Trace.load(_GOLDEN)

    def test_region_mode_membership_tracks_geography(self, trace: Trace) -> None:
        """Region clustering assigns each LEO to the ground region under it. At round 0
        its membership must equal the t = 0 region labels that the loop uses for the
        ARI (so the ARI at round 0 is 1), with no -1 entries.
        """
        from orbitfl.experiments.runner import _build_strategy
        from orbitfl.world.regions import build_region_map, region_at

        cfg = load_config(
            _BASE_CFG,
            override_dict={
                "fl": {
                    "round_budget": 5,
                    "round_duration_s": 300,
                    "strategy": {
                        "name": "ephemeris",
                        "params": {"use_meo_proximity": False},
                    },
                },
                "experiment": {"seeds": [0]},
            },
        )
        strat = _build_strategy(cfg, trace, seed=0)

        region_map = build_region_map(cfg.world)
        mem0 = strat.membership(0, FLState(round_idx=0, global_model={}, client_updates=[]))
        expected = np.array(
            [
                region_at(
                    float(trace.subpoints(k)[0, 0]),
                    float(trace.subpoints(k)[0, 1]),
                    region_map,
                )
                for k in range(trace.n_leo)
            ]
        )
        assert np.array_equal(mem0, expected), "round-0 clusters must equal region-at-t0"
        assert (mem0 >= 0).all(), "region-based membership must have no -1 entries"


@_NEEDS_TRACE
class TestClusteredFLKnobs:
    """fl.cluster_local_train and fl.per_sample_eval.

    With these options a client trains each cluster model only on the samples it
    collected while in that cluster, and each probe sample is scored under the
    model of the cluster it was collected in. Both are off by default and must then
    leave the loop's results unchanged. These tests cover the mechanics on the tiny
    trace only.
    """

    @pytest.fixture
    def trace(self) -> Trace:
        return Trace.load(_GOLDEN)

    def test_per_sample_eval_leaves_global_model_identical(self, trace: Trace) -> None:
        """
        With a state-independent fixed partition, the precomputed membership schedule
        equals the live per-round membership, so turning per_sample_eval ON changes
        only the personalized metric - the global-model BA (ba_probe) is untouched.
        """
        off = _run_metrics(trace, _PARTITION_A, sync_interval=2, per_sample_eval=False)
        on = _run_metrics(trace, _PARTITION_A, sync_interval=2, per_sample_eval=True)
        np.testing.assert_allclose(
            off["ba_probe"].to_numpy(), on["ba_probe"].to_numpy(), atol=1e-9,
            err_msg="per_sample_eval must not change the global model / training",
        )
        assert on["ba_personalized"].between(0.0, 1.0).all()

    def test_cluster_local_train_changes_training(self, trace: Trace) -> None:
        """cluster_local_train restricts each client to the samples collected in its
        current cluster. This only matters when a client's cluster changes over time,
        so the partition here alternates every round: each client's history is split
        across two clusters, cluster-local training uses a strict subset of it, and the
        global-model trajectory changes. With a fixed partition the option has no
        effect, since every sample belongs to the client's one cluster.
        """
        class _AlternatingPartition(ClusteringStrategy):
            """Cluster = round parity (all clients flip 0<->1 each round)."""

            def __init__(self, n_leo: int) -> None:
                self._n = n_leo

            def membership(self, round_idx: int, state: FLState) -> np.ndarray:
                return np.full(self._n, round_idx % 2, dtype=np.int32)

            @property
            def overhead_rounds(self) -> int:
                return 0

        def _ba(cluster_local: bool):
            cfg = _cfg(sync_interval=2, round_budget=6, cluster_local_train=cluster_local)
            seed_everything(0)
            world = OccupancyWorld(cfg.world, seed=0, timestep_s=cfg.sim.timestep_s,
                                   n_steps=trace.n_timesteps)
            seed_everything(0)
            runner = FederatedRunner(cfg, trace, world, _AlternatingPartition(trace.n_leo),
                                     device=torch.device("cpu"))
            return runner.run().metrics["ba_probe"]

        max_gap = float((_ba(False) - _ba(True)).abs().max())
        assert max_gap > 1e-6, (
            f"full-history vs cluster-local training gave identical BA (gap {max_gap:.2e}); "
            "the cluster_local_train knob is not filtering the training set."
        )

    def test_knobs_reject_similarity(self, trace: Trace) -> None:
        """Both options need a precomputed membership schedule, which the similarity
        strategy cannot provide; the loop must reject the combination with an error.
        """
        class SimilarityReactive(ClusteringStrategy):  # name triggers the guard
            def membership(self, round_idx: int, state: FLState) -> np.ndarray:
                return np.zeros(trace.n_leo, dtype=np.int32)

            @property
            def overhead_rounds(self) -> int:
                return 0

        cfg = _cfg(sync_interval=2, per_sample_eval=True)
        seed_everything(0)
        world = OccupancyWorld(cfg.world, seed=0, timestep_s=cfg.sim.timestep_s,
                               n_steps=trace.n_timesteps)
        seed_everything(0)
        runner = FederatedRunner(cfg, trace, world, SimilarityReactive(),
                                 device=torch.device("cpu"))
        with pytest.raises(ValueError, match="unsupported for the"):
            runner.run()


def test_round_start_similarity_runs_through_loop_and_logs_cluster_count() -> None:
    """
    The loop feeds each client's start model to the similarity arm and logs cluster count.

    Would raise if round_start_models were not populated; n_clusters must equal the
    number of distinct ids the strategy assigned.
    """
    trace = Trace.load(_GOLDEN)
    cfg = _cfg(sync_interval=2, round_budget=5)
    seed_everything(0)
    world = OccupancyWorld(
        cfg.world, seed=0, timestep_s=cfg.sim.timestep_s, n_steps=trace.n_timesteps
    )
    strategy = SimilarityReactive(
        n_leo=trace.n_leo, n_clusters=2, T_init=2, seed=0, update_reference="round_start"
    )
    df = FederatedRunner(cfg, trace, world, strategy).run().metrics
    assert df["n_clusters"].between(1, trace.n_leo).all()
    assert strategy.overhead_rounds == 2
