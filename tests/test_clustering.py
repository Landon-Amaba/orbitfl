"""Tests for src/orbitfl/fl/clustering/.

Every strategy returns a valid partition each round; the ephemeris schedule
matches the trace assignment; the plane partition never changes; the measured
churn of random_churn is within 10% of its target; similarity recovers planted
clusters from synthetic updates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from orbitfl.fl.aggregation import ClientUpdate
from orbitfl.fl.clustering.base import FLState
from orbitfl.fl.clustering.ephemeris import EphemerisClustering
from orbitfl.fl.clustering.plane import PlaneClustering
from orbitfl.fl.clustering.random_churn import RandomChurnClustering
from orbitfl.fl.clustering.similarity import SimilarityReactive

_GOLDEN = Path(__file__).parent / "golden" / "tiny_trace"
_NEEDS_TRACE = pytest.mark.skipif(
    not _GOLDEN.exists(), reason="golden tiny_trace fixture not found"
)


# --- Helpers ---


def _empty_state(n_leo: int) -> FLState:
    return FLState(round_idx=0, global_model={}, client_updates=[])


def _make_update(sat_id: int, n_samples: int, params: dict) -> ClientUpdate:
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()}
    return ClientUpdate(
        state_dict=sd, n_samples=n_samples, grad_norm=0.0,
        sat_id=sat_id, computed_at_t=0,
    )


# --- EphemerisClustering (arm a) ---


@_NEEDS_TRACE
class TestEphemerisClustering:
    @pytest.fixture
    def trace(self):
        from orbitfl.trace.reader import Trace
        return Trace.load(_GOLDEN)

    def test_returns_valid_partition(self, trace):
        arm = EphemerisClustering(trace, round_duration_s=60, round_budget=10)
        mem = arm.membership(0, _empty_state(trace.n_leo))
        assert mem.shape == (trace.n_leo,)
        valid = np.logical_or(mem == -1, (mem >= 0) & (mem < trace.n_meo))
        assert valid.all(), "Cluster ids outside {-1} ∪ [0, N_meo)"

    def test_schedule_matches_trace_assignment(self, trace):
        """At t=0 the MEO assignments must match the trace directly."""
        arm = EphemerisClustering(trace, round_duration_s=15, round_budget=5)
        mem = arm.membership(0, _empty_state(trace.n_leo))
        trace_asgn = trace.assignment_at(0)  # (N_leo,) meo_id or -1
        np.testing.assert_array_equal(mem, trace_asgn)

    def test_overhead_is_zero(self, trace):
        arm = EphemerisClustering(trace, round_duration_s=60, round_budget=5)
        assert arm.overhead_rounds == 0

    def test_round_beyond_budget_is_clipped(self, trace):
        arm = EphemerisClustering(trace, round_duration_s=60, round_budget=3)
        mem_last = arm.membership(2, _empty_state(trace.n_leo))
        mem_over = arm.membership(999, _empty_state(trace.n_leo))
        np.testing.assert_array_equal(mem_last, mem_over)

    def test_from_region_schedule(self, trace):
        N = trace.n_leo
        sched = np.zeros((5, N), dtype=np.int32)
        arm = EphemerisClustering.from_region_schedule(sched, round_budget=5)
        mem = arm.membership(2, _empty_state(N))
        assert (mem == 0).all()


# --- PlaneClustering (arm c) ---


@_NEEDS_TRACE
class TestPlaneClustering:
    @pytest.fixture
    def trace(self):
        from orbitfl.trace.reader import Trace
        return Trace.load(_GOLDEN)

    def test_returns_valid_partition(self, trace):
        arm = PlaneClustering(trace)
        mem = arm.membership(0, _empty_state(trace.n_leo))
        assert mem.shape == (trace.n_leo,)
        assert (mem >= 0).all(), "Plane clustering never assigns -1"

    def test_never_changes(self, trace):
        """Partition must be identical across all rounds."""
        arm = PlaneClustering(trace)
        state = _empty_state(trace.n_leo)
        m0 = arm.membership(0, state)
        for r in [1, 5, 20, 100]:
            np.testing.assert_array_equal(m0, arm.membership(r, state))

    def test_matches_trace_plane_ids(self, trace):
        arm = PlaneClustering(trace)
        mem = arm.membership(0, _empty_state(trace.n_leo))
        np.testing.assert_array_equal(mem, trace.leo_plane_ids)

    def test_overhead_is_zero(self, trace):
        assert PlaneClustering(trace).overhead_rounds == 0


# --- RandomChurnClustering (arm d) ---


@_NEEDS_TRACE
class TestRandomChurnClustering:
    @pytest.fixture
    def trace(self):
        from orbitfl.trace.reader import Trace
        return Trace.load(_GOLDEN)

    def test_returns_valid_partition(self, trace):
        arm = RandomChurnClustering(
            trace, n_leo=trace.n_leo, n_clusters=2,
            round_duration_s=60, seed=0,
        )
        mem = arm.membership(0, _empty_state(trace.n_leo))
        assert mem.shape == (trace.n_leo,)
        assert ((mem >= 0) & (mem < 2)).all()

    def test_overhead_is_zero(self, trace):
        arm = RandomChurnClustering(
            trace, n_leo=trace.n_leo, n_clusters=2, round_duration_s=60, seed=0
        )
        assert arm.overhead_rounds == 0

    def test_churn_rate_within_10_percent_of_target(self, trace):
        """
        Measure the per-round change fraction over many rounds and verify it
        falls within 10% of the arm's p_switch target.
        """
        N = trace.n_leo
        arm = RandomChurnClustering(
            trace, n_leo=N, n_clusters=3, round_duration_s=60, seed=42
        )
        target = arm.p_switch

        # Need at least a few switches to measure; skip if rate is effectively 0
        if target < 0.01:
            pytest.skip("p_switch < 0.01; too few changes to measure reliably")

        n_rounds = 2000
        state = _empty_state(N)
        prev = arm.membership(0, state).copy()
        change_fracs = []
        for r in range(1, n_rounds):
            cur = arm.membership(r, state)
            change_fracs.append(float((cur != prev).mean()))
            prev = cur.copy()

        measured = float(np.mean(change_fracs))
        tol = 0.10 * target + 0.02   # 10% relative + 2% absolute slack
        assert abs(measured - target) <= tol, (
            f"Churn rate mismatch: measured={measured:.4f}, "
            f"target={target:.4f}, tol={tol:.4f}"
        )

    def test_different_seeds_differ(self, trace):
        """Two seeds should produce different sequences."""
        a = RandomChurnClustering(
            trace, n_leo=trace.n_leo, n_clusters=3, round_duration_s=60, seed=0
        )
        b = RandomChurnClustering(
            trace, n_leo=trace.n_leo, n_clusters=3, round_duration_s=60, seed=1
        )
        state = _empty_state(trace.n_leo)
        mems_a = [a.membership(r, state).copy() for r in range(50)]
        mems_b = [b.membership(r, state).copy() for r in range(50)]
        # Very unlikely all 50 rounds match with different seeds
        assert not all(np.array_equal(ma, mb) for ma, mb in zip(mems_a, mems_b))


# --- SimilarityReactive (arm b) ---


class TestSimilarityReactive:
    def _make_state(self, n_leo: int, updates: list[ClientUpdate]) -> FLState:
        """Fake global model of zeros."""
        dim = sum(v.numel() for v in updates[0].state_dict.values()) if updates else 4
        global_model = {"w": torch.zeros(dim)}
        return FLState(round_idx=0, global_model=global_model, client_updates=updates)

    def test_returns_valid_partition_during_bootstrap(self):
        N, K = 8, 2
        arm = SimilarityReactive(n_leo=N, n_clusters=K, T_init=5, seed=0)
        state = _empty_state(N)
        mem = arm.membership(0, state)
        assert mem.shape == (N,)
        assert ((mem >= 0) & (mem < K)).all()

    def test_overhead_increments_after_bootstrap(self):
        N, K = 6, 2
        arm = SimilarityReactive(n_leo=N, n_clusters=K, T_init=3, seed=0)
        state = _empty_state(N)
        for r in range(3):
            arm.membership(r, state)
        # After T_init rounds, clustering fires once -> overhead = T_init
        assert arm.overhead_rounds == 3

    def test_overhead_is_zero_during_bootstrap(self):
        arm = SimilarityReactive(n_leo=4, n_clusters=2, T_init=5, seed=0)
        state = _empty_state(4)
        arm.membership(0, state)
        assert arm.overhead_rounds == 0

    def test_recovers_planted_clusters(self):
        """
        Two planted clusters: clients 0,1,2 share one gradient direction,
        clients 3,4,5 share the opposite.  After T_init bootstrap rounds the
        similarity arm should separate them into exactly 2 clusters.
        """
        N, K, D, T = 6, 2, 16, 5
        arm = SimilarityReactive(
            n_leo=N, n_clusters=K, T_init=T,
            gamma=0.5, d_th=0.5, seed=0
        )
        global_model = {"w": torch.zeros(D)}
        # Group A: gradient +1; Group B: gradient -1
        group_a_delta = [1.0] * D
        group_b_delta = [-1.0] * D

        for r in range(T):
            updates = [
                _make_update(i, 10, {"w": group_a_delta}) for i in range(3)
            ] + [
                _make_update(i, 10, {"w": group_b_delta}) for i in range(3, 6)
            ]
            state = FLState(round_idx=r, global_model=global_model, client_updates=updates)
            mem = arm.membership(r, state)

        # After bootstrap: mem[0]==mem[1]==mem[2] and mem[3]==mem[4]==mem[5]
        # and the two groups are in different clusters
        assert mem[0] == mem[1] == mem[2], "Group A not in same cluster"
        assert mem[3] == mem[4] == mem[5], "Group B not in same cluster"
        assert mem[0] != mem[3], "Groups A and B should be in different clusters"

    def test_reclusters_after_R_rounds(self):
        """With recluster_every_R=5, overhead after 10 rounds = 2 x T_init."""
        N, K = 6, 2
        T_init, R = 3, 5
        arm = SimilarityReactive(
            n_leo=N, n_clusters=K, T_init=T_init,
            recluster_every_R=R, seed=0
        )
        state = _empty_state(N)
        # First bootstrap: rounds 0,1,2 -> cluster at round 3
        # Second bootstrap starts at round 5: rounds 5,6,7 -> cluster at round 8
        for r in range(10):
            arm.membership(r, state)
        # Should have paid T_init overhead at least twice
        assert arm.overhead_rounds >= 2 * T_init

    def test_recluster_with_real_updates_does_not_crash(self):
        """Re-clustering with a finite recluster_every_R and real client updates.

        The accumulators are reset at the start of every bootstrap and must be
        allocated again when the next updates arrive. This runs several full
        re-clustering cycles with real gradients and checks that each one ends with a
        valid partition.
        """
        N, K, D, T_init, R = 6, 2, 16, 3, 5
        arm = SimilarityReactive(
            n_leo=N, n_clusters=K, T_init=T_init, recluster_every_R=R,
            gamma=0.5, d_th=0.5, seed=0,
        )
        global_model = {"w": torch.zeros(D)}
        group_a = [1.0] * D
        group_b = [-1.0] * D
        mem = None
        for r in range(20):  # spans multiple bootstrap/re-cluster cycles
            updates = (
                [_make_update(i, 10, {"w": group_a}) for i in range(3)]
                + [_make_update(i, 10, {"w": group_b}) for i in range(3, 6)]
            )
            state = FLState(round_idx=r, global_model=global_model, client_updates=updates)
            mem = arm.membership(r, state)

        assert arm.overhead_rounds >= 2 * T_init, "should have re-clustered ≥ twice"
        assert mem is not None and (mem >= 0).all(), "valid partition after re-clustering"


class TestSimilarityUpdateReference:
    """update_reference="round_start": updates measured against each client's start model."""

    _DIM = 8

    def _round(self, arm: SimilarityReactive, r: int, sats: list[int]) -> np.ndarray:
        """
        Drive one round: two groups move in opposite directions from a shared start.

        The shared start is far from the global model, as after several rounds of
        cluster training between syncs.
        """
        shared = torch.full((self._DIM,), 100.0)  # (dim,)
        direction = torch.tensor([1.0, -1.0] * (self._DIM // 2))  # (dim,)
        updates, starts = [], {}
        for k in sats:
            sign = 1.0 if k < 3 else -1.0
            final = (shared + sign * direction).tolist()
            updates.append(_make_update(k, 10, {"w": final}))
            starts[k] = {"w": shared.clone()}
        state = FLState(
            round_idx=r,
            global_model={"w": torch.zeros(self._DIM)},
            client_updates=updates,
            round_start_models=starts,
        )
        return arm.membership(r, state)

    def _arm(self, reference: str) -> SimilarityReactive:
        """Six clients, clustering after three rounds, d_th below 1."""
        return SimilarityReactive(
            n_leo=6, n_clusters=2, T_init=3, d_th=0.5, seed=0, update_reference=reference
        )

    def test_round_start_separates_what_last_global_merges(self):
        """A large shared drift hides opposing updates unless it is subtracted."""
        for reference, expected in (("last_global", 1), ("round_start", 2)):
            arm = self._arm(reference)
            for r in range(3):
                mem = self._round(arm, r, list(range(6)))
            assert len(set(mem.tolist())) == expected, reference

    def test_round_start_partition_matches_groups(self):
        """The two opposing groups land in two clusters, with no mixing."""
        arm = self._arm("round_start")
        for r in range(3):
            mem = self._round(arm, r, list(range(6)))
        assert len(set(mem[:3].tolist())) == 1
        assert len(set(mem[3:].tolist())) == 1
        assert mem[0] != mem[3]

    def test_absent_client_keeps_its_last_update(self):
        """Sitting out the clustering round does not make a client a singleton."""
        arm = self._arm("round_start")
        self._round(arm, 0, list(range(6)))
        self._round(arm, 1, list(range(6)))
        mem = self._round(arm, 2, [0, 1, 3, 4, 5])  # client 2 absent
        assert len(set(mem.tolist())) == 2
        assert mem[2] == mem[0]

    def test_missing_start_model_raises(self):
        """A round_start run with a missing start model raises instead of using another reference.
        """
        arm = self._arm("round_start")
        state = FLState(
            round_idx=0,
            global_model={"w": torch.zeros(self._DIM)},
            client_updates=[_make_update(0, 10, {"w": [1.0] * self._DIM})],
        )
        with pytest.raises(ValueError, match="start model"):
            arm.membership(0, state)

    def test_unknown_reference_raises(self):
        """Typos in a sweep config fail at construction."""
        with pytest.raises(ValueError, match="update_reference"):
            SimilarityReactive(n_leo=4, n_clusters=2, update_reference="global")
