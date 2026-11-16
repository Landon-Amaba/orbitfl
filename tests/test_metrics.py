"""Tests for src/orbitfl/metrics/allocation.py.

- jain_fairness of an equal allocation is 1.
- collision_rate on a hand-built 3-node example matches the value worked out by hand.
- normalized_churn is 1.0 when reassignments equal topology-change events.
- boundary_collision_rate and interior_collision_rate split nodes by edge membership.
- spectral_efficiency is 0 when all assigned channels are occupied.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from orbitfl.metrics.allocation import (
    boundary_collision_rate,
    collision_rate,
    interior_collision_rate,
    jain_fairness,
    normalized_churn,
    spectral_efficiency,
)
from orbitfl.metrics.efficiency import comm_to_target, rounds_to_target

# --- jain_fairness ---

def test_jain_equal_allocation_is_one():
    """Jain's index is 1 when all allocations are equal."""
    assert jain_fairness([1, 1, 1, 1]) == pytest.approx(1.0)
    assert jain_fairness([3, 3, 3]) == pytest.approx(1.0)


def test_jain_all_zeros_is_one():
    """All-zero allocations -> trivially fair (nobody served)."""
    assert jain_fairness([0, 0, 0]) == pytest.approx(1.0)


def test_jain_single_node():
    """Single node -> degenerate; returns 1.0."""
    assert jain_fairness([5]) == pytest.approx(1.0)


def test_jain_one_node_all_resource():
    """One node gets all resource -> J = 1/n."""
    x = [1, 0, 0, 0]
    expected = 1 / 4
    assert jain_fairness(x) == pytest.approx(expected)


def test_jain_monotone_in_fairness():
    """More equal allocations give higher Jain index."""
    j_equal = jain_fairness([1, 1, 1, 1])
    j_skewed = jain_fairness([2, 1, 1, 0])
    j_extreme = jain_fairness([4, 0, 0, 0])
    assert j_equal > j_skewed > j_extreme


# --- collision_rate ---

def test_collision_rate_hand_built_3_nodes():
    """
    3-node example: nodes 0, 1, 2 assigned channels 0, 1, 2.
    Ground-truth: channel 0 is occupied for node 0, channels 1 and 2 are free.
    Expected collision rate: 1/3.
    """
    assignments = {0: 0, 1: 1, 2: 2}
    occupancy = np.zeros((3, 4), dtype=int)
    occupancy[0, 0] = 1  # node 0, channel 0 -> occupied (collision)

    rate = collision_rate(assignments, occupancy)
    assert rate == pytest.approx(1 / 3)


def test_collision_rate_no_collisions():
    """Zero collision rate when all assigned channels are free."""
    assignments = {0: 0, 1: 1}
    occupancy = np.zeros((2, 4), dtype=int)
    assert collision_rate(assignments, occupancy) == pytest.approx(0.0)


def test_collision_rate_all_collide():
    """Collision rate = 1 when all served channels are occupied."""
    assignments = {0: 0, 1: 1}
    occupancy = np.ones((2, 4), dtype=int)
    assert collision_rate(assignments, occupancy) == pytest.approx(1.0)


def test_collision_rate_unserved_excluded():
    """Unserved nodes (channel=None) do not count toward collision rate."""
    assignments = {0: None, 1: None, 2: 0}
    occupancy = np.zeros((3, 4), dtype=int)
    # Only node 2 is served; its channel 0 is free -> rate = 0
    assert collision_rate(assignments, occupancy) == pytest.approx(0.0)


def test_collision_rate_nan_when_no_served():
    """Returns NaN when no node is served."""
    assignments = {0: None, 1: None}
    occupancy = np.zeros((2, 4), dtype=int)
    assert math.isnan(collision_rate(assignments, occupancy))


# --- normalized_churn ---

def test_normalized_churn_one_to_one():
    """1.0 when every geometry change triggers exactly one reassignment."""
    assert normalized_churn(10, 10) == pytest.approx(1.0)


def test_normalized_churn_zero_reassignments():
    """0.0 when no channels were reassigned (stable allocation)."""
    assert normalized_churn(0, 5) == pytest.approx(0.0)
    assert normalized_churn(0, 0) == pytest.approx(0.0)


def test_normalized_churn_stable_graph_with_churn():
    """inf when reassignments > 0 but graph topology is unchanged."""
    assert normalized_churn(3, 0) == float("inf")


def test_normalized_churn_fractional():
    """Fractional result when reassignments < geometry changes."""
    assert normalized_churn(5, 10) == pytest.approx(0.5)


# --- spectral_efficiency ---

def test_spectral_efficiency_all_collide_is_zero():
    """SE = 0 when every assigned channel is occupied."""
    assignments = {0: 0, 1: 1}
    sinr = {0: 20.0, 1: 20.0}
    occupancy = np.ones((2, 4), dtype=int)
    assert spectral_efficiency(assignments, sinr, occupancy) == pytest.approx(0.0)


def test_spectral_efficiency_all_free():
    """SE > 0 when all assigned channels are free."""
    assignments = {0: 0}
    sinr = {0: 0.0}  # SNR = 0 dB -> C = log2(2) = 1 bit/s/Hz
    occupancy = np.zeros((1, 4), dtype=int)
    se = spectral_efficiency(assignments, sinr, occupancy)
    # Mean over 1 node; that node achieves log2(1 + 1) = 1.0 bits/s/Hz
    assert se == pytest.approx(1.0)


def test_spectral_efficiency_empty():
    """Empty assignments -> 0."""
    assert spectral_efficiency({}, {}, np.zeros((0, 4))) == pytest.approx(0.0)


# --- boundary_collision_rate / interior_collision_rate ---

def test_boundary_interior_partition():
    """
    5 nodes; boundary edge (1, 2); nodes 0,3,4 are interior.
    Nodes 0 and 1 have collisions; node 1 is on the boundary, node 0 is interior.
    boundary_collision_rate should reflect node 1's collision;
    interior_collision_rate should reflect node 0's collision.
    """
    # Nodes 0,1,2,3,4; boundary edge = (1, 2)
    b_edges = [(1, 2)]
    # All assigned channel 0
    assignments = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    occupancy = np.zeros((5, 4), dtype=int)
    occupancy[0, 0] = 1  # node 0 collides (interior)
    occupancy[1, 0] = 1  # node 1 collides (boundary)
    # node 2 is on boundary but free
    # nodes 3,4 are interior and free

    bcr = boundary_collision_rate(assignments, b_edges, occupancy)
    icr = interior_collision_rate(assignments, b_edges, occupancy)

    # Boundary nodes: {1, 2}. node 1 collides, node 2 free -> bcr = 1/2
    assert bcr == pytest.approx(0.5)
    # Interior nodes: {0, 3, 4}. node 0 collides -> icr = 1/3
    assert icr == pytest.approx(1 / 3)


def test_interior_nan_when_no_interior_nodes():
    """interior_collision_rate returns NaN if all nodes are on boundary edges."""
    b_edges = [(0, 1)]
    assignments = {0: 0, 1: 1}
    occupancy = np.zeros((2, 4), dtype=int)
    assert math.isnan(interior_collision_rate(assignments, b_edges, occupancy))


# --- efficiency: rounds_to_target / comm_to_target ---


def _eff_df() -> pd.DataFrame:
    """
    6-round run with eval_interval=2 (ba_probe NaN on odd rounds) and 2 GB comm/round.

    ba_probe: r0=0.50, r2=0.56, r4=0.60 (r1,r3,r5 = NaN).
    comm_bytes_up = comm_bytes_down = 1e9 per round -> 2 GB/round.
    """
    return pd.DataFrame(
        {
            "round": [0, 1, 2, 3, 4, 5],
            "ba_probe": [0.50, np.nan, 0.56, np.nan, 0.60, np.nan],
            "comm_bytes_up": [1e9] * 6,
            "comm_bytes_down": [1e9] * 6,
        }
    )


def test_rounds_to_target_first_eval_crossing():
    """First evaluated round whose ba_probe >= target (NaN rounds ignored)."""
    df = _eff_df()
    assert rounds_to_target(df, 0.55) == 2.0  # 0.56 at round 2
    assert rounds_to_target(df, 0.60) == 4.0
    assert rounds_to_target(df, 0.50) == 0.0


def test_rounds_to_target_nan_when_never_reached():
    assert math.isnan(rounds_to_target(_eff_df(), 0.99))


def test_comm_to_target_sums_all_rounds_through_crossing():
    """Comm is logged every round, so cumulative through the crossing round is exact."""
    df = _eff_df()
    # crossing at round 2 -> rounds 0,1,2 counted -> 3 rounds x 2 GB = 6 GB
    assert comm_to_target(df, 0.55) == pytest.approx(6.0)
    # crossing at round 4 -> rounds 0..4 -> 5 rounds x 2 GB = 10 GB
    assert comm_to_target(df, 0.60) == pytest.approx(10.0)


def test_comm_to_target_nan_when_never_reached():
    assert math.isnan(comm_to_target(_eff_df(), 0.99))
