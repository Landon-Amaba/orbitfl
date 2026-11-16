"""Tests for src/orbitfl/fl/aggregation.py.

- hierarchical_fedavg equals fedavg to 1e-6 under full participation with one
  local training phase.
- Data-size weighting is correct on unequal splits.
- A round with no participants keeps the previous global state.
"""

from __future__ import annotations

import random

import pytest
import torch

from orbitfl.fl.aggregation import (
    ClientUpdate,
    fedavg,
    hierarchical_fedavg,
    staleness_weight,
)

# --- Helpers ---


def _make_update(
    sat_id: int,
    n_samples: int,
    params: dict[str, list[float]],
    computed_at_t: int = 0,
) -> ClientUpdate:
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()}
    return ClientUpdate(
        state_dict=state,
        n_samples=n_samples,
        grad_norm=0.0,
        sat_id=sat_id,
        computed_at_t=computed_at_t,
    )


def _assert_hierarchical_equals_flat(
    all_updates: list[ClientUpdate],
    clusters: dict[int, list[ClientUpdate]],
    tol: float = 1e-6,
) -> None:
    flat = fedavg(all_updates)
    hier = hierarchical_fedavg(clusters)
    for key in flat:
        diff = float((flat[key] - hier[key]).abs().max().item())
        assert diff <= tol, (
            f"Mismatch on param '{key}': max |flat − hier| = {diff:.2e} > {tol:.0e}. "
            "Determine whether the implementation or the proof is wrong."
        )


# --- fedavg ---


class TestFedAvg:
    def test_equal_weights(self):
        u1 = _make_update(0, 10, {"w": [1.0, 2.0]})
        u2 = _make_update(1, 10, {"w": [3.0, 4.0]})
        avg = fedavg([u1, u2])
        torch.testing.assert_close(avg["w"], torch.tensor([2.0, 3.0]))

    def test_unequal_weights(self):
        """(1x0 + 2x3) / 3 = 2.0"""
        u1 = _make_update(0, 1, {"w": [0.0]})
        u2 = _make_update(1, 2, {"w": [3.0]})
        torch.testing.assert_close(fedavg([u1, u2])["w"], torch.tensor([2.0]))

    def test_single_client(self):
        u = _make_update(0, 5, {"a": [1.0, 2.0], "b": [3.0]})
        avg = fedavg([u])
        torch.testing.assert_close(avg["a"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(avg["b"], torch.tensor([3.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fedavg([])

    def test_zero_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples == 0"):
            fedavg([_make_update(0, 0, {"w": [1.0]})])

    def test_multiple_param_keys(self):
        u1 = _make_update(0, 2, {"l1": [1.0], "l2": [2.0]})
        u2 = _make_update(1, 2, {"l1": [3.0], "l2": [4.0]})
        avg = fedavg([u1, u2])
        torch.testing.assert_close(avg["l1"], torch.tensor([2.0]))
        torch.testing.assert_close(avg["l2"], torch.tensor([3.0]))


# --- hierarchical == flat ---


class TestHierarchicalEqualsFlat:
    """
    hierarchical_fedavg == fedavg under full participation, one local phase.

    Tolerance: <= 1e-6 absolute max deviation across all parameters.
    Do not relax the tolerance - if this fails, fix the proof or the code.
    """

    def test_two_equal_clusters(self):
        updates = [
            _make_update(0, 10, {"w": [1.0, 0.0]}),
            _make_update(1, 10, {"w": [0.0, 1.0]}),
            _make_update(2, 10, {"w": [2.0, 3.0]}),
            _make_update(3, 10, {"w": [4.0, 5.0]}),
        ]
        _assert_hierarchical_equals_flat(updates, {0: updates[:2], 1: updates[2:]})

    def test_unequal_sizes_two_clusters(self):
        updates = [
            _make_update(0, 3,  {"w": [1.0]}),
            _make_update(1, 7,  {"w": [2.0]}),
            _make_update(2, 5,  {"w": [3.0]}),
            _make_update(3, 15, {"w": [4.0]}),
        ]
        _assert_hierarchical_equals_flat(updates, {0: updates[:2], 1: updates[2:]})

    def test_three_clusters_different_sizes(self):
        u = [_make_update(k, (k + 1) * 17, {"a": [float(k)], "b": [float(k) * 2]})
             for k in range(6)]
        _assert_hierarchical_equals_flat(u, {0: u[:1], 1: u[1:3], 2: u[3:]})

    def test_single_cluster_is_fedavg(self):
        updates = [_make_update(k, k + 1, {"w": [float(k)]}) for k in range(5)]
        _assert_hierarchical_equals_flat(updates, {0: updates})

    def test_one_client_per_cluster(self):
        updates = [_make_update(k, 10, {"w": [float(k)]}) for k in range(4)]
        _assert_hierarchical_equals_flat(updates, {k: [updates[k]] for k in range(4)})

    def test_high_dimensional_params(self):
        rng = random.Random(42)
        D = 1000
        updates = [
            _make_update(k, rng.randint(1, 100), {"w": [rng.gauss(0, 1) for _ in range(D)]})
            for k in range(10)
        ]
        _assert_hierarchical_equals_flat(updates, {0: updates[:3], 1: updates[3:7], 2: updates[7:]})

    def test_many_clusters_many_clients(self):
        """50 clients, 5 clusters, random sizes - stress test."""
        rng = random.Random(7)
        D = 64
        updates = [
            _make_update(k, rng.randint(1, 200), {"w": [rng.gauss(0, 1) for _ in range(D)]})
            for k in range(50)
        ]
        clusters = {c: [updates[k] for k in range(50) if k % 5 == c] for c in range(5)}
        _assert_hierarchical_equals_flat(updates, clusters)


# --- hierarchical_fedavg: additional invariants ---


class TestHierarchicalFedAvg:
    def test_empty_clusters_raises(self):
        with pytest.raises(ValueError, match="empty"):
            hierarchical_fedavg({})

    def test_empty_cluster_lists_skipped(self):
        u = _make_update(0, 5, {"w": [1.0]})
        result = hierarchical_fedavg({0: [u], 1: []})
        torch.testing.assert_close(result["w"], torch.tensor([1.0]))

    def test_all_empty_raises(self):
        with pytest.raises(ValueError):
            hierarchical_fedavg({0: [], 1: []})


# --- staleness_weight ---


class TestStalenessWeight:
    def test_zero_staleness_is_one(self):
        assert staleness_weight(0) == pytest.approx(1.0)

    def test_one_round(self):
        assert staleness_weight(1) == pytest.approx(1.0 / 4.0)

    def test_large_staleness_small(self):
        assert staleness_weight(100) < 0.01

    def test_monotone_decreasing(self):
        vals = [staleness_weight(d) for d in range(10)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_custom_zeta(self):
        assert staleness_weight(1, zeta=1.0) == pytest.approx(0.5)

    def test_zero_zeta(self):
        """zeta=0 -> always 1.0 regardless of staleness."""
        for d in range(5):
            assert staleness_weight(d, zeta=0.0) == pytest.approx(1.0)
