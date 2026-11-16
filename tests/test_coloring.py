"""Tests for src/orbitfl/allocation/coloring.py.

- No adjacent nodes share a channel, checked over random graphs.
- With plenty of channels every node is served.
- One channel on two connected nodes serves exactly one of them.
- Channels reserved by incumbent_mask are never assigned.
- random_allocator and greedy_threshold_allocator produce proper colourings.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from orbitfl.allocation.coloring import (
    allocate,
    greedy_threshold_allocator,
    random_allocator,
)

# --- Helpers ---

def _uniform_p_hat(n_nodes: int, n_channels: int, value: float = 0.1) -> np.ndarray:
    """Flat p_hat array: all nodes, all channels at `value`."""
    return np.full((n_nodes, n_channels), value)


def _assert_proper_colouring(G: nx.Graph, assignment: dict[int, int | None]) -> None:
    """Assert no two adjacent served nodes share a channel."""
    for u, v in G.edges():
        cu, cv = assignment.get(u), assignment.get(v)
        if cu is not None and cv is not None:
            assert cu != cv, (
                f"Adjacent nodes {u} and {v} share channel {cu}; invalid colouring"
            )


# --- allocate() - DSATUR with min-p_hat tie-breaking ---

def test_no_adjacent_nodes_share_channel_path():
    """Proper colouring on a 10-node path graph."""
    G = nx.path_graph(10)
    n, C = 10, 4
    assignment = allocate(G, _uniform_p_hat(n, C), C, theta=0.5)
    _assert_proper_colouring(G, assignment)


def test_no_adjacent_nodes_share_channel_random_graphs():
    """Property test: proper colouring on 20 random Erdős-Rényi graphs."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = int(rng.integers(5, 20))
        p_edge = float(rng.uniform(0.1, 0.5))
        G = nx.erdos_renyi_graph(n, p_edge, seed=int(rng.integers(1000)))
        C = max(n, 8)
        p_hat = rng.uniform(0.0, 0.4, size=(n, C))
        assignment = allocate(G, p_hat, C, theta=0.5)
        _assert_proper_colouring(G, assignment)


def test_abundant_channels_all_served():
    """With n_channels >> max_degree, every node should receive a channel."""
    G = nx.petersen_graph()   # 10 nodes, max degree 3
    n, C = 10, 32
    assignment = allocate(G, _uniform_p_hat(n, C), C, theta=0.5)
    unserved = [v for v, ch in assignment.items() if ch is None]
    assert unserved == [], f"Nodes {unserved} unserved despite abundant channels"


def test_single_channel_k2_one_served():
    """
    K2 (one edge) with 1 channel and theta=0.5: exactly one node served,
    one unserved (the second cannot use the only channel taken by its neighbour).
    """
    G = nx.complete_graph(2)  # K2: nodes 0 and 1, one edge
    p_hat = np.array([[0.1], [0.2]])  # both below theta
    assignment = allocate(G, p_hat, n_channels=1, theta=0.5)

    served = [v for v, ch in assignment.items() if ch is not None]
    unserved = [v for v, ch in assignment.items() if ch is None]
    assert len(served) == 1, f"Expected 1 served node, got {len(served)}"
    assert len(unserved) == 1, f"Expected 1 unserved node, got {len(unserved)}"
    # DSATUR picks the node with lower p_hat first; that node gets channel 0
    assert assignment[0] == 0, "Node 0 (lower p_hat) should be served on channel 0"
    assert assignment[1] is None, "Node 1 should be unserved (channel 0 taken)"


def test_incumbent_masked_channels_never_assigned():
    """Channels marked True in incumbent_mask must never appear in any assignment."""
    G = nx.path_graph(5)
    n, C = 5, 8
    p_hat = _uniform_p_hat(n, C, value=0.05)
    mask = np.array([True, False, True, False, True, False, True, False])  # even channels blocked
    assignment = allocate(G, p_hat, C, theta=0.5, incumbent_mask=mask)

    for node, ch in assignment.items():
        if ch is not None:
            assert not mask[ch], f"Node {node} assigned to masked channel {ch}"


def test_threshold_filters_high_p_hat():
    """Channels with p_hat >= theta are not assigned even when no others conflict."""
    G = nx.path_graph(3)
    n, C = 3, 4
    p_hat = np.zeros((n, C))
    p_hat[:, 0] = 0.9  # channel 0 is above theta=0.5 -> must not be chosen
    assignment = allocate(G, p_hat, C, theta=0.5)

    for node, ch in assignment.items():
        if ch is not None:
            assert ch != 0, f"Node {node} assigned to above-threshold channel 0"


def test_empty_graph_returns_empty():
    """Empty graph produces empty assignment."""
    G = nx.Graph()
    assignment = allocate(G, np.zeros((0, 4)), n_channels=4, theta=0.5)
    assert assignment == {}


def test_min_p_hat_selection():
    """DSATUR picks the channel with minimum p_hat among legal options."""
    G = nx.Graph()
    G.add_node(0)  # single isolated node, can pick any channel
    p_hat = np.array([[0.4, 0.1, 0.3]])  # channel 1 is cheapest
    assignment = allocate(G, p_hat, n_channels=3, theta=0.5)
    assert assignment[0] == 1, "Should pick channel 1 (lowest p_hat=0.1)"


# --- random_allocator() ---

def test_random_allocator_proper_colouring():
    """random_allocator produces a valid proper colouring on random graphs."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(5, 20))
        G = nx.erdos_renyi_graph(n, 0.3, seed=int(rng.integers(1000)))
        C = max(n, 6)
        assignment = random_allocator(G, C, rng=np.random.default_rng(int(rng.integers(1000))))
        _assert_proper_colouring(G, assignment)


def test_random_allocator_incumbent_mask():
    """random_allocator never assigns incumbent-masked channels."""
    G = nx.path_graph(6)
    C = 8
    mask = np.zeros(C, dtype=bool)
    mask[0] = mask[2] = True
    assignment = random_allocator(G, C, rng=np.random.default_rng(0), incumbent_mask=mask)
    for node, ch in assignment.items():
        if ch is not None:
            assert not mask[ch]


# --- greedy_threshold_allocator() ---

def test_greedy_threshold_allocator_proper_colouring():
    """greedy_threshold_allocator produces a valid proper colouring."""
    rng = np.random.default_rng(99)
    for _ in range(20):
        n = int(rng.integers(5, 20))
        G = nx.erdos_renyi_graph(n, 0.3, seed=int(rng.integers(1000)))
        C = max(n, 6)
        p_hat = rng.uniform(0.0, 0.4, size=(n, C))
        assignment = greedy_threshold_allocator(G, p_hat, C, theta=0.5)
        _assert_proper_colouring(G, assignment)


def test_greedy_threshold_allocator_incumbent_mask():
    """greedy_threshold_allocator never assigns incumbent-masked channels."""
    G = nx.path_graph(4)
    n, C = 4, 6
    p_hat = _uniform_p_hat(n, C, 0.1)
    mask = np.array([True, False, False, True, False, False])
    assignment = greedy_threshold_allocator(G, p_hat, C, theta=0.5, incumbent_mask=mask)
    for node, ch in assignment.items():
        if ch is not None:
            assert not mask[ch]
