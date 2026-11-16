"""Tests for src/orbitfl/allocation/igraph.py.

- The node set is all LEOs.
- In-plane neighbours in the thesis constellation (24 deg apart, about 2671 km
  between sub-satellite points) are never interference edges for any elevation
  mask >= 0 deg.
- rivalry_radius_km(550, 12) < 2671 km.
- build_interference_graph agrees with an independent reference implementation
  on the same input.
- cluster_subgraphs and boundary_edges split and classify edges correctly.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from orbitfl.allocation.igraph import (
    boundary_edges,
    build_interference_graph,
    cluster_subgraphs,
)
from orbitfl.config import load_config
from orbitfl.geometry.linkbudget import is_interference_edge, rivalry_radius_km
from orbitfl.geometry.orbits import build_walker_delta, propagate

_BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


# --- Helpers ---

def _frozen_geometry() -> dict:
    """Propagate the thesis Walker-delta constellation to t = 0; return the geometry dict."""
    cfg = load_config(_BASE)
    lc = cfg.constellation.leo
    from skyfield.api import load
    ts = load.timescale()
    t0 = ts.from_datetime(
        __import__("datetime").datetime.fromisoformat(
            cfg.sim.epoch.replace("Z", "+00:00")
        )
    )
    sats = build_walker_delta(
        lc.n_planes, lc.sats_per_plane, lc.alt_km, lc.inc_deg, lc.phasing_f, t0
    )
    eph = propagate(sats, ts.tt_jd([t0.tt]))  # single timestep
    return {
        "ecef": eph.ecef[0],           # (N, 3) km
        "lat":  eph.subpoint_lat[0],   # (N,) deg
        "lon":  eph.subpoint_lon[0],   # (N,) deg
    }


def _reference_build_graph(geo: dict, threshold_db: float, elevation_mask_deg: float) -> nx.Graph:
    """Reference implementation matching gate_b_edge_density._build_graph exactly."""
    N = len(geo["lat"])
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if (
                is_interference_edge(i, j, geo, threshold_db, elevation_mask_deg)
                or is_interference_edge(j, i, geo, threshold_db, elevation_mask_deg)
            ):
                G.add_edge(i, j)
    return G


# --- Tests ---

def test_node_set_is_all_leos():
    """build_interference_graph includes all LEOs, never filters on activity."""
    cfg = load_config(_BASE)
    N_leo = cfg.constellation.leo.n_planes * cfg.constellation.leo.sats_per_plane
    geo = _frozen_geometry()
    G = build_interference_graph(
        geo, cfg.allocation.threshold_db, cfg.allocation.elevation_mask_deg
    )
    assert len(G.nodes()) == N_leo, (
        f"Expected {N_leo} nodes (all LEOs), got {len(G.nodes())}"
    )


def test_rivalry_radius_margin_at_12deg():
    """rivalry_radius_km(550, 12) is below the in-plane sub-satellite spacing (about 2671 km).

    With a 12 deg mask, neighbouring satellites in the same plane are therefore
    outside each other's interference range, by a factor of about 1.74.
    """
    r = rivalry_radius_km(550.0, 12.0)
    inplane_spacing_km = 2671.0  # great-circle distance, 24 deg / 360 deg x 2piR_earth
    assert r < inplane_spacing_km, (
        f"rivalry_radius({r:.1f} km) >= in-plane spacing ({inplane_spacing_km} km)"
        f"; in-plane neighbours would interfere"
    )
    margin = inplane_spacing_km / r
    assert margin > 1.5, f"Margin {margin:.2f}× is below the expected 1.74×"


@pytest.mark.parametrize("mask_deg", [0, 5, 10, 12, 15, 20, 25])
def test_inplane_neighbours_never_edges(mask_deg):
    """
    Adjacent satellites in the same orbital plane are never interference edges
    at any elevation mask >= 0 deg.

    The in-plane sub-satellite great-circle distance (~ 2671 km) exceeds the
    rivalry radius at mask=0 deg (2557 km), so this holds for all masks.
    """
    cfg = load_config(_BASE)
    lc = cfg.constellation.leo
    geo = _frozen_geometry()
    spp = lc.sats_per_plane

    G = build_interference_graph(geo, cfg.allocation.threshold_db, float(mask_deg))

    for plane in range(lc.n_planes):
        for seat in range(spp):
            i = plane * spp + seat
            j = plane * spp + (seat + 1) % spp  # wraparound within plane
            assert not G.has_edge(i, j), (
                f"In-plane edge ({i}, {j}) found at mask={mask_deg}°; "
                f"C2 premise violated (rivalry radius too large)"
            )


def test_agrees_with_reference_implementation():
    """build_interference_graph gives the same graph as the reference implementation
    at a 12 deg mask and a -6 dB threshold.
    """
    cfg = load_config(_BASE)
    geo = _frozen_geometry()
    threshold = cfg.allocation.threshold_db
    mask = cfg.allocation.elevation_mask_deg

    G_module = build_interference_graph(geo, threshold, mask)
    G_ref = _reference_build_graph(geo, threshold, mask)

    assert set(G_module.edges()) == set(G_ref.edges()), (
        "igraph.build_interference_graph disagrees with the reference implementation"
    )


def test_cluster_subgraphs_partitions_nodes():
    """cluster_subgraphs returns disjoint node sets covering all non-(-1) members."""
    # 4-node path graph: 0-1-2-3; clusters {0: [0,1], 1: [2,3]}
    G = nx.path_graph(4)
    membership = np.array([0, 0, 1, 1])
    subs = cluster_subgraphs(G, membership)

    assert set(subs.keys()) == {0, 1}
    assert set(subs[0].nodes()) == {0, 1}
    assert set(subs[1].nodes()) == {2, 3}
    # Edge 1-2 is between clusters -> not in either subgraph
    assert not subs[0].has_edge(1, 2)
    assert not subs[1].has_edge(1, 2)
    # Intra-cluster edges are preserved
    assert subs[0].has_edge(0, 1)
    assert subs[1].has_edge(2, 3)


def test_cluster_subgraphs_skips_minus_one():
    """Nodes with membership == -1 are excluded from all cluster subgraphs."""
    G = nx.path_graph(4)
    membership = np.array([-1, 0, 0, -1])
    subs = cluster_subgraphs(G, membership)

    assert -1 not in subs
    assert set(subs[0].nodes()) == {1, 2}


def test_boundary_edges_correct():
    """boundary_edges returns exactly the edges crossing cluster boundaries."""
    # 4-node path: 0-1-2-3; clusters 0={0,1}, 1={2,3}; boundary edge is (1,2)
    G = nx.path_graph(4)
    membership = np.array([0, 0, 1, 1])
    b = boundary_edges(G, membership)

    assert len(b) == 1
    assert set(b[0]) == {1, 2}


def test_boundary_edges_excludes_minus_one():
    """Edges touching a node with membership == -1 are not boundary edges."""
    G = nx.path_graph(3)  # 0-1-2
    membership = np.array([-1, 0, 1])
    b = boundary_edges(G, membership)

    # Edge (0,1) involves node 0 (membership -1) -> excluded
    # Edge (1,2) is the true boundary
    assert len(b) == 1
    assert set(b[0]) == {1, 2}


def test_boundary_edges_empty_for_single_cluster():
    """No boundary edges when all nodes belong to the same cluster."""
    G = nx.complete_graph(5)
    membership = np.zeros(5, dtype=int)
    b = boundary_edges(G, membership)
    assert b == []
