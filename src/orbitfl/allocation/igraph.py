"""Interference graph construction.

build_interference_graph  one node per LEO; edge (i, j) iff is_interference_edge
                          holds in either direction
cluster_subgraphs         induced subgraph per cluster id (membership -1 excluded)
boundary_edges            interference edges that cross cluster boundaries
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from orbitfl.geometry.linkbudget import is_interference_edge


def build_interference_graph(
    geometry: dict,
    threshold_db: float,
    elevation_mask_deg: float,
) -> nx.Graph:
    """Build the LEO interference graph for one timestep.

    Nodes are all LEOs in the constellation (integer indices 0..N-1). Edge (i, j)
    is added when is_interference_edge returns True in either direction.

    Parameters
    ----------
    geometry : dict
        Must contain:
          'ecef' : np.ndarray (N, 3) km   satellite positions
          'lat'  : np.ndarray (N,) deg    sub-satellite latitudes
          'lon'  : np.ndarray (N,) deg    sub-satellite longitudes
    threshold_db       : float  co-channel INR threshold (AllocationConfig.threshold_db)
    elevation_mask_deg : float  elevation mask in degrees (AllocationConfig.elevation_mask_deg)

    Returns
    -------
    nx.Graph  undirected; N nodes (int indices); edges are interference pairs
    """
    N: int = len(geometry["lat"])
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            if (
                is_interference_edge(i, j, geometry, threshold_db, elevation_mask_deg)
                or is_interference_edge(j, i, geometry, threshold_db, elevation_mask_deg)
            ):
                G.add_edge(i, j)

    return G


def cluster_subgraphs(
    graph: nx.Graph,
    membership: np.ndarray,
) -> dict[int, nx.Graph]:
    """
    Induce one subgraph per FL cluster.

    Parameters
    ----------
    graph      : nx.Graph  full LEO interference graph from build_interference_graph
    membership : np.ndarray (N,) int  cluster id for each satellite; -1 = no cluster

    Returns
    -------
    dict[int, nx.Graph]  cluster_id -> induced subgraph on that cluster's nodes.
        Nodes with membership == -1 are excluded from all subgraphs.
        Edges in a subgraph are the interference edges *within* the cluster.
    """
    clusters: dict[int, list[int]] = {}
    for node, cid in enumerate(membership):
        if int(cid) == -1:
            continue
        clusters.setdefault(int(cid), []).append(node)

    return {cid: graph.subgraph(nodes).copy() for cid, nodes in clusters.items()}


def boundary_edges(
    graph: nx.Graph,
    membership: np.ndarray,
) -> list[tuple[int, int]]:
    """Return interference edges that cross FL cluster boundaries.

    An edge (u, v) is a boundary edge when membership[u] != membership[v] and
    neither node has membership == -1. Two nodes on such an edge can be given the
    same channel by their separate per-cluster colourings and then interfere.

    Parameters
    ----------
    graph      : nx.Graph  full interference graph
    membership : np.ndarray (N,) int  cluster id per satellite; -1 = unassigned

    Returns
    -------
    list[tuple[int, int]]  boundary edge pairs (unordered; each appears once)
    """
    result: list[tuple[int, int]] = []
    for u, v in graph.edges():
        cu, cv = int(membership[u]), int(membership[v])
        if cu == -1 or cv == -1:
            continue
        if cu != cv:
            result.append((u, v))
    return result
