"""Channel-allocation performance metrics.

These metrics are unit-tested but are not used by any experiment in the thesis.

spectral_efficiency      mean Shannon capacity over satellites; a collision gives 0 bits
collision_rate           fraction of served assignments on a channel that is occupied
jain_fairness            Jain's index, (sum x_i)^2 / (n * sum x_i^2)
normalized_churn         reassignments per interference-graph topology change
boundary_collision_rate  collision rate over nodes on a cluster-boundary edge
interior_collision_rate  collision rate over nodes on no boundary edge
"""

from __future__ import annotations

import math

import numpy as np


def spectral_efficiency(
    assignments: dict[int, int | None],
    sinr_achieved: dict[int, float],
    occupancy_truth: np.ndarray,
) -> float:
    """
    Mean Shannon capacity (bits/s/Hz) across served satellites, gated by occupancy.

    A satellite is considered served if its assigned channel is not actually occupied
    (occupancy_truth[node, channel] == 0).  A collision (occupied channel) yields
    0 bits.  The mean is taken over all satellites (served + unserved + colliding),
    so unserved nodes contribute 0 to the numerator.

    Parameters
    ----------
    assignments    : {node: channel|None}  channel assignment per satellite
    sinr_achieved  : {node: float}         achieved SNR in dB (from link budget)
    occupancy_truth: np.ndarray (N, C) int/bool  ground-truth occupancy at service time

    Returns
    -------
    float  mean bits/s/Hz (>= 0)
    """
    n_total = len(assignments)
    if n_total == 0:
        return 0.0

    total_se = 0.0
    for node, channel in assignments.items():
        if channel is None:
            continue  # unserved
        occupied = bool(occupancy_truth[node, channel])
        if occupied:
            continue  # collision -> 0 bits
        snr_db = sinr_achieved.get(node, 0.0)
        snr_linear = 10.0 ** (snr_db / 10.0)
        total_se += math.log2(1.0 + snr_linear)

    return total_se / n_total


def collision_rate(
    assignments: dict[int, int | None],
    occupancy_truth: np.ndarray,
) -> float:
    """
    Fraction of served assignments landing on a truly-occupied channel.

    Only served nodes (channel is not None) contribute to the denominator.
    Returns NaN when no node is served.

    Parameters
    ----------
    assignments    : {node: channel|None}
    occupancy_truth: np.ndarray (N, C) int/bool

    Returns
    -------
    float in [0, 1], or float('nan') if no served nodes
    """
    served = {n: c for n, c in assignments.items() if c is not None}
    if not served:
        return float("nan")
    n_collisions = sum(
        1 for node, ch in served.items() if occupancy_truth[node, ch]
    )
    return n_collisions / len(served)


def jain_fairness(allocations: np.ndarray | list[float]) -> float:
    """Jain's fairness index: J(x) = (sum x_i)^2 / (n * sum x_i^2).

    Equals 1 when all x_i are equal and approaches 1/n when one node takes all.
    Pass per-satellite served-channel counts (0 or 1 in the binary case).

    Parameters
    ----------
    allocations : array-like (n,)  non-negative resource share per node

    Returns
    -------
    float in (0, 1], or 1.0 for a single node
    """
    x = np.asarray(allocations, dtype=float)
    n = len(x)
    if n <= 1:
        return 1.0
    sum_x = float(np.sum(x))
    sum_x2 = float(np.sum(x ** 2))
    if sum_x2 == 0.0:
        return 1.0  # all zeros - trivially fair (nobody served)
    return (sum_x ** 2) / (n * sum_x2)


def normalized_churn(reassignments: int, geometry_change_events: int) -> float:
    """Reassignments per interference-graph topology-change event.

    The raw number of reassignments grows with the size of the graph, so it cannot
    be compared across constellation sizes. Dividing by the number of edges added
    or removed since the previous round gives a rate that equals 1.0 when every
    topology change triggers exactly one reassignment.

    Parameters
    ----------
    reassignments          : int  channel changes between two consecutive rounds
    geometry_change_events : int  interference-graph edge adds + removes

    Returns
    -------
    float (>= 0). 0.0 when reassignments == 0; inf when geometry_change_events == 0
    and reassignments > 0.
    """
    if reassignments == 0:
        return 0.0
    if geometry_change_events == 0:
        return float("inf")
    return reassignments / geometry_change_events


def boundary_collision_rate(
    flat_assignments: dict[int, int | None],
    b_edges: list[tuple[int, int]],
    occupancy_truth: np.ndarray,
) -> float:
    """Collision rate restricted to nodes that appear on at least one boundary edge.

    These are the collisions between clusters, which boundary reconciliation
    (allocation.peering) is meant to reduce. Returns NaN when no boundary node is
    served.

    Parameters
    ----------
    flat_assignments: {node: channel|None}
    b_edges         : boundary edge list from boundary_edges()
    occupancy_truth : np.ndarray (N, C)

    Returns
    -------
    float in [0, 1] or float('nan')
    """
    boundary_nodes: set[int] = {n for edge in b_edges for n in edge}
    boundary_assignments = {
        n: c for n, c in flat_assignments.items() if n in boundary_nodes
    }
    return collision_rate(boundary_assignments, occupancy_truth)


def interior_collision_rate(
    flat_assignments: dict[int, int | None],
    b_edges: list[tuple[int, int]],
    occupancy_truth: np.ndarray,
) -> float:
    """Collision rate for nodes that do not appear on any boundary edge.

    Comparing interior and boundary collision rates separates the cost of
    interference between clusters from the quality of the colouring inside each
    cluster.

    Parameters
    ----------
    flat_assignments: {node: channel|None}
    b_edges         : boundary edge list from boundary_edges()
    occupancy_truth : np.ndarray (N, C)

    Returns
    -------
    float in [0, 1] or float('nan')
    """
    boundary_nodes: set[int] = {n for edge in b_edges for n in edge}
    interior_assignments = {
        n: c for n, c in flat_assignments.items() if n not in boundary_nodes
    }
    return collision_rate(interior_assignments, occupancy_truth)
