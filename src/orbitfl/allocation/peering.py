"""Reconciliation of channel conflicts across cluster boundaries.

reconcile_boundaries(assignments, b_edges, p_hat, n_channels) takes the
per-cluster allocations {cluster_id: {node: channel|None}} and the boundary
edges, resolves every same-channel conflict on a boundary edge, and returns a
flat dict {node: channel|None}. The config flag allocation.peering says whether
a caller should apply this step.
"""

from __future__ import annotations

import numpy as np


def reconcile_boundaries(
    assignments: dict[int, dict[int, int | None]],
    b_edges: list[tuple[int, int]],
    p_hat: np.ndarray,
    n_channels: int,
) -> dict[int, int | None]:
    """Resolve same-channel conflicts on boundary interference edges.

    For each edge (u, v) in b_edges where both nodes are served on the same
    channel, the node with lower p_hat[node, channel] keeps the channel, since its
    use of it is less likely to collide. The other node picks the channel with
    minimum p_hat among those not used inside its own cluster, or becomes unserved
    (None) if there is none.

    Conflicts are processed in order of increasing p_hat of the node that keeps its
    channel, so later re-colourings can use channels freed by earlier ones.

    Parameters
    ----------
    assignments : dict[int, dict[int, int|None]]
        {cluster_id: {satellite_node: assigned_channel_or_None}}
        as returned by allocate() applied per cluster.
    b_edges     : list[tuple[int, int]]
        Boundary interference edges from boundary_edges().
    p_hat       : np.ndarray (N_all_satellites, C)
        Predicted busy probability per channel at this timestep.
    n_channels  : int  total number of channels.

    Returns
    -------
    dict[int, int | None]  flat {node: channel|None} after reconciliation.
    """
    # Flatten to a mutable working dict
    flat: dict[int, int | None] = {
        node: ch
        for cluster_dict in assignments.values()
        for node, ch in cluster_dict.items()
    }

    # Map each node to its cluster. The intra-cluster graph is not available here,
    # so re-colouring avoids every channel used by any node of the same cluster.
    node_to_cluster: dict[int, int] = {
        node: cid
        for cid, cluster_dict in assignments.items()
        for node in cluster_dict
    }

    # Process conflicts in order of increasing p_hat of the node that keeps its channel.
    def _conflict_key(edge: tuple[int, int]) -> float:
        u, v = edge
        cu, cv = flat.get(u), flat.get(v)
        if cu is None or cv is None or cu != cv:
            return float("inf")  # not a conflict; sort last (will be skipped)
        pu = float(p_hat[u, cu])
        pv = float(p_hat[v, cv])
        return min(pu, pv)  # winner's p_hat

    sorted_edges = sorted(b_edges, key=_conflict_key)

    for u, v in sorted_edges:
        cu_ch, cv_ch = flat.get(u), flat.get(v)
        if cu_ch is None or cv_ch is None or cu_ch != cv_ch:
            continue  # no conflict: a node is unserved or the channels differ

        channel = cu_ch
        pu = float(p_hat[u, channel])
        pv = float(p_hat[v, channel])

        # The node with lower p_hat keeps its channel
        loser, winner = (u, v) if pu > pv else (v, u)

        # Channels used by the other nodes of the re-coloured node's cluster
        loser_cluster = node_to_cluster.get(loser)
        cluster_used: set[int] = set()
        if loser_cluster is not None:
            for node, ch in assignments[loser_cluster].items():
                if node != loser and ch is not None:
                    cluster_used.add(ch)

        # Re-colour: minimum-p_hat channel not used in its cluster
        palette: list[int] = [
            c for c in range(n_channels) if c not in cluster_used
        ]
        if palette:
            new_ch = min(palette, key=lambda c: float(p_hat[loser, c]))
            flat[loser] = new_ch
            # Record the new channel so later conflicts see it
            assignments[loser_cluster][loser] = new_ch  # type: ignore[index]
        else:
            flat[loser] = None
            if loser_cluster is not None:
                assignments[loser_cluster][loser] = None  # type: ignore[index]

    return flat
