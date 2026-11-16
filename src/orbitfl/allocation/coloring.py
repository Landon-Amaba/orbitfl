"""Channel allocation by graph colouring.

allocate                    DSATUR node ordering; each node takes the legal channel
                            with the lowest predicted busy probability
random_allocator            proper colouring with random channels; ignores p_hat
greedy_threshold_allocator  degree-ordered greedy colouring with the same threshold

A node with no legal channel is left unserved (None).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.coloring import strategy_saturation_largest_first


def allocate(
    subgraph: nx.Graph,
    p_hat: np.ndarray,
    n_channels: int,
    theta: float,
    incumbent_mask: np.ndarray | None = None,
) -> dict[int, int | None]:
    """DSATUR channel allocation that prefers the channel least likely to be busy.

    Nodes are processed in DSATUR order (highest saturation first, ties broken by
    degree). The legal palette of a node is
      {c in 0..n_channels-1 : p_hat[node, c] < theta
                             and not incumbent_mask[c]
                             and c not used by any coloured neighbour}
    The palette channel with minimum p_hat[node, c] is chosen. A node with an empty
    palette is left unserved (None).

    Parameters
    ----------
    subgraph       : nx.Graph    induced subgraph for one FL cluster
    p_hat          : np.ndarray  (N_all_satellites, C) predicted busy probability
    n_channels     : int         total number of channels (C)
    theta          : float       channels with p_hat >= theta are excluded
    incumbent_mask : np.ndarray (C,) bool or None
                                 channels reserved for incumbent users; None = no
                                 channel reserved

    Returns
    -------
    dict[int, int | None]  node -> assigned channel index, or None if unserved
    """
    if len(subgraph) == 0:
        return {}

    blocked: set[int] = set()
    if incumbent_mask is not None:
        blocked = {c for c in range(n_channels) if incumbent_mask[c]}

    # The colours dict is updated between yields of the DSATUR generator. Unserved
    # nodes get the value n_channels, which marks them as coloured without occupying
    # any real channel (valid channels are 0..n_channels-1).
    colors: dict[int, int] = {}
    assignment: dict[int, int | None] = {}

    for node in strategy_saturation_largest_first(subgraph, colors):
        neighbor_used: set[int] = {
            colors[nb] for nb in subgraph.neighbors(node) if nb in colors
        }
        palette: list[int] = [
            c
            for c in range(n_channels)
            if c not in blocked
            and c not in neighbor_used
            and float(p_hat[node, c]) < theta
        ]

        if palette:
            chosen: int = min(palette, key=lambda c: float(p_hat[node, c]))
            colors[node] = chosen
            assignment[node] = chosen
        else:
            colors[node] = n_channels  # sentinel: counts as coloured, never in palette
            assignment[node] = None

    return assignment


def random_allocator(
    subgraph: nx.Graph,
    n_channels: int,
    rng: np.random.Generator | None = None,
    incumbent_mask: np.ndarray | None = None,
) -> dict[int, int | None]:
    """Baseline allocator: proper graph colouring with random channel selection.

    Each node, in random order, gets a uniformly random channel that is not used by
    a coloured neighbour and not reserved by incumbent_mask. Nodes with no legal
    channel get None. p_hat is ignored.

    Parameters
    ----------
    subgraph       : nx.Graph
    n_channels     : int
    rng            : np.random.Generator or None  (default: fresh default_rng)
    incumbent_mask : np.ndarray (C,) bool or None

    Returns
    -------
    dict[int, int | None]
    """
    if len(subgraph) == 0:
        return {}

    if rng is None:
        rng = np.random.default_rng()

    blocked: set[int] = set()
    if incumbent_mask is not None:
        blocked = {c for c in range(n_channels) if incumbent_mask[c]}

    node_order: list[int] = list(subgraph.nodes())
    rng.shuffle(node_order)  # type: ignore[arg-type]

    colors: dict[int, int] = {}
    assignment: dict[int, int | None] = {}

    for node in node_order:
        neighbor_used: set[int] = {colors[nb] for nb in subgraph.neighbors(node) if nb in colors}
        palette: list[int] = [
            c for c in range(n_channels) if c not in blocked and c not in neighbor_used
        ]
        if palette:
            idx: int = int(rng.integers(len(palette)))
            chosen_r: int = palette[idx]
            colors[node] = chosen_r
            assignment[node] = chosen_r
        else:
            colors[node] = n_channels
            assignment[node] = None

    return assignment


def greedy_threshold_allocator(
    subgraph: nx.Graph,
    p_hat: np.ndarray,
    n_channels: int,
    theta: float,
    incumbent_mask: np.ndarray | None = None,
) -> dict[int, int | None]:
    """Baseline allocator: threshold-gated proper colouring without DSATUR ordering.

    Nodes are processed in order of decreasing degree. The palette of a node is
    {c : p_hat[node, c] < theta, not reserved, not used by a neighbour} and the
    channel with minimum p_hat is chosen. If the threshold leaves no option, any
    channel not used by a neighbour is accepted. Nodes with no legal channel at all
    get None.

    Parameters
    ----------
    subgraph       : nx.Graph
    p_hat          : np.ndarray  (N_all_satellites, C)
    n_channels     : int
    theta          : float
    incumbent_mask : np.ndarray (C,) bool or None

    Returns
    -------
    dict[int, int | None]
    """
    if len(subgraph) == 0:
        return {}

    blocked: set[int] = set()
    if incumbent_mask is not None:
        blocked = {c for c in range(n_channels) if incumbent_mask[c]}

    node_order: list[int] = sorted(subgraph.nodes(), key=lambda n: subgraph.degree(n), reverse=True)

    colors: dict[int, int] = {}
    assignment: dict[int, int | None] = {}

    for node in node_order:
        neighbor_used: set[int] = {colors[nb] for nb in subgraph.neighbors(node) if nb in colors}
        palette: list[int] = [
            c
            for c in range(n_channels)
            if c not in blocked and c not in neighbor_used and float(p_hat[node, c]) < theta
        ]

        if palette:
            chosen_g: int = min(palette, key=lambda c: float(p_hat[node, c]))
        else:
            # Fallback: ignore threshold, pick any legal channel
            fallback: list[int] = [
                c for c in range(n_channels) if c not in blocked and c not in neighbor_used
            ]
            chosen_g = min(fallback, key=lambda c: float(p_hat[node, c])) if fallback else -1

        if chosen_g == -1:
            colors[node] = n_channels
            assignment[node] = None
        else:
            colors[node] = chosen_g
            assignment[node] = chosen_g

    return assignment
