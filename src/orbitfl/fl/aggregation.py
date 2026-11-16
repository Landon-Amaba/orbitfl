"""FedAvg aggregation rules, written out by hand.

fedavg               data-size-weighted average of client updates
hierarchical_fedavg  FedAvg inside each cluster, then a data-size-weighted
                     average of the cluster models
staleness_weight     (1 + delta)^(-zeta), a FedBuff-style staleness discount

With the same participants and one local training phase per merge,
hierarchical_fedavg equals fedavg up to floating-point error.
tests/test_aggregation.py checks this to 1e-6.
"""

from __future__ import annotations

import dataclasses

import torch

# Model parameters keyed by name.
StateDict = dict[str, torch.Tensor]


@dataclasses.dataclass
class ClientUpdate:
    """
    Result of one round of local training on a satellite client.

    Attributes
    ----------
    state_dict     : StateDict   Post-training model weights (not the delta).
    n_samples      : int         Number of dataset samples trained on this round.
    grad_norm      : float       Cumulative gradient norm across all local steps.
    sat_id         : int         LEO satellite id (0 .. N_leo - 1).
    computed_at_t  : int         Round index when this update was produced.
    """

    state_dict: StateDict
    n_samples: int
    grad_norm: float
    sat_id: int
    computed_at_t: int


def fedavg(updates: list[ClientUpdate]) -> StateDict:
    """Data-size-weighted average of client state dictionaries (flat FedAvg).

    Parameters
    ----------
    updates : list[ClientUpdate]  Non-empty; every update must have n_samples > 0.

    Returns
    -------
    StateDict  Weighted-average model parameters as float32 tensors.

    Raises
    ------
    ValueError  If updates is empty or all n_samples are zero.
    """
    if not updates:
        raise ValueError("fedavg requires at least one ClientUpdate.")
    total = sum(u.n_samples for u in updates)
    if total == 0:
        raise ValueError("fedavg: all updates have n_samples == 0.")

    result: StateDict = {}
    for key in updates[0].state_dict:
        # sum_i (n_i * theta_i) / sum_i n_i, one tensor per parameter name
        result[key] = sum(u.n_samples * u.state_dict[key] for u in updates) / total
    return result


def hierarchical_fedavg(clusters: dict[int, list[ClientUpdate]]) -> StateDict:
    """Two-level FedAvg: aggregate within clusters, then across clusters.

    Equal to flat fedavg over the same updates:
      global = sum_k N_k * ( sum_{i in k} n_i theta_i / N_k ) / sum_k N_k
             = sum_k sum_{i in k} n_i theta_i  /  sum_i n_i
             = fedavg(all updates)

    Parameters
    ----------
    clusters : dict[int, list[ClientUpdate]]
        Mapping from cluster_id to the updates from that cluster.
        Empty cluster lists are skipped.

    Returns
    -------
    StateDict  Global weighted-average model parameters.

    Raises
    ------
    ValueError  If clusters is empty or all clusters have zero samples.
    """
    active = {cid: ups for cid, ups in clusters.items() if ups}
    if not active:
        raise ValueError("hierarchical_fedavg: all clusters are empty.")

    # Step 1: FedAvg and sample count per cluster
    cluster_states: dict[int, StateDict] = {}
    cluster_sizes: dict[int, int] = {}
    for cid, updates in active.items():
        cluster_states[cid] = fedavg(updates)
        cluster_sizes[cid] = sum(u.n_samples for u in updates)

    # Step 2: data-size-weighted average of cluster models
    total = sum(cluster_sizes.values())
    if total == 0:
        raise ValueError("hierarchical_fedavg: all cluster sample counts are zero.")

    result: StateDict = {}
    for key in next(iter(cluster_states.values())):
        result[key] = (
            sum(cluster_sizes[cid] * cluster_states[cid][key] for cid in cluster_states)
            / total
        )
    return result


def staleness_weight(delta_rounds: int, zeta: float = 2.0) -> float:
    """FedBuff-style staleness discount for asynchronous client updates.

    A client whose update is delta rounds old is down-weighted by
    (1 + delta)^(-zeta). zeta = 2 is the FedBuff default.

    Parameters
    ----------
    delta_rounds : int    Rounds elapsed since this update was produced (>= 0).
    zeta         : float  Staleness exponent (default 2.0).

    Returns
    -------
    float  Discount factor in (0, 1].
    """
    return 1.0 / (1.0 + delta_rounds) ** zeta
