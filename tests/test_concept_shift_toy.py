"""Minimal demonstration that clustering improves accuracy under hidden concept shift.

Clustering cannot improve accuracy when all clusters share one best predictor
that each client can learn from its own input. This test checks the converse
in the smallest possible setting: when clusters have conflicting best
predictors that cannot be told apart from the input, a correct partition wins.

Construction (2 clusters, identical inputs, opposite labels):
  * Every client draws x ~ N(0, I_d), the same for both clusters, so x carries
    no information about the cluster.
  * Cluster A labels: y = 1[x.w > 0].
  * Cluster B labels: y = 1[x.w < 0].
  The gradients of the two clusters are equal and opposite, so a single pooled
  linear model stays at chance. Only a model that serves one cluster can be right.

Three partitions are run through a small harness that uses the real fedavg /
hierarchical_fedavg and handles cluster models as loop.py does (kept between
global syncs, reset at each sync, each client scored under its cluster's
model). Everything except the partition is identical:
  (i)   correct clustering  (membership = true cluster labels)
  (ii)  flat                (one cluster with all clients, i.e. flat FedAvg)
  (iii) random clustering   (balanced random partition mixing A and B)

Expected: correct clustering is near 1.0, flat and random are near chance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from orbitfl.fl.aggregation import ClientUpdate, StateDict, fedavg, hierarchical_fedavg

# --- toy hyperparameters (kept in one place; no magic numbers scattered) -----
_DIM: int = 8               # input feature dimension d
_N_PER_CLIENT: int = 256    # samples per client
_CLIENTS_PER_CLUSTER: int = 6
_LOCAL_EPOCHS: int = 5
_LOCAL_LR: float = 0.05
_SYNC_INTERVAL: int = 10    # global-sync cadence for the clustered arms
_ROUND_BUDGET: int = 20     # multiple of _SYNC_INTERVAL -> final round is a sync round
_CHANCE: float = 0.5        # balanced-label accuracy of an uninformed model


# --- Minimal linear classifier (the smallest model that can express the conflict) ---


class _LinearClassifier(nn.Module):
    """Single linear layer d -> 1; logit output for BCEWithLogitsLoss."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, d) -> (B,)
        return self.fc(x).squeeze(-1)


# --- Synthetic data: identical inputs, conflicting (inverted) optima by cluster ---


def _make_clients(
    seed: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    """
    Build 2*_CLIENTS_PER_CLUSTER clients: half cluster A, half cluster B.

    Returns
    -------
    clients   : list of (X (N, d) float32, y (N,) float32) per client.
    labels    : (n_clients,) int  true cluster id (0 = A, 1 = B).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(_DIM).astype(np.float32)  # (d,) shared decision vector

    clients: list[tuple[torch.Tensor, torch.Tensor]] = []
    labels: list[int] = []
    for cluster in (0, 1):
        for _ in range(_CLIENTS_PER_CLUSTER):
            x = rng.standard_normal((_N_PER_CLIENT, _DIM)).astype(np.float32)  # (N, d)
            score = x @ w                                                       # (N,)
            # Cluster A: y = 1[score > 0]; cluster B: the exact inversion.
            y_a = (score > 0.0).astype(np.float32)
            y = y_a if cluster == 0 else (1.0 - y_a)
            clients.append((torch.from_numpy(x), torch.from_numpy(y)))
            labels.append(cluster)
    return clients, np.asarray(labels, dtype=np.int32)


# --- Thin harness mirroring loop.py's cluster-model persistence + real FedAvg ---


def _train_local(
    start_state: StateDict, x: torch.Tensor, y: torch.Tensor, sat_id: int, r: int
) -> ClientUpdate:
    """Load start_state, run _LOCAL_EPOCHS full-batch steps, return the update."""
    model = _LinearClassifier(_DIM)
    model.load_state_dict({k: v.clone() for k, v in start_state.items()})
    opt = torch.optim.Adam(model.parameters(), lr=_LOCAL_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(_LOCAL_EPOCHS):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return ClientUpdate(
        state_dict=state, n_samples=int(x.shape[0]), grad_norm=0.0, sat_id=sat_id,
        computed_at_t=r,
    )


@torch.no_grad()
def _accuracy(state: StateDict, x: torch.Tensor, y: torch.Tensor) -> tuple[int, int]:
    """Return (n_correct, n_total) for `state` on (x, y)."""
    model = _LinearClassifier(_DIM)
    model.load_state_dict(state)
    model.eval()
    pred = (torch.sigmoid(model(x)) >= 0.5).float()
    return int((pred == y).sum().item()), int(y.numel())


def _run_arm(
    clients: list[tuple[torch.Tensor, torch.Tensor]],
    membership: np.ndarray,
    sync_interval: int,
    seed: int,
) -> float:
    """
    Run the fixed-partition FL loop and return final personalized accuracy.

    Mirrors loop.py exactly: cluster models persist within a sync window and
    reset at each global sync; each client is scored under its own cluster's
    current model (falling back to the global model for a cluster with no model
    yet). The final round is a sync round, so the personalized metric is read
    off the specialized cluster models before the deferred reset - precisely the
    `ba_personalized` semantics.
    """
    torch.manual_seed(seed)
    global_state: StateDict = _LinearClassifier(_DIM).state_dict()
    global_state = {k: v.clone() for k, v in global_state.items()}
    cluster_models: dict[int, StateDict] = {}

    final_pers_acc = _CHANCE
    for r in range(_ROUND_BUDGET):
        clusters: dict[int, list[ClientUpdate]] = {}
        for k, (x, y) in enumerate(clients):
            cid = int(membership[k])
            start_state = cluster_models.get(cid, global_state)
            u = _train_local(start_state, x, y, sat_id=k, r=r)
            clusters.setdefault(cid, []).append(u)

        # Cluster-level FedAvg -> each cluster's persistent model (real code path).
        for cid, ups in clusters.items():
            cluster_models[cid] = fedavg(ups)

        # Personalized metric (mechanism metric): score each client under its
        # own cluster's current model, pooled across all clients.
        n_correct = n_total = 0
        for k, (x, y) in enumerate(clients):
            state = cluster_models.get(int(membership[k]), global_state)
            c, t = _accuracy(state, x, y)
            n_correct += c
            n_total += t
        final_pers_acc = n_correct / n_total

        if (r + 1) % sync_interval == 0:
            global_state = hierarchical_fedavg(clusters)
            cluster_models = {}

    return final_pers_acc


# --- Tests ---


def test_clustering_beats_flat_and_random_under_hidden_concept_shift() -> None:
    """Correct clustering is near 1.0; flat and random are near chance.

    The correct partition must beat both baselines by a wide margin.
    """
    seed = 0
    clients, true_labels = _make_clients(seed)
    n_clients = len(clients)

    # (iii) random clustering: a balanced two-way partition unrelated to the true
    # clusters. The draw is repeated until both random clusters contain members of
    # A and of B, so that neither can specialise.
    rng = np.random.default_rng(seed + 1)
    while True:
        random_labels = np.array([0, 1] * (n_clients // 2), dtype=np.int32)
        rng.shuffle(random_labels)
        mixes = all(
            {int(true_labels[k]) for k in np.where(random_labels == c)[0]} == {0, 1}
            for c in (0, 1)
        )
        if mixes:
            break

    # (ii) flat: a single cluster pools A and B into one model - flat FedAvg
    # exactly. By symmetry the pooled gradients cancel, so it stays at chance.
    flat_labels = np.zeros(n_clients, dtype=np.int32)

    acc_correct = _run_arm(clients, true_labels, _SYNC_INTERVAL, seed)
    acc_flat = _run_arm(clients, flat_labels, _SYNC_INTERVAL, seed)
    acc_random = _run_arm(clients, random_labels, _SYNC_INTERVAL, seed)

    # The triple, at its core: correct >> flat ~ random ~ chance.
    assert acc_correct > 0.9, f"correct clustering should be near-perfect, got {acc_correct:.3f}"
    assert acc_flat < 0.65, f"flat FedAvg should sit near chance, got {acc_flat:.3f}"
    assert acc_random < 0.65, f"random clustering should sit near chance, got {acc_random:.3f}"
    # A wide, unambiguous margin (not a knife-edge): correct beats both baselines
    # by far more than either baseline's deviation from chance.
    margin = acc_correct - max(acc_flat, acc_random)
    assert margin > 0.3, f"clustering benefit margin too small ({margin:.3f})"


def test_conflict_is_hidden_from_the_input() -> None:
    """Check of the construction: the two clusters have the same input distribution,
    so a classifier fitted on the pooled data, without the cluster identity,
    cannot beat chance.
    """
    clients, labels = _make_clients(seed=0)
    x_all = torch.cat([c[0] for c in clients], dim=0)
    y_all = torch.cat([c[1] for c in clients], dim=0)

    torch.manual_seed(0)
    model = _LinearClassifier(_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=_LOCAL_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(200):  # train hard on the pooled mixture
        opt.zero_grad()
        loss_fn(model(x_all), y_all).backward()
        opt.step()
    n_correct, n_total = _accuracy(
        {k: v.detach().clone() for k, v in model.state_dict().items()}, x_all, y_all
    )
    pooled_acc = n_correct / n_total
    assert abs(pooled_acc - _CHANCE) < 0.05, (
        f"pooled model should be stuck at chance ({_CHANCE}); got {pooled_acc:.3f}; "
        "the conflict is not actually hidden from the input."
    )
