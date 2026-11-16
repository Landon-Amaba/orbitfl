"""Checkpoint A: hierarchical FedAvg equals flat FedAvg under three conditions.

Checks numerically, on three small linear models, that two-level (cluster then
global) FedAvg gives the same model as flat FedAvg when
  A1  every client participates,
  A2  there is one local training phase per global merge, and
  A3  averages are weighted by data size at both levels,
and that the two differ as soon as any one of the conditions is broken.

Self-contained: does not import orbitfl.

Usage: python scripts/checkpoint_a_fedavg_equivalence.py
"""

import copy
import sys

import torch
import torch.nn as nn

# --- Configuration ---

SEED = 42
INPUT_DIM = 4
OUTPUT_DIM = 2
E_LOCAL = 5        # local SGD steps per phase
LR = 0.01

# Client data sizes, unequal on purpose
D = [50, 120, 30]

# Cluster A = {client 0, client 1}, cluster B = {client 2}
CLUSTERS = [[0, 1], [2]]

EQUAL_TOL = 1e-6   # max abs diff for "algebraically identical"
DIFFER_TOL = 1e-3  # minimum difference that counts as "different"

# --- Helpers ---


def make_model() -> nn.Linear:
    """Single linear layer in float64."""
    return nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).double()


def local_train(init_state: dict, client_idx: int, n_samples: int) -> dict:
    """Run E_LOCAL SGD steps on a synthetic dataset.

    The data seed is seed + client_idx, so a client sees the same samples in every
    call; only the starting weights differ.
    """
    torch.manual_seed(SEED + client_idx)
    X = torch.randn(n_samples, INPUT_DIM).double()
    y = torch.randn(n_samples, OUTPUT_DIM).double()

    model = make_model()
    model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    for _ in range(E_LOCAL):
        opt.zero_grad()
        nn.MSELoss()(model(X), y).backward()
        opt.step()

    return model.state_dict()


def wavg(states: list[dict], weights: list[float]) -> dict:
    """Weighted average of state_dicts; weights need not be normalised."""
    total = float(sum(weights))
    result: dict = {}
    for key in states[0]:
        result[key] = sum(
            (w / total) * s[key] for s, w in zip(states, weights)
        )
    return result


def max_diff(a: dict, b: dict) -> float:
    """Max absolute per-parameter difference between two state_dicts."""
    return max((a[k] - b[k]).abs().max().item() for k in a)


def hierarchical(client_states: list[dict],
                 clusters: list[list[int]],
                 data_sizes: list[int]) -> dict:
    """Two-level hierarchical FedAvg (as in HierFAVG, Liu et al.).

    Step 1: cluster average, weighted by member data sizes.
    Step 2: global average, weighted by cluster data sizes.
    """
    cluster_states, cluster_weights = [], []
    for members in clusters:
        cluster_states.append(
            wavg([client_states[i] for i in members],
                 [data_sizes[i] for i in members])
        )
        cluster_weights.append(sum(data_sizes[i] for i in members))
    return wavg(cluster_states, cluster_weights)


def flat(client_states: list[dict], data_sizes: list[int]) -> dict:
    """Flat FedAvg: single data-size-weighted average over all clients."""
    return wavg(client_states, data_sizes)


# --- Shared initial state and first-phase client updates ---

torch.manual_seed(SEED)
init_state = make_model().state_dict()

# Each client runs E_LOCAL steps from the same starting model.
client_states = [local_train(init_state, i, D[i]) for i in range(3)]

# --- Checks ---

rows: list[tuple[str, str, float, bool]] = []
# (label, expected, max_diff_value, passed)


# --- All three conditions hold ---
# A1: all 3 clients train
# A2: each starts from init_state, with no aggregation inside a cluster first
# A3: wavg uses D[i] at both levels
# Expected: hierarchical == flat to machine precision.

h = hierarchical(client_states, CLUSTERS, D)
f = flat(client_states, D)
d_pos = max_diff(h, f)
rows.append((
    "POSITIVE  A1+A2+A3 all hold",
    "match",
    d_pos,
    d_pos <= EQUAL_TOL,
))


# --- Break A1 (partial participation) ---
# Client 0 is absent. Cluster A averages over client 1 only, but the global step
# still weights cluster A by D[0] + D[1] = 170:
#   hierarchical:    (170/200)*w_1 + (30/200)*w_2 = 0.850*w1 + 0.150*w2
#   flat over {1,2}: (120/150)*w_1 + (30/150)*w_2 = 0.800*w1 + 0.200*w2

cluster_A_no0 = client_states[1]   # only participating member of cluster A
h_no0 = wavg(
    [cluster_A_no0, client_states[2]],
    [D[0] + D[1], D[2]],           # original cluster-A size (not just D[1])
)
f_part = flat([client_states[1], client_states[2]], [D[1], D[2]])
d_a1 = max_diff(h_no0, f_part)
rows.append((
    "NEGATIVE  break A1: drop client 0, keep original cluster wt",
    "differ",
    d_a1,
    d_a1 >= DIFFER_TOL,
))


# --- Break A2 (more than one local phase in a cluster) ---
# After the first local phase, cluster A aggregates and clients 0 and 1 train a
# second time from that intermediate model before the global merge. Cluster B
# has one phase. The second phase of cluster A starts from a model other than
# init_state, so the two results differ.

inter_A = wavg([client_states[0], client_states[1]], [D[0], D[1]])
state0_r2 = local_train(inter_A, 0, D[0])
state1_r2 = local_train(inter_A, 1, D[1])
cluster_A_r2 = wavg([state0_r2, state1_r2], [D[0], D[1]])
h_a2 = wavg(
    [cluster_A_r2, client_states[2]],
    [D[0] + D[1], D[2]],
)
f_orig = flat(client_states, D)    # flat from first-phase states only
d_a2 = max_diff(h_a2, f_orig)
rows.append((
    "NEGATIVE  break A2: cluster A runs 2 local phases before sync",
    "differ",
    d_a2,
    d_a2 >= DIFFER_TOL,
))


# --- Break A3 (weights not proportional to data size) ---
# Clients in cluster A get equal weight instead of data-size weight; the global
# weights are unchanged:
#   hierarchical: (170/200)*[0.5*w0 + 0.5*w1] + (30/200)*w2
#               = 0.425*w0 + 0.425*w1 + 0.150*w2
#   flat:         0.250*w0 + 0.600*w1 + 0.150*w2
# The gap is proportional to |w0 - w1|.

cluster_A_uni = wavg([client_states[0], client_states[1]], [1, 1])
h_a3 = wavg(
    [cluster_A_uni, client_states[2]],
    [D[0] + D[1], D[2]],
)
d_a3 = max_diff(h_a3, f_orig)
rows.append((
    "NEGATIVE  break A3: uniform weights within cluster A",
    "differ",
    d_a3,
    d_a3 >= DIFFER_TOL,
))


# --- Print table ---

COL = 61
print()
print("Checkpoint A: hierarchical FedAvg vs flat FedAvg")
print("=" * 78)
print(f"  clients    : 3  (D = {D})")
print(f"  clusters   : A = {CLUSTERS[0]},  B = {CLUSTERS[1]}")
print(f"  local steps: E_l = {E_LOCAL},  lr = {LR},  dtype = float64")
print(f"  equal tol  : {EQUAL_TOL:.0e}  (positive checks must be below this)")
print(f"  differ tol : {DIFFER_TOL:.0e}  (negative controls must be above this)")
print()
print(f"  {'Check':{COL}}  {'expect':>7}  {'max |Δ|':>10}  {'status':>6}")
print("  " + "-" * (COL + 28))

n_passed = 0
for label, expected, d, passed in rows:
    status = "PASS" if passed else "FAIL"
    n_passed += passed
    print(f"  {label:{COL}}  {expected:>7}  {d:10.2e}  {status:>6}")

print()
if n_passed == len(rows):
    print(f"  {n_passed}/{len(rows)} checks passed.")
    print()
    print("  With sync_interval = 1 and full participation, the clustering")
    print("  partition cannot change the global model. Strategies can only")
    print("  differ when sync_interval > 1 or participation is partial.")
else:
    print(f"  {n_passed}/{len(rows)} checks passed; equivalence check failed.")
sys.exit(0 if n_passed == len(rows) else 1)
