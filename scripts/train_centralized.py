r"""Centralized baseline (Checkpoint C).

Pools the observation streams of all satellites, trains the CNN forecaster and
the position-only baseline centrally, and evaluates them together with the
persistence and marginal baselines on a held-out probe set, for several
forecast horizons. This gives the accuracy a model can reach without
federation, and shows at which horizons the CNN beats each baseline.

Usage:
  python scripts/train_centralized.py \
      --config configs/base.yaml \
      --trace  data/traces/leo180_meo3 \
      --out    results/

  # Quick local run: one horizon, one seed, fewer epochs
  python scripts/train_centralized.py \
      --config configs/base.yaml \
      --override configs/experiments/centralized_baseline.yaml \
      --trace  data/traces/leo180_meo3 \
      --delta-steps 4 --seeds 0 --epochs 10 --out results/ --dirty-ok

The summary printed at the end answers three questions:
  Q1  Does the CNN beat both persistence and the position-only model at the
      horizon where its balanced accuracy is highest?
  Q2  How much does the positional encoding change balanced accuracy?
  Q3  How far is the CNN above the position-only model? A gap near zero means
      its skill comes from geography alone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.config import load_config
from orbitfl.metrics.forecast import auc, balanced_accuracy, bce, skill_vs_persistence
from orbitfl.models.baselines import EphemerisOnlyForecaster
from orbitfl.models.forecaster import OccupancyForecaster
from orbitfl.trace.reader import Trace
from orbitfl.utils.device import get_device
from orbitfl.utils.plotting import save_figure, use_orbitfl_style
from orbitfl.utils.runs import create_run_dir
from orbitfl.utils.seeding import seed_everything
from orbitfl.world.occupancy import OccupancyWorld

# Forecast horizons (in timesteps) evaluated by default
_DEFAULT_DELTAS: list[int] = [1, 2, 4, 8, 16]
_PROBE_FRAC: float = 0.20   # last 20 % of windows are the held-out probe set
_BATCH_SIZE: int = 512


# --- Dataset helpers ---


def _build_split_datasets(
    world: OccupancyWorld,
    trace: Trace,
    N_leo: int,
    window: int,
    delta_steps: int,
) -> tuple:
    """Build the train and probe datasets over all LEOs for a given horizon.

    Returns (train_ds, probe_ds, n_train_samples, n_probe_samples).
    Train: the first (1 - PROBE_FRAC) of each satellite's windows.
    Probe: the last PROBE_FRAC.
    """
    train_subsets = []
    probe_subsets = []

    for sat_id in range(N_leo):
        ds = world.observation_stream(trace, sat_id, window=window, delta_steps=delta_steps)
        n = len(ds)
        if n == 0:
            continue
        n_probe = max(1, int(n * _PROBE_FRAC))
        n_train = n - n_probe
        if n_train > 0:
            train_subsets.append(Subset(ds, range(n_train)))
        probe_subsets.append(Subset(ds, range(n_train, n)))

    train_ds = ConcatDataset(train_subsets) if train_subsets else ConcatDataset([])
    probe_ds = ConcatDataset(probe_subsets) if probe_subsets else ConcatDataset([])
    return train_ds, probe_ds


# --- Training helpers ---


def _train_cnn(
    model: OccupancyForecaster,
    train_ds: ConcatDataset,
    use_position: bool,
    device: torch.device,
    n_epochs: int,
    lr: float = 1e-3,
) -> list[float]:
    """Train the CNN on pooled data; return per-epoch train loss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for obs_win, pos_enc, target in loader:
            obs_win = obs_win.to(device)
            pos_enc = pos_enc.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(obs_win, pos_enc if use_position else None)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


def _train_eph(
    model: EphemerisOnlyForecaster,
    train_ds: ConcatDataset,
    device: torch.device,
    n_epochs: int,
    lr: float = 1e-3,
) -> list[float]:
    """Train the position-only model on (pos_enc, target) pairs; obs_window is ignored."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for _, pos_enc, target in loader:
            pos_enc = pos_enc.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(pos_enc)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


# --- Evaluation helper ---


def _evaluate_probe(
    cnn: OccupancyForecaster,
    eph: EphemerisOnlyForecaster,
    probe_ds: ConcatDataset,
    use_position: bool,
    device: torch.device,
) -> dict:
    """Evaluate the CNN, position-only, persistence and marginal models on the probe set.

    All four are scored on the same samples. Returns a dict of balanced accuracy
    and AUC values, plus the CNN's AUC skill over persistence and over the
    position-only model.
    """
    loader = DataLoader(probe_ds, batch_size=_BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    # Accumulators
    cnn_probs_list: list = []
    eph_probs_list: list = []
    pers_probs_list: list = []
    marg_probs_list: list = []
    y_true_list: list = []

    cnn.eval()
    eph.eval()

    with torch.no_grad():
        for obs_win, pos_enc, target in loader:
            y_true_list.append(target.numpy())                         # (B, C)

            # CNN
            logits = cnn(obs_win.to(device), pos_enc.to(device) if use_position else None)
            cnn_probs_list.append(torch.sigmoid(logits).cpu().numpy())

            # Position-only model
            eph_logits = eph(pos_enc.to(device))
            eph_probs_list.append(torch.sigmoid(eph_logits).cpu().numpy())

            # Persistence: last observation in window
            pers_probs_list.append(obs_win[:, -1, :].numpy())         # (B, C)

            # Marginal: window mean
            marg_probs_list.append(obs_win.mean(dim=1).numpy())       # (B, C)

    # Flatten to (N_probe * C,) for sklearn metrics
    y_true = np.concatenate(y_true_list).ravel()                  # (N*C,) binary float
    cnn_p = np.concatenate(cnn_probs_list).ravel()
    eph_p = np.concatenate(eph_probs_list).ravel()
    pers_p = np.concatenate(pers_probs_list).ravel()
    marg_p = np.concatenate(marg_probs_list).ravel()

    def _ba(p: np.ndarray) -> float:
        return balanced_accuracy(y_true, (p >= 0.5).astype(float))

    results = {
        "ba_cnn":  _ba(cnn_p),
        "ba_eph":  _ba(eph_p),
        "ba_pers": _ba(pers_p),
        "ba_marg": _ba(marg_p),
        "auc_cnn":  auc(y_true, cnn_p),
        "auc_eph":  auc(y_true, eph_p),
        "auc_pers": auc(y_true, pers_p),
        "skill_cnn_pers": skill_vs_persistence(y_true, cnn_p, pers_p),
        "skill_cnn_eph":  skill_vs_persistence(y_true, cnn_p, eph_p),
        "bce_cnn":  bce(y_true, cnn_p),
        "bce_eph":  bce(y_true, eph_p),
    }
    return results


# --- Figures ---


def _plot_ba_vs_delta(rows_df: pd.DataFrame, run_dir: Path) -> None:
    """BA vs horizon for all four models (mean and 95 % CI across seeds)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"cnn": "steelblue", "eph": "darkorange", "pers": "firebrick", "marg": "gray"}

    for ax, use_pos in zip(axes, [True, False]):
        sub = rows_df[rows_df["use_position"] == use_pos]
        for model, col in [("CNN", "ba_cnn"), ("EphOnly", "ba_eph"),
                            ("Persist", "ba_pers"), ("Marginal", "ba_marg")]:
            grp = sub.groupby("delta_steps")[col]
            mean = grp.mean()
            sem = grp.sem()
            x = mean.index.to_numpy() * 15   # steps -> seconds
            color_key = col.split("_", 1)[1]   # "cnn", "eph", "pers", "marg"
            ax.plot(x, mean.to_numpy(), color=colors[color_key], marker="o", label=model)
            ax.fill_between(x,
                            (mean - 1.96 * sem).to_numpy(),
                            (mean + 1.96 * sem).to_numpy(),
                            alpha=0.15, color=colors[color_key])

        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, label="Chance")
        ax.set_xlabel("Forecast horizon Δ (s)")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"{'With' if use_pos else 'Without'} positional encoding")
        ax.set_xscale("log")
        ax.legend(fontsize=9)

    fig.suptitle("Centralized baseline: BA vs Δ (mean ± 95 % CI)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, run_dir, "centralized_ba_vs_delta")
    plt.close(fig)


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized baseline over forecast horizons.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--override", type=Path, default=None)
    parser.add_argument("--trace", type=Path, required=True,
                        help="Trace directory (output of make_trace.py)")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="Results root (run subdir created automatically)")
    parser.add_argument("--delta-steps", type=int, nargs="+", default=None,
                        help="Δ values in timesteps (default: 1 2 4 8 16)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds (default: from config)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per run (default: 30)")
    parser.add_argument("--dirty-ok", action="store_true",
                        help="Allow uncommitted changes (results will note dirty state)")
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.override)
    delta_list: list[int] = args.delta_steps or _DEFAULT_DELTAS
    seeds: list[int] = args.seeds or list(cfg.experiment.seeds)

    run_dir = create_run_dir("centralized_baseline", args.out, allow_dirty=args.dirty_ok)
    use_orbitfl_style()

    device = get_device()
    print(f"Device  : {device}")
    print(f"Trace   : {args.trace}")
    print(f"Δ sweep : {delta_list} steps = {[d * cfg.sim.timestep_s for d in delta_list]} s")
    print(f"Seeds   : {seeds}")
    print(f"Epochs  : {args.epochs}")
    print(f"Run dir : {run_dir}")
    print()

    trace = Trace.load(args.trace)
    N_leo = trace.n_leo
    T = trace.n_timesteps
    C = cfg.world.n_channels
    W = cfg.model.window

    print(f"Trace   : N_leo={N_leo}, T={T}, C={C}, W={W}")
    print()

    all_rows: list[dict] = []

    for delta_steps in delta_list:
        delta_s = delta_steps * cfg.sim.timestep_s
        print(f"-- Δ={delta_steps} steps ({delta_s} s) --")

        for seed in seeds:
            t_seed = time.perf_counter()
            seed_everything(seed)

            # World and datasets for this (horizon, seed)
            world = OccupancyWorld(cfg.world, seed=seed,
                                   timestep_s=cfg.sim.timestep_s, n_steps=T)
            train_ds, probe_ds = _build_split_datasets(world, trace, N_leo, W, delta_steps)
            print(f"  seed={seed}  train={len(train_ds):,}  probe={len(probe_ds):,}", flush=True)

            # The position-only model is trained once per (horizon, seed); it does not depend
            # on use_position
            seed_everything(seed)
            eph_model = EphemerisOnlyForecaster(C, hidden=32).to(device)
            _train_eph(eph_model, train_ds, device, n_epochs=args.epochs)

            for use_position in [True, False]:
                seed_everything(seed)  # fresh weights for each condition
                cnn = OccupancyForecaster(C, W, use_position=use_position,
                                          hidden=cfg.model.hidden).to(device)
                _train_cnn(cnn, train_ds, use_position, device, n_epochs=args.epochs)

                metrics = _evaluate_probe(cnn, eph_model, probe_ds, use_position, device)

                row = {
                    "delta_steps": delta_steps,
                    "delta_s": delta_s,
                    "use_position": use_position,
                    "seed": seed,
                    **metrics,
                }
                all_rows.append(row)

                q1_pass = (metrics["ba_cnn"] > metrics["ba_pers"] and
                           metrics["ba_cnn"] > metrics["ba_eph"])
                q3_gap = metrics["ba_cnn"] - metrics["ba_eph"]
                print(
                    f"    pos={str(use_position)[0]}  "
                    f"ba_cnn={metrics['ba_cnn']:.4f}  "
                    f"ba_pers={metrics['ba_pers']:.4f}  "
                    f"ba_eph={metrics['ba_eph']:.4f}  "
                    f"Q1={'PASS' if q1_pass else 'FAIL'}  "
                    f"Q3_gap={q3_gap:+.4f}"
                )

            elapsed = time.perf_counter() - t_seed
            print(f"  ({elapsed:.0f}s)")

        print()

    # Save the results
    df = pd.DataFrame(all_rows)
    metrics_path = run_dir / "metrics.parquet"
    df.to_parquet(metrics_path, index=False)
    print(f"Metrics saved -> {metrics_path}")

    # Mean and std across seeds for each (horizon, use_position)
    summary_cols = ["ba_cnn", "ba_eph", "ba_pers", "skill_cnn_pers", "skill_cnn_eph"]
    summary = df.groupby(["delta_steps", "use_position"])[summary_cols].agg(["mean", "std"])
    print("\n-- Summary (mean ± std across seeds) --")
    print(summary.to_string())

    # The three questions
    best_row = df[df["use_position"]].groupby("delta_steps")["ba_cnn"].mean()
    chosen_delta = int(best_row.idxmax())
    print("\n-- Questions --")
    print(f"Q1: CNN beats both baselines at Δ={chosen_delta} steps "
          f"({chosen_delta * cfg.sim.timestep_s} s)?")
    at_chosen = df[(df["delta_steps"] == chosen_delta) & df["use_position"]]
    q1 = (at_chosen["ba_cnn"].mean() > at_chosen["ba_pers"].mean() and
          at_chosen["ba_cnn"].mean() > at_chosen["ba_eph"].mean())
    print(f"   {'PASS' if q1 else 'FAIL'}")

    pos_delta = (df[df["use_position"]].groupby("delta_steps")["ba_cnn"].mean()
                 - df[~df["use_position"]].groupby("delta_steps")["ba_cnn"].mean())
    print(f"Q2: Positional-encoding delta at chosen Δ={chosen_delta}: "
          f"{pos_delta.get(chosen_delta, float('nan')):+.4f}")

    eph_gap = df[df["use_position"]].groupby("delta_steps").apply(
        lambda g: (g["ba_cnn"] - g["ba_eph"]).mean()
    )
    print(f"Q3: CNN − EphOnly at chosen Δ={chosen_delta}: "
          f"{eph_gap.get(chosen_delta, float('nan')):+.4f}  "
          f"(near zero: skill comes from geography)")

    print(f"\nHorizon with the highest CNN BA: {chosen_delta} steps "
          f"= {chosen_delta * cfg.sim.timestep_s} s")

    # Figure
    _plot_ba_vs_delta(df, run_dir)
    print(f"Figure   -> {run_dir}/centralized_ba_vs_delta.{{png,pdf}}")
    print(f"Run dir  -> {run_dir}")


if __name__ == "__main__":
    main()
