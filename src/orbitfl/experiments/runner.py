"""Experiment runner.

run_experiment creates results/<exp_name>/<UTC-timestamp>_<git-hash>_<token>/,
writes the resolved config and the trace metadata into it, runs the FL loop for
every seed, and writes metrics.parquet, a summary table and figures. It refuses
to run from a git tree with uncommitted changes unless allow_dirty is set.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from orbitfl.config import Config, load_config, to_dict
from orbitfl.fl.clustering.base import ClusteringStrategy
from orbitfl.fl.clustering.ephemeris import EphemerisClustering
from orbitfl.fl.clustering.plane import PlaneClustering
from orbitfl.fl.clustering.random_churn import RandomChurnClustering
from orbitfl.fl.clustering.similarity import SimilarityReactive
from orbitfl.fl.loop import FederatedRunner, RunResult
from orbitfl.trace.reader import Trace
from orbitfl.utils.device import get_device
from orbitfl.utils.plotting import save_figure, use_orbitfl_style
from orbitfl.utils.runs import create_run_dir
from orbitfl.utils.seeding import seed_everything
from orbitfl.world.occupancy import OccupancyWorld

# --- Strategy factory ---


def _region_schedule(cfg: Config, trace: Trace) -> np.ndarray:
    """Build the (round_budget, N_leo) ground-region schedule for region clustering.

    Entry [r, k] is the region id of LEO k's sub-satellite point at the first
    timestep of round r (r * steps_per_round, clamped to the trace length), looked
    up in the same region map the occupancy world uses. Region ids are 0..R-1 with
    no -1, because region_at falls back to the catch-all region.

    Parameters
    ----------
    cfg   : Config  Resolved config (world regions and round timing).
    trace : Trace   Loaded trace (sub-satellite tracks).

    Returns
    -------
    np.ndarray  (round_budget, N_leo) int32  region id per satellite per round.
    """
    from orbitfl.world.regions import build_region_map, region_at

    region_map = build_region_map(cfg.world)
    spr = max(1, cfg.fl.round_duration_s // cfg.sim.timestep_s)
    R = cfg.fl.round_budget
    N = trace.n_leo
    T = trace.n_timesteps

    schedule = np.zeros((R, N), dtype=np.int32)  # (R, N_leo) region id per round
    for k in range(N):
        subpts = trace.subpoints(k)  # (T, 2) [lat_deg, lon_deg]
        if len(subpts) == 0:
            continue
        for r in range(R):
            t = min(r * spr, T - 1, len(subpts) - 1)
            schedule[r, k] = region_at(float(subpts[t, 0]), float(subpts[t, 1]), region_map)
    return schedule


def _build_strategy(
    cfg: Config,
    trace: Trace,
    seed: int,
) -> ClusteringStrategy:
    """Instantiate the clustering strategy named in cfg.fl.strategy.

    Parameters
    ----------
    cfg   : Config  Resolved config.
    trace : Trace   Loaded trace.
    seed  : int     Run seed (used by random_churn and similarity).

    Returns
    -------
    ClusteringStrategy

    Raises
    ------
    ValueError  If cfg.fl.strategy.name is not recognised.
    """
    name = cfg.fl.strategy.name
    params: dict = dict(cfg.fl.strategy.params)

    if name == "ephemeris":
        if params.get("use_meo_proximity", True):
            return EphemerisClustering(
                trace=trace,
                round_duration_s=cfg.fl.round_duration_s,
                round_budget=cfg.fl.round_budget,
                use_meo_proximity=True,
            )
        # Region clustering: the ground region under each satellite at the round's time.
        schedule = _region_schedule(cfg, trace)
        return EphemerisClustering.from_region_schedule(
            schedule, round_budget=cfg.fl.round_budget
        )

    if name == "plane":
        return PlaneClustering(trace=trace)

    if name == "similarity":
        return SimilarityReactive(
            n_leo=trace.n_leo,
            n_clusters=trace.n_meo,
            T_init=int(params.get("T_init", 10)),
            recluster_every_R=params.get("recluster_every_R", None),
            gamma=float(params.get("gamma", 0.5)),
            d_th=float(params.get("d_th", 1.0)),
            seed=seed,
            update_reference=str(params.get("update_reference", "last_global")),
        )

    if name == "random_churn":
        return RandomChurnClustering(
            trace=trace,
            n_leo=trace.n_leo,
            n_clusters=trace.n_meo,
            round_duration_s=cfg.fl.round_duration_s,
            seed=seed,
        )

    raise ValueError(
        f"Unknown strategy '{name}'. "
        "Valid choices: ephemeris, plane, similarity, random_churn."
    )


# --- Figure generation ---


def _plot_run_result(result: RunResult, run_dir: Path) -> None:
    """
    Write the standard per-run figures to run_dir/.

    Figure 1: ba_probe + auc_probe vs round
    Figure 2: comm bytes (cumulative) vs round
    """
    import matplotlib.pyplot as plt

    df = result.metrics

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: accuracy vs round. Probe metrics are logged only every eval_interval
    # rounds, so NaN rows are dropped to connect the evaluated points with a line.
    ax = axes[0]
    ba = df[["round", "ba_probe"]].dropna()
    auc = df[["round", "auc_probe"]].dropna()
    ax.plot(ba["round"], ba["ba_probe"], marker="o", markersize=3,
            label="BA (probe)", linewidth=1.5)
    ax.plot(auc["round"], auc["auc_probe"], marker="o", markersize=3,
            label="AUC (probe)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title(f"{result.strategy_name}: single seed")
    ax.legend()

    # Panel B: cumulative communication
    ax = axes[1]
    cumul = (df["comm_bytes_up"] + df["comm_bytes_down"]).cumsum() / 1e6  # MB
    ax.plot(df["round"], cumul, linewidth=1.5, color="darkorange")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative comm (MB)")
    ax.set_title("Communication cost")

    fig.suptitle(
        f"{result.strategy_name} · churn={result.churn_rate:.3f}/orbit",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, run_dir, "fl_run")
    plt.close(fig)


# --- Runner ---


def run_one(
    cfg: Config,
    trace: Trace,
    seed: int,
    run_dir: Path,
    override_dict: dict | None = None,
) -> RunResult:
    """Run one seed and write its outputs to run_dir.

    Parameters
    ----------
    cfg          : Config    Resolved config for this run.
    trace        : Trace     Loaded trace (shared across seeds).
    seed         : int       Seed for this run.
    run_dir      : Path      Directory where outputs are written.
    override_dict: dict|None Override dict used to build cfg.

    Returns
    -------
    RunResult  Metrics and metadata for this seed.
    """
    seed_everything(seed)
    device = get_device()

    world = OccupancyWorld(cfg.world, seed=seed,
                           timestep_s=cfg.sim.timestep_s,
                           n_steps=trace.n_timesteps)
    strategy = _build_strategy(cfg, trace, seed)

    t0 = time.perf_counter()
    runner = FederatedRunner(cfg, trace, world, strategy, device=device)
    result = runner.run()
    elapsed = time.perf_counter() - t0

    # Save per-seed metrics
    seed_dir = run_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_parquet(seed_dir / "metrics.parquet", index=False)

    print(
        f"  seed={seed:2d}  {result.strategy_name:<20s}"
        f"  final_ba={result.metrics['ba_probe'].iloc[-1]:.4f}"
        f"  final_auc={result.metrics['auc_probe'].iloc[-1]:.4f}"
        f"  overhead={result.metrics['overhead_rounds'].iloc[-1]}"
        f"  ({elapsed:.0f}s)"
    )
    return result


def run_experiment(
    config_path: Path,
    trace_path: Path,
    override_path: Path | None = None,
    override_dict: dict | None = None,
    out_root: Path = Path("results/"),
    allow_dirty: bool = False,
    array_idx: int | None = None,
) -> Path:
    """Run an experiment: load the config, run all seeds, aggregate, save outputs.

    Parameters
    ----------
    config_path  : Path        Base config YAML.
    trace_path   : Path        Trace directory.
    override_path: Path|None   Optional override YAML, merged over the base.
    override_dict: dict|None   Per-run override from sweep expansion, applied after
                               override_path (used by slurm_submit).
    out_root     : Path        Results root.
    allow_dirty  : bool        Passed to create_run_dir.
    array_idx    : int|None    If set, run only the seed at this index (SLURM arrays).

    Returns
    -------
    Path  The run directory that was created.
    """
    use_orbitfl_style()

    cfg = load_config(
        config_path,
        override_path=override_path,
        override_dict=override_dict or None,
    )

    seeds: list[int] = list(cfg.experiment.seeds)
    if array_idx is not None:
        if array_idx >= len(seeds):
            raise ValueError(
                f"array_idx={array_idx} is out of range for seeds list of length {len(seeds)}."
            )
        seeds = [seeds[array_idx]]

    run_dir = create_run_dir(cfg.experiment.name, out_root, allow_dirty=allow_dirty)

    # Save the resolved config and trace metadata
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(to_dict(cfg), f, default_flow_style=False)

    trace = Trace.load(trace_path)
    with open(run_dir / "trace_meta.yaml", "w") as f:
        yaml.dump(trace.meta.to_dict(orient="records")[0], f)

    print(f"\nExperiment : {cfg.experiment.name}")
    print(f"Strategy   : {cfg.fl.strategy.name}")
    print(f"Seeds      : {seeds}")
    print(f"Rounds     : {cfg.fl.round_budget}")
    print(f"Trace      : {trace_path}  (N_leo={trace.n_leo}, T={trace.n_timesteps})")
    print(f"Run dir    : {run_dir}")
    print(f"Device     : {get_device()}\n")

    all_results: list[RunResult] = []
    for seed in seeds:
        result = run_one(cfg, trace, seed, run_dir)
        all_results.append(result)

    # Concatenate the seeds and add the strategy name (read by compare_arms.py)
    strategy_name = all_results[0].strategy_name
    all_metrics = pd.concat(
        [r.metrics.assign(seed=seeds[i], strategy=strategy_name)
         for i, r in enumerate(all_results)],
        ignore_index=True,
    )
    all_metrics.to_parquet(run_dir / "metrics.parquet", index=False)

    summary = (
        all_metrics.groupby("round")[["ba_probe", "auc_probe", "comm_bytes_up"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(run_dir / "summary.csv")

    # Figure with mean and standard deviation across seeds
    _plot_aggregate(all_metrics, all_results[0], run_dir)

    print(f"\nRun complete -> {run_dir}")
    final_ba = all_metrics.groupby("seed")["ba_probe"].last().mean()
    final_auc = all_metrics.groupby("seed")["auc_probe"].last().mean()
    print(f"Final BA (mean over seeds): {final_ba:.4f}")
    print(f"Final AUC (mean over seeds): {final_auc:.4f}")

    return run_dir


# --- Aggregate figure ---


def _plot_aggregate(
    df: pd.DataFrame,
    representative_result: RunResult,
    run_dir: Path,
) -> None:
    """BA and AUC vs round (mean and one standard deviation across seeds)."""
    import matplotlib.pyplot as plt

    from orbitfl.utils.plotting import mean_std_by_round

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Probe metrics are NaN on rounds skipped by eval_interval; mean_std_by_round
    # drops those rounds.
    for ax, col, ylabel, label in [
        (axes[0], "ba_probe", "Balanced accuracy", "BA"),
        (axes[1], "auc_probe", "AUC", "AUC"),
    ]:
        rounds, mean, std = mean_std_by_round(df, col)
        ax.plot(rounds, mean, marker="o", markersize=3, linewidth=1.5, label=label)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{representative_result.strategy_name} "
            f"(n={df['seed'].nunique()} seeds)"
        )

    fig.suptitle(
        f"{representative_result.strategy_name} · "
        f"churn={representative_result.churn_rate:.3f}/orbit",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, run_dir, "fl_aggregate")
    plt.close(fig)
