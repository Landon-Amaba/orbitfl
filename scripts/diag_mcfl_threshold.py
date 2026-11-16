r"""Diagnostics for the MCFL distance-threshold sweep (results/mcfl_threshold).

Replays a stored run from its own config.yaml and records two things the sweep
did not log. Training, aggregation and evaluation are unchanged: the subclasses
below only observe, and tests/test_diag_mcfl_threshold.py checks that the
metrics match the plain runner.

  cluster_events.parquet    one row per (clustering event, d_th): number of
                            clusters, singletons, largest cluster, distances,
                            and ARI against regions and planes
  operating_points.parquet  one row per (evaluated round, model): positive
                            rates, BA at threshold 0.5 and BA at the best
                            threshold, for the global and personalized models

Settings are in configs/diagnostics/mcfl_threshold_diag.yaml.

Usage:
  # Cluster make-up: a 10-round replay scores every d_th at the first clustering.
  python scripts/diag_mcfl_threshold.py run --mode membership --round-budget 10 \
      --source-run results/mcfl_threshold/<dir> --trace data/traces/leo180 --out results/

  # Operating point: a full replay of one stored run.
  python scripts/diag_mcfl_threshold.py run --mode threshold \
      --source-run results/mcfl_threshold/<dir> --trace data/traces/leo180 --out results/

  python scripts/diag_mcfl_threshold.py summarize --out results/
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.config import Config, load_config, to_dict  # noqa: E402
from orbitfl.experiments.runner import _region_schedule  # noqa: E402
from orbitfl.fl.aggregation import StateDict  # noqa: E402
from orbitfl.fl.clustering.base import FLState  # noqa: E402
from orbitfl.fl.clustering.similarity import SimilarityReactive, _pairwise_cosine  # noqa: E402
from orbitfl.fl.loop import FederatedRunner, RunResult  # noqa: E402
from orbitfl.metrics.forecast import balanced_accuracy  # noqa: E402
from orbitfl.trace.reader import Trace  # noqa: E402
from orbitfl.utils.runs import create_run_dir  # noqa: E402
from orbitfl.utils.seeding import seed_everything  # noqa: E402
from orbitfl.world.occupancy import OccupancyWorld  # noqa: E402

_DEFAULT_SETTINGS = Path(__file__).parent.parent / "configs/diagnostics/mcfl_threshold_diag.yaml"
_DIAG_EXPERIMENT_PREFIX = "mcfl_threshold_diag_"
_DECISION_THRESHOLD = 0.5  # threshold the FL loop applies when logging BA


# --- Measurements ---


def partition_at_threshold(dist: np.ndarray, d_th: float) -> np.ndarray:
    """Cluster a distance matrix the same way SimilarityReactive._cluster does.

    Parameters
    ----------
    dist : (N, N) float64  Pairwise distances in [0, 2].
    d_th : float           Average-linkage distance threshold.

    Returns
    -------
    np.ndarray  (N,) int  Cluster label per satellite.
    """
    agg = AgglomerativeClustering(
        n_clusters=None, distance_threshold=d_th, metric="precomputed", linkage="average"
    )
    return agg.fit_predict(dist).astype(np.int32)


def partition_summary(labels: np.ndarray, min_real_size: int) -> dict[str, float]:
    """Describe the shape of a partition beyond its number of clusters.

    Parameters
    ----------
    labels        : (N,) int  Cluster label per satellite.
    min_real_size : int       Smallest cluster size counted as a real cluster.

    Returns
    -------
    dict  k, n_small, n_real, largest, and frac_in_real (satellites in real clusters).
    """
    sizes = np.bincount(labels - labels.min())  # (max_label + 1,) members per label
    sizes = sizes[sizes > 0]  # (K,) members per non-empty cluster
    real = sizes[sizes >= min_real_size]  # (n_real,)
    return {
        "k": int(sizes.size),
        "n_small": int((sizes < min_real_size).sum()),
        "n_real": int(real.size),
        "largest": int(sizes.max()),
        "frac_in_real": float(real.sum() / labels.size),
    }


def ari_against_null(
    labels: np.ndarray, reference: np.ndarray, n_permutations: int, seed: int
) -> dict[str, float]:
    """ARI of a partition against a reference, with a permutation null.

    The null permutes the labels, which keeps the cluster sizes.

    Parameters
    ----------
    labels         : (N,) int  Partition under test.
    reference      : (N,) int  Reference labels (regions or planes).
    n_permutations : int       Label permutations in the null.
    seed           : int       Seed of the permutation stream.

    Returns
    -------
    dict  ari, null_mean, null_sd, and z (NaN when the null has no spread).
    """
    observed = float(adjusted_rand_score(reference, labels))
    rng = np.random.default_rng(seed)
    null = np.array(  # (n_permutations,) ARI of shuffled labels
        [adjusted_rand_score(reference, rng.permutation(labels)) for _ in range(n_permutations)]
    )
    sd = float(null.std())
    return {
        "ari": observed,
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "z": (observed - float(null.mean())) / sd if sd > 0 else float("nan"),
    }


def modal_labels(schedule_window: np.ndarray) -> np.ndarray:
    """
    Most frequent label per satellite over a span of rounds.

    Parameters
    ----------
    schedule_window : (n_rounds, N) int  Non-negative label per round per satellite.

    Returns
    -------
    np.ndarray  (N,) int  Modal label; ties go to the lowest label.
    """
    return np.array(
        [np.bincount(column).argmax() for column in schedule_window.T], dtype=np.int32
    )


def operating_point(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Where a model sits on its ROC curve, and the best BA that curve allows.

    best_threshold is chosen on the same samples it is scored on, so best_ba is an
    upper bound useful for comparing two curves. It is not a score that could be
    achieved in deployment.

    Parameters
    ----------
    y_true : (S,) float  Binary labels.
    y_prob : (S,) float  Predicted probabilities.

    Returns
    -------
    dict  base_rate, mean_prob, pred_pos_rate, tpr, tnr, ba_at_fixed, best_ba, best_threshold.
    """
    pred = y_prob >= _DECISION_THRESHOLD  # (S,) bool
    positive = y_true >= _DECISION_THRESHOLD  # (S,) bool
    fpr, tpr, thresholds = roc_curve(positive, y_prob)  # each (n_points,)
    ba_curve = (tpr + 1.0 - fpr) / 2.0  # (n_points,)
    finite = np.isfinite(thresholds)  # (n_points,) drops sklearn's leading +inf point
    best = int(np.argmax(np.where(finite, ba_curve, -np.inf)))
    return {
        "base_rate": float(positive.mean()),
        "mean_prob": float(y_prob.mean()),
        "pred_pos_rate": float(pred.mean()),
        "tpr": float(pred[positive].mean()),
        "tnr": float((~pred[~positive]).mean()),
        "ba_at_fixed": balanced_accuracy(positive.astype(float), pred.astype(float)),
        "best_ba": float(ba_curve[best]),
        "best_threshold": float(thresholds[best]),
    }


def offdiagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Upper-triangle entries of a square matrix.

    Parameters
    ----------
    matrix : (N, N) float

    Returns
    -------
    np.ndarray  (N * (N - 1) / 2,) float
    """
    return matrix[np.triu_indices_from(matrix, k=1)]


# --- Observers ---


class RecordingSimilarity(SimilarityReactive):
    """
    SimilarityReactive that records each clustering event without changing it.

    Attributes
    ----------
    events        : list[dict]  One record per event: round, window_start, dist,
                                mean cosines, and update counts per satellite.
    current_round : int         Round of the latest membership() call.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Forward all arguments to SimilarityReactive and start empty records."""
        super().__init__(*args, **kwargs)
        self.events: list[dict] = []
        self.current_round: int = 0
        self._window_first_round: int = 0
        self._updates_in_window: np.ndarray = np.zeros(self._n_leo, dtype=np.int32)  # (N_leo,)

    def membership(self, round_idx: int, state: FLState) -> np.ndarray:
        """
        Note the round, then delegate unchanged.

        Parameters
        ----------
        round_idx : int      Current round.
        state     : FLState  Previous round's updates.

        Returns
        -------
        np.ndarray  (N_leo,) int  Cluster id per satellite.
        """
        self.current_round = round_idx
        return super().membership(round_idx, state)

    def _start_bootstrap(self, round_idx: int) -> None:
        """
        Reset the window records, then delegate unchanged.

        Parameters
        ----------
        round_idx : int  Round the new bootstrap window starts.
        """
        self._window_first_round = round_idx
        self._updates_in_window[:] = 0
        super()._start_bootstrap(round_idx)

    def _accumulate(self, state: FLState) -> None:
        """
        Count which satellites reported, then delegate unchanged.

        Parameters
        ----------
        state : FLState  Carries the previous round's client_updates.
        """
        for update in state.client_updates:
            self._updates_in_window[update.sat_id] += 1
        super()._accumulate(state)

    def _cluster(self) -> None:
        """Cluster unchanged, then record the distances that clustering used."""
        super()._cluster()
        if self._accumulated is None or self._instantaneous is None:
            return
        inst_sim = _pairwise_cosine(self._instantaneous)  # (N, N)
        accum_sim = _pairwise_cosine(self._accumulated)  # (N, N)
        sim = self._gamma * inst_sim + (1.0 - self._gamma) * accum_sim  # (N, N)
        self.events.append(
            {
                "round": self.current_round,
                "window_start": self._window_first_round,
                "dist": np.clip(1.0 - sim, 0.0, 2.0),  # (N, N)
                "labels": self._current.copy(),  # (N,)
                "mean_cos_inst": float(offdiagonal(inst_sim).mean()),
                "mean_cos_accum": float(offdiagonal(accum_sim).mean()),
                "updates_in_window": self._updates_in_window.copy(),  # (N,)
            }
        )


class ObservingRunner(FederatedRunner):
    """FederatedRunner that also records the operating point of each evaluated model."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Forward all arguments to FederatedRunner and start empty records."""
        super().__init__(*args, **kwargs)
        self.operating_points: list[dict] = []
        self._captured: list[tuple[np.ndarray, np.ndarray]] = []

    def _predict(self, loader: object) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict unchanged and keep a reference to the output.

        Parameters
        ----------
        loader : DataLoader  Probe samples.

        Returns
        -------
        (y_true, y_prob) : each (S,) float
        """
        y_true, y_prob = super()._predict(loader)
        self._captured.append((y_true, y_prob))
        return y_true, y_prob

    def _record(self, model: str, logged_auc: float, logged_ba: float) -> None:
        """
        Turn the captured predictions into one operating-point record.

        Parameters
        ----------
        model      : str    "global" or "personalized".
        logged_auc : float  AUC the loop logged for these predictions.
        logged_ba  : float  BA the loop logged for these predictions.
        """
        y_true = np.concatenate([pair[0] for pair in self._captured])  # (S,)
        y_prob = np.concatenate([pair[1] for pair in self._captured])  # (S,)
        self._captured = []
        self.operating_points.append(
            {
                "round": getattr(self._strategy, "current_round", -1),
                "model": model,
                "auc": logged_auc,
                "ba": logged_ba,
                **operating_point(y_true, y_prob),
            }
        )

    def _evaluate_probe(self, global_state: StateDict) -> tuple[float, float]:
        """
        Evaluate the global model unchanged and record its operating point.

        Parameters
        ----------
        global_state : StateDict  Current global model.

        Returns
        -------
        (auc, balanced_accuracy) : tuple[float, float]
        """
        self._captured = []
        result = super()._evaluate_probe(global_state)
        self._record("global", *result)
        return result

    def _evaluate_personalized(
        self, cluster_models: dict[int, StateDict], membership: np.ndarray, global_state: StateDict
    ) -> tuple[float, float]:
        """
        Evaluate personalized models unchanged and record their pooled operating point.

        Parameters
        ----------
        cluster_models : dict[int, StateDict]  Current per-cluster models.
        membership     : (N_leo,) int          Cluster id per satellite.
        global_state   : StateDict             Fallback model.

        Returns
        -------
        (auc, balanced_accuracy) : tuple[float, float]
        """
        self._captured = []
        result = super()._evaluate_personalized(cluster_models, membership, global_state)
        self._record("personalized", *result)
        return result


# --- Event scoring ---


def score_event(
    event: dict,
    region_schedule: np.ndarray,
    plane_ids: np.ndarray,
    d_th_grid: list[float],
    settings: dict,
) -> list[dict]:
    """
    Score one clustering event at every threshold in the grid.

    Parameters
    ----------
    event           : dict                One RecordingSimilarity.events record.
    region_schedule : (R, N) int          Ground region per round per satellite.
    plane_ids       : (N,) int            Orbital plane per satellite.
    d_th_grid       : list[float]         Thresholds to score.
    settings        : dict                Parsed mcfl_threshold_diag.yaml.

    Returns
    -------
    list[dict]  One row per threshold.
    """
    n_perm = int(settings["ari_null_permutations"])
    null_seed = int(settings["ari_null_seed"])
    min_size = int(settings["min_real_cluster_size"])
    round_idx, start = int(event["round"]), int(event["window_start"])
    window = region_schedule[start : round_idx + 1]  # (n_rounds, N)
    references = {
        "region_now": region_schedule[round_idx],  # (N,)
        "region_start": region_schedule[start],  # (N,)
        "region_modal": modal_labels(window),  # (N,)
        "plane": plane_ids,  # (N,)
    }
    updates = event["updates_in_window"]  # (N,)
    distances = offdiagonal(event["dist"])  # (N * (N - 1) / 2,)
    rows: list[dict] = []
    for d_th in d_th_grid:
        labels = partition_at_threshold(event["dist"], d_th)  # (N,)
        row: dict = {
            "round": round_idx,
            "window_start": start,
            "d_th": d_th,
            **partition_summary(labels, min_size),
            "mean_dist": float(distances.mean()),
            "p05_dist": float(np.quantile(distances, 0.05)),
            "p50_dist": float(np.quantile(distances, 0.50)),
            "p95_dist": float(np.quantile(distances, 0.95)),
            "mean_cos_inst": event["mean_cos_inst"],
            "mean_cos_accum": event["mean_cos_accum"],
            "n_without_updates": int((updates == 0).sum()),
        }
        sizes = np.bincount(labels)  # (K,) members per label
        in_real = sizes[labels] >= min_size  # (N,) bool
        row["mean_updates_real"] = float(updates[in_real].mean()) if in_real.any() else np.nan
        row["mean_updates_small"] = float(updates[~in_real].mean()) if (~in_real).any() else np.nan
        for name, reference in references.items():
            scores = ari_against_null(labels, reference, n_perm, null_seed)
            row.update({f"{name}_{key}": value for key, value in scores.items()})
        rows.append(row)
    return rows


# --- Running one replay ---


def build_observed_runner(
    cfg: Config, trace: Trace, seed: int, device: torch.device
) -> tuple[ObservingRunner, RecordingSimilarity]:
    """Build the runner, seeding in the same order as experiments.runner.run_one.

    Parameters
    ----------
    cfg    : Config        Resolved config; its strategy must be "similarity".
    trace  : Trace         Loaded trace.
    seed   : int           Run seed.
    device : torch.device  Compute device.

    Returns
    -------
    (runner, strategy) : tuple[ObservingRunner, RecordingSimilarity]

    Raises
    ------
    ValueError  If the config's strategy is not similarity.
    """
    if cfg.fl.strategy.name != "similarity":
        raise ValueError(f"Expected a similarity run, got {cfg.fl.strategy.name!r}.")
    params = dict(cfg.fl.strategy.params)
    seed_everything(seed)
    world = OccupancyWorld(
        cfg.world, seed=seed, timestep_s=cfg.sim.timestep_s, n_steps=trace.n_timesteps
    )
    strategy = RecordingSimilarity(
        n_leo=trace.n_leo,
        n_clusters=trace.n_meo,
        T_init=int(params["T_init"]),
        recluster_every_R=params.get("recluster_every_R"),
        gamma=float(params["gamma"]),
        d_th=float(params["d_th"]),
        seed=seed,
        update_reference=str(params["update_reference"]),
    )
    return ObservingRunner(cfg, trace, world, strategy, device=device), strategy


def compare_to_source(replay: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """
    Line a replay's metrics up with the stored run it reproduces.

    Parameters
    ----------
    replay : pd.DataFrame  Replay metrics, one row per round.
    source : pd.DataFrame  Stored metrics of the source run.

    Returns
    -------
    pd.DataFrame  One row per shared round with stored_ and replay_ columns.
    """
    columns = ["n_clusters", "ba_probe", "auc_probe", "ba_personalized", "auc_personalized"]
    merged = source[["round", *columns]].merge(
        replay[["round", *columns]], on="round", suffixes=("_stored", "_replay")
    )
    return merged


def _provenance(source_run: Path, trace_path: Path, round_budget: int | None) -> dict:
    """Record what produced a diagnostic run.

    Includes a hash of this script, because the git hash in the run directory name
    does not identify the code when the working tree has uncommitted changes.

    Parameters
    ----------
    source_run   : Path        Stored run being replayed.
    trace_path   : Path        Trace directory.
    round_budget : int | None  Budget override, if any.

    Returns
    -------
    dict  Plain values ready for YAML.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "source_run": str(source_run),
        "trace": str(trace_path),
        "round_budget_override": round_budget,
        "git_head": head,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
    }


def run_replay(args: argparse.Namespace) -> Path:
    """
    Replay one stored run on CPU and write the diagnostic records to a new directory.

    Parameters
    ----------
    args : argparse.Namespace  Parsed `run` arguments.

    Returns
    -------
    Path  The new run directory.

    Raises
    ------
    ValueError  If the source run does not hold exactly one seed.
    """
    settings = yaml.safe_load(args.settings.read_text())
    override = {"fl": {"round_budget": args.round_budget}} if args.round_budget else None
    cfg = load_config(args.source_run / "config.yaml", override_dict=override)
    seeds = list(cfg.experiment.seeds)
    if len(seeds) != 1:
        raise ValueError(f"Expected one seed in {args.source_run}, got {seeds}.")
    seed = int(seeds[0])

    # create_run_dir never reuses an existing directory.
    run_dir = create_run_dir(_DIAG_EXPERIMENT_PREFIX + args.mode, args.out, allow_dirty=True)
    (run_dir / "config.yaml").write_text(yaml.dump(to_dict(cfg), default_flow_style=False))
    (run_dir / "provenance.yaml").write_text(
        yaml.dump(_provenance(args.source_run, args.trace, args.round_budget))
    )

    trace = Trace.load(args.trace)
    (run_dir / "trace_meta.yaml").write_text(yaml.dump(trace.meta.to_dict(orient="records")[0]))

    # CPU, as in the sweep, so the replay follows the same code path.
    runner, strategy = build_observed_runner(cfg, trace, seed, torch.device("cpu"))
    result: RunResult = runner.run()
    metrics = result.metrics.assign(seed=seed)
    metrics.to_parquet(run_dir / "metrics.parquet", index=False)

    source = pd.read_parquet(args.source_run / "metrics.parquet")
    compare_to_source(metrics, source).to_parquet(run_dir / "reproduction.parquet", index=False)

    schedule = _region_schedule(cfg, trace)  # (R, N)
    grid = settings["d_th_grid"]
    rows = [
        row
        for event in strategy.events
        for row in score_event(event, schedule, trace.leo_plane_ids, grid, settings)
    ]
    run_d_th = float(cfg.fl.strategy.params["d_th"])
    pd.DataFrame(rows).assign(seed=seed, run_d_th=run_d_th).to_parquet(
        run_dir / "cluster_events.parquet", index=False
    )
    pd.DataFrame(runner.operating_points).assign(seed=seed, run_d_th=run_d_th).to_parquet(
        run_dir / "operating_points.parquet", index=False
    )
    np.savez_compressed(
        run_dir / "memberships.npz",
        rounds=np.array([event["round"] for event in strategy.events]),  # (n_events,)
        labels=np.stack([event["labels"] for event in strategy.events]),  # (n_events, N)
        updates=np.stack([event["updates_in_window"] for event in strategy.events]),
    )
    print(f"Replay complete: seed={seed} d_th={run_d_th} -> {run_dir}", flush=True)
    return run_dir


# --- Summary ---


def _load_all(root: Path, filename: str) -> pd.DataFrame:
    """Concatenate one parquet file across every run directory under root.

    Parameters
    ----------
    root     : Path  A mcfl_threshold_diag_<mode> directory.
    filename : str   Parquet file name.

    Returns
    -------
    pd.DataFrame  Empty when no run holds the file.
    """
    frames = [pd.read_parquet(path) for path in sorted(root.glob(f"*/{filename}"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(out_root: Path) -> None:
    """Print the partition and operating-point tables.

    Parameters
    ----------
    out_root : Path  Results root holding the mcfl_threshold_diag_* directories.
    """
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    events = _load_all(out_root / f"{_DIAG_EXPERIMENT_PREFIX}membership", "cluster_events.parquet")
    if not events.empty:
        columns = [
            "k", "n_real", "n_small", "largest", "frac_in_real", "n_without_updates",
            "mean_updates_real", "mean_updates_small", "region_now_ari", "region_now_z",
            "region_modal_ari", "region_modal_z", "plane_ari", "plane_z",
        ]
        print("Partition at the first clustering (mean over seeds)")
        print(events.groupby("d_th")[columns].mean().sort_index(ascending=False).round(3))
        print("\nPer seed")
        print(events.sort_values(["d_th", "seed"], ascending=[False, True])[
            ["d_th", "seed", "k", "n_real", "largest", "region_now_ari", "plane_ari"]
        ].round(3).to_string(index=False))

    root = out_root / f"{_DIAG_EXPERIMENT_PREFIX}threshold"
    points = _load_all(root, "operating_points.parquet")
    if not points.empty:
        last = points[points["round"] == points["round"].max()]
        columns = [
            "ba", "auc", "best_ba", "best_threshold", "pred_pos_rate", "base_rate",
            "mean_prob", "tpr", "tnr",
        ]
        print("\nOperating point at the final round (mean over seeds)")
        print(last.groupby(["run_d_th", "model"])[columns].mean().round(4))
        print("\nPer seed")
        print(last.sort_values(["model", "run_d_th", "seed"])[
            ["model", "run_d_th", "seed", "ba", "auc", "best_ba", "pred_pos_rate"]
        ].round(4).to_string(index=False))
        schedule = _load_all(root, "cluster_events.parquet")
        own = schedule[np.isclose(schedule["d_th"], schedule["run_d_th"])]
        print("\nDistances at each clustering event (mean over seeds)")
        print(own.groupby(["run_d_th", "round"])[
            ["k", "n_real", "largest", "mean_dist", "p05_dist", "mean_cos_inst", "mean_cos_accum"]
        ].mean().round(4).to_string())


def main() -> None:
    """Parse arguments and dispatch to `run` or `summarize`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=2)[1])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Replay one stored run with observers attached.")
    run.add_argument("--mode", choices=["membership", "threshold"], required=True)
    run.add_argument("--source-run", type=Path, required=True)
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--out", type=Path, default=Path("results/"))
    run.add_argument("--round-budget", type=int, default=None)
    run.add_argument("--settings", type=Path, default=_DEFAULT_SETTINGS)

    summary = commands.add_parser("summarize", help="Print the diagnostic tables.")
    summary.add_argument("--out", type=Path, default=Path("results/"))

    args = parser.parse_args()
    if args.command == "run":
        run_replay(args)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
