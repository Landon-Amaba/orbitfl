r"""Compare clustering strategies across the runs of one experiment and draw figures.

Reads every run folder under the given results directory and writes:

  efficiency     BA vs round (global model solid, personalized dashed), rounds
                 and communication needed to reach a target BA, and
                 cluster-discovery overhead, per strategy
  comparison     BA vs round, cumulative communication, and ARI against the
                 ground regions, per strategy
  contrast       final BA vs world.predictability_contrast (only if the runs
                 cover at least two contrast levels)
  concept_shift  final BA vs world.concept_shift (only if the runs cover at
                 least two shift levels)

It also prints a summary table with one row per strategy and sync interval.

Strategy keys used in the code and in the tables:
  EphemerisRegion        ephemeris strategy with ground-region clusters (GTRFL)
  EphemerisMEO           ephemeris strategy with nearest-MEO clusters
  PlaneClustering        one cluster per orbital plane
  RandomChurnClustering  random clusters with matched churn
  SimilarityReactive_R*  similarity clustering (MCFL), one key per re-cluster period R

Usage:
  python scripts/compare_arms.py results/efficiency \
      --out figures/regenerated/efficiency --sync 20 --target 0.58
  python scripts/compare_arms.py results/concept_shift_hidden_180 \
      --out figures/regenerated/concept_shift_hidden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.metrics.efficiency import DEFAULT_TARGETS, comm_to_target, rounds_to_target
from orbitfl.utils.plotting import mean_std_by_round, save_figure, use_orbitfl_style

# Display label, colour and line style per strategy key (see _load_run_dir).
_ARM_META: dict[str, dict] = {
    "EphemerisRegion": {
        "label": "(a) GTRFL (proposed)", "color": "steelblue", "lw": 2.5, "ls": "-",
    },
    "EphemerisMEO": {
        "label": "(a′) GTRFL MEO-prox (control)", "color": "navy", "lw": 1.5, "ls": "--",
    },
    "PlaneClustering": {
        "label": "(c) Plane (P-Orion style)", "color": "firebrick", "lw": 1.5, "ls": "-.",
    },
    "RandomChurnClustering": {
        "label": "(d) Random-churn control", "color": "gray", "lw": 1.5, "ls": ":",
    },
    "SimilarityReactive": {
        "label": "(b) MCFL", "color": "darkorange", "lw": 1.5, "ls": "-.",
    },
}

# One colour per re-cluster period R.
_SIMILARITY_COLORS: dict[str, str] = {
    "5": "#e6851a", "10": "#b85c00", "20": "#e6b980", "inf": "#805000",
}
_SIMILARITY_PREFIX = "SimilarityReactive_R"
_SIMILARITY_FALLBACK_COLOR = "darkorange"

# Figure titles, matching the experiment names used in the thesis.
TITLE_EFFICIENCY = "Efficiency"
TITLE_CONCEPT_SHIFT_HIDDEN = "Concept shift, position hidden"
TITLE_CONCEPT_SHIFT_VISIBLE = "Concept shift, position visible"
TITLE_CONCEPT_SHIFT_UNKNOWN = "Concept shift"

_NAME_TO_CLASS = {
    "ephemeris": "EphemerisClustering",
    "plane": "PlaneClustering",
    "similarity": "SimilarityReactive",
    "random_churn": "RandomChurnClustering",
}


# --- Data loading ---


def _load_run_dir(path: Path) -> pd.DataFrame | None:
    """Load metrics.parquet from one run directory and add arm, sync_interval and run_dir.

    The `arm` key splits the ephemeris strategy by cluster type
    (use_meo_proximity false: "EphemerisRegion", true: "EphemerisMEO") and appends
    the re-cluster period R to similarity runs, so each R is drawn as its own line.

    Returns None if the directory has no metrics.parquet.
    """
    metrics_file = path / "metrics.parquet"
    config_file = path / "config.yaml"
    if not metrics_file.exists():
        return None

    df = pd.read_parquet(metrics_file)
    cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    fl = cfg.get("fl", {})

    # Skip a run that did not reach its own round_budget, for example a task stopped
    # by the time limit. A complete run ends at round == round_budget - 1. A run with
    # no recorded round_budget is kept. Runs whose budget is smaller than the rest
    # are removed in _load_all.
    round_budget = fl.get("round_budget")
    if round_budget is not None and "round" in df.columns and len(df):
        max_round = int(df["round"].max())
        if max_round < int(round_budget) - 1:
            print(
                f"  [skip] {path.name}: incomplete, reached round {max_round} "
                f"< budget {round_budget}; excluded from aggregation",
                file=sys.stderr,
            )
            return None
    df["round_budget"] = round_budget  # may be None -> NaN; read by _load_all guard B

    strat = fl.get("strategy", {})
    params = strat.get("params", {}) or {}

    # Strategy class name: from the column written by the runner, else from the config.
    if "strategy" in df.columns:
        strategy = str(df["strategy"].iloc[0])
    else:
        strategy = _NAME_TO_CLASS.get(strat.get("name", "unknown"), "unknown")
        df["strategy"] = strategy

    # Strategy key used for display.
    if strategy == "EphemerisClustering":
        arm = "EphemerisRegion" if params.get("use_meo_proximity") is False else "EphemerisMEO"
    elif strategy == "SimilarityReactive":
        R = params.get("recluster_every_R", None)
        r_str = "inf" if R is None else str(int(R))
        arm = f"SimilarityReactive_R{r_str}"
        # Only non-default settings extend the key, so runs with a different d_th or
        # update_reference form their own group.
        d_th = float(params.get("d_th", 1.0))
        reference = params.get("update_reference", "last_global")
        if d_th != 1.0 or reference != "last_global":
            arm += f"_dth{d_th:g}"
        if reference != "last_global":
            arm += f"_{reference}"
    else:
        arm = strategy

    df["arm"] = arm
    df["sync_interval"] = fl.get("sync_interval", np.nan)
    df["predictability_contrast"] = cfg.get("world", {}).get("predictability_contrast", 0.0)
    df["concept_shift"] = cfg.get("world", {}).get("concept_shift", 0.0)
    df["use_position"] = cfg.get("model", {}).get("use_position")  # None when unrecorded
    df["run_dir"] = str(path)
    return df


def _discover_run_dirs(root: Path) -> list[Path]:
    """Find all immediate subdirectories of root that contain metrics.parquet."""
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "metrics.parquet").exists()
    )


def _load_all(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate the metrics of all run directories.

    All runs in a comparison must share one round_budget. A run with a smaller
    budget than the largest one present (for example a short test run in the same
    folder) is dropped, so it cannot enter the averages or replace a full run with
    the same (strategy, sync, seed). Runs with no recorded budget are kept.
    """
    frames = [d for p in paths if (d := _load_run_dir(p)) is not None]
    if not frames:
        raise FileNotFoundError(f"No valid run directories found in: {paths}")
    df = pd.concat(frames, ignore_index=True)

    budgets = (
        df["round_budget"].dropna()
        if "round_budget" in df.columns else pd.Series(dtype=float)
    )
    if len(budgets) and (budgets != budgets.max()).any():
        max_budget = int(budgets.max())
        under = df["round_budget"].notna() & (df["round_budget"] != max_budget)
        for rd in sorted(df.loc[under, "run_dir"].unique()):
            b = int(df.loc[df["run_dir"] == rd, "round_budget"].iloc[0])
            print(
                f"  [skip] {Path(rd).name}: round_budget {b} < comparison budget "
                f"{max_budget}; excluded (fixed-budget rule)",
                file=sys.stderr,
            )
        df = df[~under].reset_index(drop=True)
    return df


# --- Per-run efficiency table ---


def _per_run_table(df: pd.DataFrame, targets: tuple[float, ...]) -> pd.DataFrame:
    """One row per run (seed), with efficiency and summary metrics.

    If a (strategy, sync, contrast, shift, seed) combination was run more than
    once, only the latest run is kept, so every seed counts once.
    """
    rows: list[dict] = []
    for run_dir, g in df.groupby("run_dir"):
        ev = g.dropna(subset=["ba_probe"])
        ari = (
            g["ari_vs_regions"].dropna()
            if "ari_vs_regions" in g.columns else pd.Series(dtype=float)
        )
        rec: dict = {
            "run_dir": run_dir,
            "arm": g["arm"].iloc[0],
            "sync": g["sync_interval"].iloc[0],
            "contrast": float(g["predictability_contrast"].iloc[0])
            if "predictability_contrast" in g.columns else 0.0,
            "concept_shift": float(g["concept_shift"].iloc[0])
            if "concept_shift" in g.columns else 0.0,
            "use_position": g["use_position"].iloc[0] if "use_position" in g.columns else None,
            "seed": g["seed"].iloc[0] if "seed" in g.columns else 0,
            "final_ba": float(ev["ba_probe"].iloc[-1]) if len(ev) else np.nan,
            "final_pers": (
                float(ev["ba_personalized"].iloc[-1])
                if len(ev) and "ba_personalized" in g.columns else np.nan
            ),
            "total_comm_gb": float((g["comm_bytes_up"] + g["comm_bytes_down"]).sum()) / 1e9,
            "overhead": int(g["overhead_rounds"].iloc[-1]) if "overhead_rounds" in g.columns else 0,
            "final_ari": float(ari.iloc[-1]) if len(ari) else np.nan,
            "comm_gb_target": comm_to_target(g, targets[-1]),
            # Number of clusters formed, as mean and maximum over the run's rounds. NaN for
            # runs whose metrics have no n_clusters column.
            "mean_clusters": (
                float(g["n_clusters"].mean()) if "n_clusters" in g.columns else np.nan
            ),
            "max_clusters": (
                float(g["n_clusters"].max()) if "n_clusters" in g.columns else np.nan
            ),
        }
        for t in targets:
            rec[f"rounds_{t}"] = rounds_to_target(g, t)
        rows.append(rec)
    tbl = pd.DataFrame(rows).sort_values("run_dir")
    return tbl.drop_duplicates(["arm", "sync", "contrast", "concept_shift", "seed"], keep="last")


# --- Figure helpers ---


def _arm_style(arm: str) -> tuple[str, str, float, str]:
    """Return (label, color, linewidth, linestyle) for an arm key."""
    if arm in _ARM_META:
        m = _ARM_META[arm]
        return m["label"], m["color"], m["lw"], m["ls"]
    if arm.startswith(_SIMILARITY_PREFIX):
        r, _, variant = arm.removeprefix(_SIMILARITY_PREFIX).partition("_dth")
        r_disp = "∞" if r == "inf" else r
        color = _SIMILARITY_COLORS.get(r, _SIMILARITY_FALLBACK_COLOR)
        label = f"(b) MCFL R={r_disp}"
        if variant:
            d_th, _, reference = variant.partition("_")
            label += f", d_th={d_th}" + (f" ({reference})" if reference else "")
        return label, color, 1.5, "-."
    return arm, "black", 1.5, "-"


def _extra_arm_key(arm: str) -> tuple[int, float, str]:
    """Sort key for strategy keys that have no _ARM_META entry.

    Similarity keys are ordered by re-cluster period R, ascending, with R = inf
    last. Other keys follow in alphabetical order.
    """
    prefix = "SimilarityReactive_R"
    if arm.startswith(prefix):
        r = arm[len(prefix):].partition("_dth")[0]
        return (0, float("inf") if r == "inf" else float(r), arm)
    return (1, 0.0, arm)


def _arms_sorted(df: pd.DataFrame) -> list[str]:
    """Strategy keys in display order: GTRFL first, similarity by ascending R (inf last)."""
    order = list(_ARM_META.keys())
    present = list(df["arm"].unique())
    known = [a for a in order if a in present]
    extra = sorted((a for a in present if a not in order), key=_extra_arm_key)
    return known + extra


def _shared_legend(fig, ax, ncol: int) -> None:
    """Draw ax's legend centred below the whole figure.

    With large fonts a legend inside the axes covers the curves. Call after
    fig.tight_layout(); save_figure's tight bounding box then includes the legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol)


def _ba_panel_legend(ax) -> None:
    """Draw the key of a BA-vs-round panel directly beneath that panel.

    The bar panels label their strategies on the x axis, so only the line panel
    needs a key. Colour identifies the strategy; two extra black entries explain
    the line styles. Call before fig.tight_layout(), which then leaves room for
    the legend.
    """
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="black", ls="-", lw=1.8),
        Line2D([], [], color="black", ls="--", lw=1.2),
    ]
    labels += ["Global model", "Personalized model"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)


def _concept_shift_title(tbl: pd.DataFrame) -> str:
    """Figure title for a concept-shift grid, from the runs' model.use_position.

    The hidden and visible grids use the same plotting code, so the title is read
    from the data to avoid mislabelling a grid.

    Raises ValueError if the grid mixes both settings.
    """
    values = set(tbl["use_position"].dropna().astype(bool)) if "use_position" in tbl else set()
    if len(values) > 1:
        raise ValueError("Concept-shift grid mixes position-hidden and position-visible runs.")
    if values == {True}:
        return TITLE_CONCEPT_SHIFT_VISIBLE
    if values == {False}:
        return TITLE_CONCEPT_SHIFT_HIDDEN
    return TITLE_CONCEPT_SHIFT_UNKNOWN


# --- Figure: efficiency ---


def plot_efficiency(
    df: pd.DataFrame,
    tbl: pd.DataFrame,
    out_dir: Path,
    sync: int,
    target: float,
    ceiling_df: pd.DataFrame | None = None,
    round_duration_s: int = 300,
) -> None:
    """Efficiency figure at one sync_interval."""
    import matplotlib.pyplot as plt

    sub = df[df["sync_interval"] == sync]
    tsub = tbl[tbl["sync"] == sync]
    arms = _arms_sorted(sub)

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    ax_ba, ax_rounds, ax_comm, ax_ovhd = axes.flat

    # Panel 1: BA vs round (global solid, personalized dashed)
    for arm in arms:
        g = sub[sub["arm"] == arm]
        label, color, lw, _ = _arm_style(arm)
        r, m, s = mean_std_by_round(g, "ba_probe")
        ax_ba.plot(r, m, color=color, lw=lw, ls="-", marker="o", markersize=3, label=label)
        ax_ba.fill_between(r, m - s, m + s, alpha=0.12, color=color)
        rp, mp, _ = mean_std_by_round(g, "ba_personalized")
        if len(rp):
            ax_ba.plot(rp, mp, color=color, lw=1.2, ls="--", alpha=0.8)
    if ceiling_df is not None and "ba_cnn" in ceiling_df.columns:
        ax_ba.axhline(float(ceiling_df["ba_cnn"].mean()), color="black", ls="--", lw=1.0,
                      label="Centralized ceiling")
    ax_ba.axhline(0.5, color="gray", ls=":", lw=0.8, label="Chance")
    ax_ba.axhline(target, color="green", ls=":", lw=0.8, label=f"Target BA={target}")
    ax_ba.set_xlabel(f"Round  ({round_duration_s} s each)")
    ax_ba.set_ylabel("Balanced accuracy")
    ax_ba.set_title("BA vs round")
    _ba_panel_legend(ax_ba)

    # Panels 2 and 3: rounds and communication to target, bars per strategy
    # (mean and std over seeds)
    x = np.arange(len(arms))
    colors = [_arm_style(a)[1] for a in arms]
    labels = [_arm_style(a)[0] for a in arms]
    rounds_mean = [tsub.loc[tsub["arm"] == a, f"rounds_{target}"].mean() for a in arms]
    rounds_std = [tsub.loc[tsub["arm"] == a, f"rounds_{target}"].std() for a in arms]
    comm_mean = [tsub.loc[tsub["arm"] == a, "comm_gb_target"].mean() for a in arms]
    comm_std = [tsub.loc[tsub["arm"] == a, "comm_gb_target"].std() for a in arms]

    ax_rounds.bar(x, rounds_mean, yerr=np.nan_to_num(rounds_std), color=colors, alpha=0.85)
    ax_rounds.set_xticks(x)
    ax_rounds.set_xticklabels(labels, rotation=30, ha="right")
    ax_rounds.set_ylabel("Rounds")
    ax_rounds.set_title(f"Rounds to reach BA={target} (sync={sync})")

    ax_comm.bar(x, comm_mean, yerr=np.nan_to_num(comm_std), color=colors, alpha=0.85)
    ax_comm.set_xticks(x)
    ax_comm.set_xticklabels(labels, rotation=30, ha="right")
    ax_comm.set_ylabel("Cumulative comm (GB)")
    ax_comm.set_title(f"Comm to reach BA={target} (sync={sync})")

    # Panel 4: cluster-discovery overhead, 0 for ephemeris, plane and random churn,
    # T_init times the number of clusterings for similarity.
    ovhd_mean = [tsub.loc[tsub["arm"] == a, "overhead"].mean() for a in arms]
    ax_ovhd.bar(x, ovhd_mean, color=colors, alpha=0.85)
    for xi, v in zip(x, ovhd_mean):
        if not np.isnan(v):
            ax_ovhd.text(xi, v, f"{v:.0f}", ha="center", va="bottom")
    ax_ovhd.set_xticks(x)
    ax_ovhd.set_xticklabels(labels, rotation=30, ha="right")
    ax_ovhd.set_ylabel("Cluster-discovery rounds")
    ax_ovhd.set_title("Cluster-discovery overhead")
    ax_ovhd.margins(y=0.12)  # room for the value label above the tallest bar

    fig.suptitle(TITLE_EFFICIENCY, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "efficiency")
    plt.close(fig)
    print(f"Figure -> {out_dir}/efficiency.{{png,pdf}}")


# --- Figure: comparison ---


def plot_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    sync: int,
    ceiling_df: pd.DataFrame | None = None,
    round_duration_s: int = 300,
) -> None:
    """Three panels (BA vs round, cumulative communication, ARI) at one sync_interval."""
    import matplotlib.pyplot as plt

    sub = df[df["sync_interval"] == sync]
    arms = _arms_sorted(sub)
    fig, (ax_acc, ax_comm, ax_ari) = plt.subplots(1, 3, figsize=(16, 6))

    for arm in arms:
        g = sub[sub["arm"] == arm]
        label, color, lw, ls = _arm_style(arm)

        # Panel A: BA vs round
        r, m, s = mean_std_by_round(g, "ba_probe")
        ax_acc.plot(r, m, label=label, color=color, lw=lw, ls=ls, marker="o", markersize=3)
        ax_acc.fill_between(r, m - s, m + s, alpha=0.12, color=color)

        # Panel B: cumulative communication vs round (mean across seeds)
        comm_per_round = g.groupby("round").apply(
            lambda x: (x["comm_bytes_up"] + x["comm_bytes_down"]).mean()
        )
        cumul = comm_per_round.astype(float).cumsum() / 1e9
        ax_comm.plot(cumul.index.to_numpy(), cumul.to_numpy(),
                     label=label, color=color, lw=lw, ls=ls)

        # Panel C: ARI vs round (logged every round)
        if "ari_vs_regions" in g.columns:
            ari = g.groupby("round")["ari_vs_regions"]
            am, astd = ari.mean(), ari.std().fillna(0)
            ax_ari.plot(am.index.to_numpy(), am.to_numpy(),
                        label=label, color=color, lw=lw, ls=ls)
            ax_ari.fill_between(am.index.to_numpy(),
                                (am - astd).to_numpy(), (am + astd).to_numpy(),
                                alpha=0.12, color=color)

    if ceiling_df is not None and "ba_cnn" in ceiling_df.columns:
        ax_acc.axhline(float(ceiling_df["ba_cnn"].mean()), color="black", ls="--", lw=1.0,
                       label="Centralized ceiling")
    ax_acc.axhline(0.5, color="gray", ls=":", lw=0.8, label="Chance")
    ax_acc.set_xlabel(f"Round  ({round_duration_s} s each)")
    ax_acc.set_ylabel("Balanced accuracy (probe)")
    ax_acc.set_title(f"Panel A: BA vs round (sync={sync})")

    ax_comm.set_xlabel(f"Round  ({round_duration_s} s each)")
    ax_comm.set_ylabel("Cumulative comm (GB)")
    ax_comm.set_title("Panel B: Communication cost")

    ax_ari.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax_ari.set_xlabel(f"Round  ({round_duration_s} s each)")
    ax_ari.set_ylabel("ARI vs regions")
    ax_ari.set_title("Panel C: ARI vs initial regions")

    fig.suptitle("Efficiency diagnostics", fontweight="bold")
    fig.tight_layout()
    _shared_legend(fig, ax_acc, ncol=4)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "comparison")
    plt.close(fig)
    print(f"Figure -> {out_dir}/comparison.{{png,pdf}}")


# --- Figure: contrast ---


def plot_contrast(tbl: pd.DataFrame, out_dir: Path) -> None:
    """Final global and personalized BA vs predictability_contrast, one line per strategy.

    Shows whether the region strategy's personalized BA separates from plane and
    random churn as the regions' predictability is spread apart. No figure is
    written unless at least two contrast levels are present.
    """
    import matplotlib.pyplot as plt

    contrasts = sorted(c for c in tbl["contrast"].unique() if not np.isnan(c))
    if len(contrasts) < 2:
        print("plot_contrast: <2 predictability_contrast levels present, skipping.")
        return

    arms = _arms_sorted(tbl)
    fig, (ax_g, ax_p) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in [
        (ax_g, "final_ba", "Global-model BA vs contrast"),
        (ax_p, "final_pers", "Personalized (per-cluster) BA vs contrast"),
    ]:
        for arm in arms:
            stats = (
                tbl[tbl["arm"] == arm].groupby("contrast")[col]
                .agg(["mean", "std"]).reindex(contrasts)
            )
            label, color, lw, ls = _arm_style(arm)
            mean = stats["mean"].to_numpy()
            std = stats["std"].fillna(0.0).to_numpy()
            ax.plot(contrasts, mean, color=color, lw=lw, ls=ls,
                    marker="o", markersize=4, label=label)
            ax.fill_between(contrasts, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_xlabel("Predictability contrast  (0 = same for all regions)")
        ax.set_ylabel("Final balanced accuracy")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Accuracy vs predictability contrast", fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "contrast")
    plt.close(fig)
    print(f"Figure -> {out_dir}/contrast.{{png,pdf}}")


# --- Figure: concept_shift ---


def plot_concept_shift(tbl: pd.DataFrame, out_dir: Path) -> None:
    """Final global and personalized BA vs world.concept_shift, one line per strategy.

    concept_shift inverts the forecast target in half of the regions without
    changing the input window. No figure is written unless at least two shift
    levels are present. Both panels share one y scale.
    """
    import matplotlib.pyplot as plt

    if "concept_shift" not in tbl.columns:
        print("plot_concept_shift: no concept_shift column, skipping.")
        return
    shifts = sorted(c for c in tbl["concept_shift"].unique() if not np.isnan(c))
    if len(shifts) < 2:
        print("plot_concept_shift: <2 concept_shift levels present, skipping.")
        return

    arms = _arms_sorted(tbl)
    fig, (ax_g, ax_p) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col, title in [
        (ax_g, "final_ba", "Global-model BA vs concept shift"),
        (ax_p, "final_pers", "Personalized BA vs concept shift"),
    ]:
        for arm in arms:
            stats = (
                tbl[tbl["arm"] == arm].groupby("concept_shift")[col]
                .agg(["mean", "std"]).reindex(shifts)
            )
            label, color, lw, ls = _arm_style(arm)
            mean = stats["mean"].to_numpy()
            std = stats["std"].fillna(0.0).to_numpy()
            ax.plot(shifts, mean, color=color, lw=lw, ls=ls,
                    marker="o", markersize=4, label=label)
            ax.fill_between(shifts, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_xlabel("Concept shift  (0 = no conflict)")
        ax.set_ylabel("Final balanced accuracy")
        ax.tick_params(axis="y", labelleft=True)  # sharey would otherwise hide the right panel's
        ax.set_title(title)

    fig.suptitle(_concept_shift_title(tbl), fontweight="bold")
    fig.tight_layout()
    _shared_legend(fig, ax_p, ncol=len(arms))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "concept_shift")
    plt.close(fig)
    print(f"Figure -> {out_dir}/concept_shift.{{png,pdf}}")


# --- Summary table ---


def _print_verdict(tbl: pd.DataFrame, targets: tuple[float, ...]) -> None:
    """Print efficiency and accuracy per strategy and sync interval, averaged over seeds."""
    print("\n-- Summary (mean over seeds, per strategy and sync interval) --")
    round_cols = [f"rounds_{t}" for t in targets]
    hdr_r = "  ".join(f"r>={t}" for t in targets)
    print(f"{'arm':<26}{'sync':>5}{'n':>4}{'BA':>8}{'sdBA':>7}{'pers':>8}{'sdprs':>7}  {hdr_r}"
          f"{'commGB@'+str(targets[-1]):>12}{'ovhd':>6}{'clus':>6}{'ARI':>7}")
    print("-" * 124)
    degenerate: list[str] = []
    g = tbl.groupby(["arm", "sync"])
    for (arm, sync), grp in sorted(g, key=lambda kv: (_arm_order(kv[0][0]), kv[0][1])):
        n = grp["seed"].nunique()
        r_strs = "  ".join(
            (f"{grp[c].mean():5.0f}" if grp[c].notna().any() else "   --")
            for c in round_cols
        )
        commt = grp["comm_gb_target"].mean()
        commt_s = f"{commt:12.2f}" if not np.isnan(commt) else f"{'--':>12}"
        # Sample std across seeds; NaN (printed as "--") when there is a single seed.
        sd_ba, sd_pers = grp["final_ba"].std(), grp["final_pers"].std()
        sd_ba_s = f"{sd_ba:7.4f}" if not np.isnan(sd_ba) else f"{'--':>7}"
        sd_pers_s = f"{sd_pers:7.4f}" if not np.isnan(sd_pers) else f"{'--':>7}"
        clus = grp["mean_clusters"].mean()
        clus_s = f"{clus:6.1f}" if not np.isnan(clus) else f"{'--':>6}"
        if not np.isnan(clus) and float(grp["max_clusters"].max()) <= 1.0:
            degenerate.append(arm)
        print(
            f"{arm:<26}{int(sync):>5}{n:>4}"
            f"{grp['final_ba'].mean():>8.4f}{sd_ba_s}"
            f"{grp['final_pers'].mean():>8.4f}{sd_pers_s}  {r_strs}"
            f"{commt_s}{int(grp['overhead'].mean()):>6}{clus_s}{grp['final_ari'].mean():>7.3f}"
        )
    if (tbl["overhead"] == 0).all():
        print("\nNote: overhead_rounds = 0 for all strategies present (no cluster discovery).")
        print("      Comparing overhead requires runs of the similarity strategy.")
    if degenerate:
        print(
            "\nSingle-cluster partition: "
            + ", ".join(sorted(set(degenerate)))
            + "\n    never had more than one cluster in any round, so hierarchical"
            "\n    aggregation was identical to flat FedAvg and the partition had no"
            "\n    effect on training. Its accuracy equals that of unclustered FedAvg."
            "\n    The discovery rounds counted in overhead_rounds were still spent."
        )
    if "mean_clusters" in tbl.columns and tbl["mean_clusters"].isna().all():
        print("\nNote: these runs have no n_clusters column, so the number of clusters")
        print("      each strategy formed cannot be checked from this directory.")
    print()


def _arm_order(arm: str) -> int:
    order = list(_ARM_META.keys())
    return order.index(arm) if arm in order else len(order)


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clustering strategies and draw figures.")
    parser.add_argument("paths", type=Path, nargs="+",
                        help="Run dir(s), or one dir whose subdirs are run dirs.")
    parser.add_argument("--ceiling", type=Path, default=None,
                        help="metrics.parquet of a centralized baseline run, drawn as a ceiling.")
    parser.add_argument("--out", type=Path, default=Path("figures/regenerated"))
    parser.add_argument("--sync", type=int, default=None,
                        help="Representative sync_interval for vs-round panels/bars "
                             "(default: max present).")
    parser.add_argument("--target", type=float, default=0.58,
                        help="Target BA for the rounds/comm-to-target bars (default 0.58).")
    parser.add_argument("--round-duration-s", type=int, default=300)
    args = parser.parse_args()

    if len(args.paths) == 1 and args.paths[0].is_dir() \
            and not (args.paths[0] / "metrics.parquet").exists():
        run_dirs = _discover_run_dirs(args.paths[0])
        if not run_dirs:
            print(f"No run directories found under {args.paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dirs = args.paths

    print(f"Loading {len(run_dirs)} run directories...")
    df = _load_all(run_dirs)
    print(f"  Loaded {len(df):,} rows, arms: {sorted(df['arm'].unique())}")

    sync = args.sync if args.sync is not None else int(df["sync_interval"].max())
    print(f"  Representative sync_interval = {sync}")

    ceiling_df = None
    if args.ceiling is not None:
        ceiling_df = pd.read_parquet(args.ceiling)
        print(f"  Ceiling: {args.ceiling} ({len(ceiling_df)} rows)")

    targets = tuple(sorted(set(DEFAULT_TARGETS) | {args.target}))
    tbl = _per_run_table(df, targets)

    use_orbitfl_style()
    plot_efficiency(df, tbl, args.out, sync=sync, target=args.target,
                    ceiling_df=ceiling_df, round_duration_s=args.round_duration_s)
    plot_comparison(df, args.out, sync=sync, ceiling_df=ceiling_df,
                    round_duration_s=args.round_duration_s)
    plot_contrast(tbl, args.out)  # only with >= 2 predictability_contrast levels
    plot_concept_shift(tbl, args.out)  # only with >= 2 concept_shift levels
    _print_verdict(tbl, targets)


if __name__ == "__main__":
    main()
