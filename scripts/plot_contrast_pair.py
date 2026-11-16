#!/usr/bin/env python
r"""Predictability-contrast figure: position hidden next to position visible.

Two sweeps vary world.predictability_contrast, one with model.use_position
false (left panel) and one with it true (right panel). Each panel shows the
final personalized balanced accuracy of the region, plane and random-churn
strategies against the contrast.

Runs are loaded with compare_arms._load_all and _per_run_table, so incomplete
runs and runs with a smaller round budget are skipped and duplicate
(strategy, sync, contrast, seed) runs are counted once.

Usage:
  python scripts/plot_contrast_pair.py \
      --hidden results/contrast_position_hidden \
      --visible results/contrast_position_visible \
      --out figures/regenerated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_arms  # noqa: E402

from orbitfl.utils.plotting import use_orbitfl_style  # noqa: E402

# Strategy keys as written by compare_arms._load_run_dir, with label and colour.
ARMS = [
    ("EphemerisRegion", "GTRFL", "#1f77b4"),
    ("PlaneClustering", "plane", "#ff7f0e"),
    ("RandomChurnClustering", "random churn", "#2ca02c"),
]
TARGETS = (0.55, 0.575, 0.58)  # unused here, required by _per_run_table
# Figure title
TITLE_CONTRAST = "Predictability contrast"


def _agg(result_dir: Path) -> pd.DataFrame:
    """Mean and std of final personalized and global BA per (strategy, contrast)."""
    paths = compare_arms._discover_run_dirs(result_dir)
    tbl = compare_arms._per_run_table(compare_arms._load_all(paths), TARGETS)
    g = (
        tbl.groupby(["arm", "contrast"])
        .agg(
            pers_m=("final_pers", "mean"),
            pers_s=("final_pers", "std"),
            glob_m=("final_ba", "mean"),
            n=("final_pers", "size"),
        )
        .reset_index()
    )
    return g


def _plot_panel(ax, g: pd.DataFrame, title: str) -> None:
    for key, label, color in ARMS:
        s = g[g["arm"] == key].sort_values("contrast")
        if not len(s):
            continue
        ax.errorbar(
            s["contrast"], s["pers_m"], yerr=s["pers_s"].fillna(0.0),
            marker="o", capsize=3, lw=2, color=color, label=label,
        )
    # Global model of the region strategy, as a faint dashed reference line.
    eph = g[g["arm"] == "EphemerisRegion"].sort_values("contrast")
    if len(eph):
        ax.plot(eph["contrast"], eph["glob_m"], "k--", alpha=0.5, lw=1.2,
                label="global model (ba_probe)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Predictability contrast  (0 = uniform)")
    ax.set_xticks([0.0, 0.3, 0.6, 0.9])
    ax.grid(alpha=0.3)


def _verdict(g_hidden: pd.DataFrame, g_visible: pd.DataFrame) -> None:
    """Print personalized BA per contrast and the gap to the best baseline, per panel."""
    for label, g in [("Position hidden", g_hidden),
                     ("Position visible", g_visible)]:
        print(f"\n{label}: personalized BA (mean ± std over seeds), "
              "gap = eph − max(plane,random):")
        piv_m = g.pivot(index="contrast", columns="arm", values="pers_m")
        piv_s = g.pivot(index="contrast", columns="arm", values="pers_s")
        npiv = g.pivot(index="contrast", columns="arm", values="n")
        for c in sorted(piv_m.index):
            eph = piv_m.loc[c, "EphemerisRegion"]
            bases = [a for a in ["PlaneClustering", "RandomChurnClustering"] if a in piv_m.columns]
            best = max(piv_m.loc[c, a] for a in bases)
            gap = eph - best
            pooled = np.sqrt(np.nanmean([piv_s.loc[c, a] ** 2
                                         for a in (["EphemerisRegion"] + bases)]))
            sig = gap / pooled if pooled and pooled > 0 else float("nan")
            seeds = "/".join(str(int(npiv.loc[c, a])) for a in (["EphemerisRegion"] + bases))
            print(f"  contrast {c}: eph={eph:.3f}  plane={piv_m.loc[c,'PlaneClustering']:.3f}  "
                  f"random={piv_m.loc[c,'RandomChurnClustering']:.3f}  "
                  f"gap={gap:+.3f} ({sig:+.1f}σ)  [n={seeds}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=Path, default=Path("results/contrast_position_hidden"))
    ap.add_argument("--visible", type=Path, default=Path("results/contrast_position_visible"))
    ap.add_argument("--out", type=Path, default=Path("figures/regenerated"))
    args = ap.parse_args()

    g_hidden = _agg(args.hidden)
    g_visible = _agg(args.visible)

    use_orbitfl_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _plot_panel(axes[0], g_hidden, "Position hidden")
    _plot_panel(axes[1], g_visible, "Position visible")
    axes[0].set_ylabel("Final personalized BA")
    axes[0].legend(loc="best")
    fig.suptitle(TITLE_CONTRAST, fontweight="bold")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"contrast_pair.{ext}", dpi=150, bbox_inches="tight")
    print(f"Figure -> {args.out}/contrast_pair.{{png,pdf}}")
    _verdict(g_hidden, g_visible)


if __name__ == "__main__":
    main()
