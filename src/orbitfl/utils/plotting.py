"""Shared matplotlib style and figure saving.

use_orbitfl_style() applies the common rcParams; call it once at the top of a
plotting script. save_figure writes a figure as both PNG and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font sizes in points at the figure's native size. Multi-panel figures are 12-16 in
# wide and are scaled down to a page width of about 6.5 in, so the sizes are large.
# Plotting code should rely on these rather than pass fontsize explicitly.
FONT_SIZE_BASE: int = 15  # tick labels, in-axes annotations
FONT_SIZE_LEGEND: int = 14
FONT_SIZE_AXIS_LABEL: int = 17
FONT_SIZE_PANEL_TITLE: int = 17
FONT_SIZE_FIGURE_TITLE: int = 21

# Plain style with no extra dependencies.
_ORBITFL_RC: dict = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": FONT_SIZE_BASE,
    "axes.labelsize": FONT_SIZE_AXIS_LABEL,
    "axes.titlesize": FONT_SIZE_PANEL_TITLE,
    "figure.titlesize": FONT_SIZE_FIGURE_TITLE,
    "xtick.labelsize": FONT_SIZE_BASE,
    "ytick.labelsize": FONT_SIZE_BASE,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
}


def use_orbitfl_style() -> None:
    """Apply the shared orbitfl matplotlib rcParams. Call once per script."""
    plt.rcParams.update(_ORBITFL_RC)


def mean_std_by_round(
    df: pd.DataFrame, col: str, round_col: str = "round"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard deviation of a metric per round across seeds.

    Probe metrics (ba_probe, auc_probe, ba_personalized, ...) are logged only every
    fl.eval_interval rounds and are NaN in between. Rounds where the metric is NaN
    for every seed are dropped, so the remaining points are contiguous and
    matplotlib connects them with a line. Use this helper for every
    metric-versus-round plot.

    Parameters
    ----------
    df        : pd.DataFrame  Long-form metrics with a round column and one row per
                              (round, seed).
    col       : str           Metric column to reduce (may contain NaN).
    round_col : str           Round column name (default "round").

    Returns
    -------
    (rounds, mean, std) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays aligned on the evaluated rounds only; a NaN std (single seed) becomes 0.
    """
    stats = df.groupby(round_col)[col].agg(["mean", "std"]).dropna(subset=["mean"])
    return (
        stats.index.to_numpy(),
        stats["mean"].to_numpy(),
        stats["std"].fillna(0.0).to_numpy(),
    )


def save_figure(fig: plt.Figure, run_dir: Path, name: str) -> None:
    """Write a figure to disk as PNG (150 dpi) and PDF.

    Parameters
    ----------
    fig     : matplotlib.figure.Figure
    run_dir : Path   Output directory.
    name    : str    Base filename without extension (e.g. "efficiency").
    """
    run_dir = Path(run_dir)
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
