"""Communication-efficiency metrics: rounds and bytes needed to reach a target accuracy.

When all clustering strategies converge to the same accuracy, they are compared
by how cheaply they get there. These helpers work on one run's per-round metrics
DataFrame (a single seed).
"""

from __future__ import annotations

import pandas as pd

# Balanced-accuracy thresholds reported by default, shared by all strategies.
DEFAULT_TARGETS: tuple[float, ...] = (0.55, 0.575, 0.58)


def rounds_to_target(
    df: pd.DataFrame,
    target: float,
    col: str = "ba_probe",
    round_col: str = "round",
) -> float:
    """
    First round at which `col` reaches `target`.

    Only evaluated rounds count: `col` is NaN on rounds skipped by eval_interval, so
    those are dropped before scanning. The first round (ascending) whose value is
    >= `target` is returned, at eval-round granularity.

    Parameters
    ----------
    df        : pd.DataFrame  One run's per-round metrics (single seed).
    target    : float         Target value of `col` (e.g. 0.58 balanced accuracy).
    col       : str           Metric column to threshold (default "ba_probe").
    round_col : str           Round-index column (default "round").

    Returns
    -------
    float  Round index of the first crossing, or NaN if `target` is never reached.
    """
    ev = df[[round_col, col]].dropna(subset=[col]).sort_values(round_col)
    hit = ev[ev[col] >= target]
    return float(hit[round_col].iloc[0]) if len(hit) else float("nan")


def comm_to_target(
    df: pd.DataFrame,
    target: float,
    col: str = "ba_probe",
    round_col: str = "round",
) -> float:
    """
    Cumulative communication (GB) through the first round that reaches `target`.

    Communication is logged every round (comm_bytes_up + comm_bytes_down), so this sums
    all rounds from the start up to and including `rounds_to_target(...)`.

    Parameters
    ----------
    df        : pd.DataFrame  One run's per-round metrics (single seed).
    target    : float         Target value of `col`.
    col       : str           Metric column to threshold (default "ba_probe").
    round_col : str           Round-index column (default "round").

    Returns
    -------
    float  Cumulative up+down communication in GB, or NaN if `target` is never reached.
    """
    r = rounds_to_target(df, target, col=col, round_col=round_col)
    if pd.isna(r):
        return float("nan")
    upto = df[df[round_col] <= r]
    total_bytes = (upto["comm_bytes_up"] + upto["comm_bytes_down"]).sum()
    return float(total_bytes) / 1e9
