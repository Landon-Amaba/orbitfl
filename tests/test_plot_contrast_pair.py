"""Tests for scripts/plot_contrast_pair.py::_agg, the aggregation per (strategy,
contrast) behind the predictability-contrast figure. Checks that it loads runs
through compare_arms (complete runs only) and averages personalized and global
BA per cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import plot_contrast_pair as pcp  # noqa: E402

_STRAT_CLASS = {
    "ephemeris": "EphemerisClustering",
    "plane": "PlaneClustering",
    "random_churn": "RandomChurnClustering",
}


def _make_run(
    run_dir: Path, *, arm: str, contrast: float, seed: int, ba: float,
    round_budget: int = 300, max_round: int | None = None,
) -> None:
    """Write a minimal run directory for one task of a predictability-contrast sweep."""
    if max_round is None:
        max_round = round_budget - 1
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(max_round + 1):
        ev = r == max_round
        rows.append({
            "round": r,
            "ba_probe": ba - 0.01 if ev else float("nan"),
            "ba_personalized": ba if ev else float("nan"),
            "comm_bytes_up": 1000, "comm_bytes_down": 1000,
            "overhead_rounds": 0, "ari_vs_regions": 1.0,
            "seed": seed, "strategy": _STRAT_CLASS[arm],
        })
    pd.DataFrame(rows).to_parquet(run_dir / "metrics.parquet")
    cfg = {
        "fl": {"round_budget": round_budget, "sync_interval": 20,
               "strategy": {"name": arm, "params": {"use_meo_proximity": False}}},
        "world": {"predictability_contrast": contrast},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))


def test_agg_means_per_arm_contrast(tmp_path):
    """_agg returns mean personalized and global BA per (strategy, contrast)."""
    root = tmp_path / "res"
    # eph @ contrast 0.0, two seeds with BA 0.60 and 0.62 -> mean 0.61.
    _make_run(root / "r0", arm="ephemeris", contrast=0.0, seed=0, ba=0.60)
    _make_run(root / "r1", arm="ephemeris", contrast=0.0, seed=1, ba=0.62)
    # plane @ contrast 0.0, one seed.
    _make_run(root / "r2", arm="plane", contrast=0.0, seed=0, ba=0.59)
    # A run with a smaller round budget, which the loader must drop.
    _make_run(root / "rx", arm="ephemeris", contrast=0.0, seed=0, ba=0.50,
              round_budget=20)

    g = pcp._agg(root)
    eph = g[(g["arm"] == "EphemerisRegion") & (g["contrast"] == 0.0)].iloc[0]
    assert eph["n"] == 2  # the budget-20 stray was excluded, not counted
    assert abs(eph["pers_m"] - 0.61) < 1e-9
    assert abs(eph["glob_m"] - 0.60) < 1e-9  # ba_probe = ba - 0.01, mean of 0.59/0.61
    plane = g[(g["arm"] == "PlaneClustering") & (g["contrast"] == 0.0)].iloc[0]
    assert plane["n"] == 1
    assert abs(plane["pers_m"] - 0.59) < 1e-9
