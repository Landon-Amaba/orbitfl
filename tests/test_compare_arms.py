"""Tests for scripts/compare_arms.py.

Run loading: a run that stopped before its round_budget, or whose budget is
smaller than that of the other runs in the folder, must not enter the averages
or replace a complete run with the same (strategy, sync, shift, seed).

Display: strategy labels, ordering and colours, figure titles and legends.

Cluster counts: the per-run table carries the number of clusters formed, and
the summary warns about a strategy that never formed more than one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import compare_arms  # noqa: E402


def _make_run(
    run_dir: Path,
    *,
    round_budget: int,
    max_round: int,
    ba: float,
    seed: int = 0,
    concept_shift: float = 0.0,
) -> Path:
    """Write a minimal run dir (metrics.parquet + config.yaml) for compare_arms."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(max_round + 1):
        evaluated = r == max_round  # only the last row carries BA (mirrors eval_interval)
        rows.append(
            {
                "round": r,
                "ba_probe": ba if evaluated else float("nan"),
                "ba_personalized": ba if evaluated else float("nan"),
                "auc_probe": ba if evaluated else float("nan"),
                "comm_bytes_up": 1000,
                "comm_bytes_down": 1000,
                "overhead_rounds": 0,
                "ari_vs_regions": 1.0,
                "seed": seed,
                "strategy": "EphemerisClustering",
            }
        )
    pd.DataFrame(rows).to_parquet(run_dir / "metrics.parquet")
    cfg = {
        "fl": {
            "round_budget": round_budget,
            "sync_interval": round_budget,
            "strategy": {"name": "ephemeris", "params": {"use_meo_proximity": False}},
        },
        "world": {"concept_shift": concept_shift},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    return run_dir


def test_incomplete_run_is_skipped(tmp_path):
    """A run that never reached round_budget-1 returns None from _load_run_dir."""
    d = _make_run(tmp_path / "short", round_budget=300, max_round=19, ba=0.55)
    assert compare_arms._load_run_dir(d) is None


def test_complete_run_is_kept(tmp_path):
    """A run that reached round_budget-1 loads normally."""
    d = _make_run(tmp_path / "full", round_budget=300, max_round=299, ba=0.83)
    out = compare_arms._load_run_dir(d)
    assert out is not None
    assert float(out["ba_personalized"].dropna().iloc[-1]) == 0.83


def test_missing_round_budget_run_is_kept(tmp_path):
    """Older runs without fl.round_budget cannot be assessed and are kept."""
    d = _make_run(tmp_path / "nobudget", round_budget=300, max_round=50, ba=0.6)
    # strip round_budget from the config to simulate an older run
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    del cfg["fl"]["round_budget"]
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert compare_arms._load_run_dir(d) is not None


def test_stray_run_cannot_overwrite_grid_point(tmp_path):
    """A short test run with the same (arm, sync, concept_shift, seed) as a complete
    run must not replace it in _per_run_table; it is dropped when the runs are loaded.
    """
    root = tmp_path / "results"
    # Complete grid run and a colliding stray, same arm/sync/shift/seed=0.
    _make_run(root / "20260930T150000Z_nogit_jGRID-0",
              round_budget=300, max_round=299, ba=0.83)
    _make_run(root / "20260930T142943Z_nogit_stray",
              round_budget=20, max_round=19, ba=0.55)  # later-sorting name -> keep="last" trap

    paths = compare_arms._discover_run_dirs(root)
    df = compare_arms._load_all(paths)
    tbl = compare_arms._per_run_table(df, targets=(0.55, 0.575, 0.58))

    # Exactly one surviving row for the (arm, shift=0, seed=0) cell, and it is the
    # complete grid run's value - not the 0.55 stray.
    cell = tbl[(tbl["concept_shift"] == 0.0) & (tbl["seed"] == 0)]
    assert len(cell) == 1
    assert float(cell["final_pers"].iloc[0]) == 0.83


def test_proposed_arm_is_labelled_gtrfl():
    """No figure label may still use the method's old name."""
    assert "GTRFL" in compare_arms._arm_style("EphemerisRegion")[0]
    labels = [compare_arms._arm_style(a)[0] for a in compare_arms._ARM_META]
    assert not any("ephemeris" in label.lower() for label in labels)


def test_similarity_arms_are_labelled_mcfl():
    """Similarity arms carry the thesis's name for the method, with their R."""
    assert compare_arms._arm_style("SimilarityReactive_R5")[0] == "(b) MCFL R=5"
    assert compare_arms._arm_style("SimilarityReactive_Rinf")[0] == "(b) MCFL R=∞"
    assert "Similarity" not in compare_arms._arm_style("SimilarityReactive")[0]


def test_arms_sorted_orders_similarity_numerically_with_distinct_colors():
    """R runs 5, 10, 20, inf (not string order), and each R gets its own colour."""
    arms = ["SimilarityReactive_R10", "SimilarityReactive_Rinf", "SimilarityReactive_R5",
            "SimilarityReactive_R20", "PlaneClustering", "EphemerisRegion"]
    ordered = compare_arms._arms_sorted(pd.DataFrame({"arm": arms}))
    assert ordered == ["EphemerisRegion", "PlaneClustering", "SimilarityReactive_R5",
                       "SimilarityReactive_R10", "SimilarityReactive_R20",
                       "SimilarityReactive_Rinf"]
    colors = [compare_arms._arm_style(a)[1] for a in ordered[2:]]
    assert len(set(colors)) == len(colors)


def test_concept_shift_title_follows_use_position():
    """The title comes from each grid's recorded use_position, and mixed grids are refused."""
    title = compare_arms._concept_shift_title
    assert title(pd.DataFrame({"use_position": [False, False]})) == "Concept shift, position hidden"
    assert title(pd.DataFrame({"use_position": [True, True]})) == "Concept shift, position visible"
    assert title(pd.DataFrame({"use_position": [None, None]})) == "Concept shift"
    with pytest.raises(ValueError):
        title(pd.DataFrame({"use_position": [True, False]}))


def test_efficiency_line_panel_has_its_own_key(tmp_path, monkeypatch):
    """The BA-vs-round panel carries the key: every arm, both line styles, and both guides."""
    d = _make_run(tmp_path / "r", round_budget=20, max_round=19, ba=0.6)
    df = compare_arms._load_all([d])
    tbl = compare_arms._per_run_table(df, (0.58,))

    captured = {}

    def capture(fig, _out_dir, _name):
        """Inspect the finished figure before plot_efficiency closes it."""
        captured["legends"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        captured["n_figure_legends"] = len(fig.legends)

    monkeypatch.setattr(compare_arms, "save_figure", capture)
    compare_arms.plot_efficiency(df, tbl, tmp_path / "out", sync=20, target=0.58)

    arm_labels = [compare_arms._arm_style(a)[0] for a in compare_arms._arms_sorted(df)]
    assert captured["legends"] == [
        *arm_labels, "Chance", "Target BA=0.58", "Global model", "Personalized model",
    ]
    assert captured["n_figure_legends"] == 0


def test_concept_shift_panels_share_a_labelled_y_scale(tmp_path, monkeypatch):
    """Both panels use one y-scale, and each keeps its own tick labels."""
    tbl = pd.DataFrame({
        "arm": ["EphemerisRegion", "EphemerisRegion"],
        "concept_shift": [0.0, 0.9],
        "final_ba": [0.5, 0.6],
        "final_pers": [0.8, 0.9],
        "use_position": [True, True],
    })
    captured = {}

    def capture(fig, _out_dir, _name):
        """Inspect the finished figure before plot_concept_shift closes it."""
        fig.canvas.draw()
        captured["ylims"] = [ax.get_ylim() for ax in fig.axes]
        captured["labelled"] = [
            any(t.get_visible() and t.get_text() for t in ax.get_yticklabels()) for ax in fig.axes
        ]

    monkeypatch.setattr(compare_arms, "save_figure", capture)
    compare_arms.plot_concept_shift(tbl, tmp_path / "out")

    assert captured["ylims"][0] == captured["ylims"][1]
    assert captured["labelled"] == [True, True]


def test_use_position_is_read_from_config(tmp_path):
    """_load_run_dir stamps model.use_position, and _per_run_table carries it through."""
    d = _make_run(tmp_path / "r", round_budget=20, max_round=19, ba=0.6)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    cfg["model"] = {"use_position": False}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    tbl = compare_arms._per_run_table(compare_arms._load_all([d]), (0.58,))
    assert compare_arms._concept_shift_title(tbl) == "Concept shift, position hidden"


# --- Display order / colour of the similarity-R arms ---


def test_similarity_arms_ordered_by_recluster_period():
    """
    Similarity arms sort as the R ladder 5/10/20/inf, not alphabetically.

    Plain `sorted()` yields R10, R20, R5, Rinf, which makes the overhead bars in
    e1_efficiency read as a scramble instead of a monotone cost ladder.
    """
    present = [
        "SimilarityReactive_R20", "SimilarityReactive_R5",
        "SimilarityReactive_Rinf", "SimilarityReactive_R10",
    ]
    arms = compare_arms._arms_sorted(pd.DataFrame({"arm": present + ["EphemerisRegion"]}))
    assert arms == [
        "EphemerisRegion",
        "SimilarityReactive_R5", "SimilarityReactive_R10",
        "SimilarityReactive_R20", "SimilarityReactive_Rinf",
    ]


def test_unknown_arms_follow_similarity_arms_alphabetically():
    """Arms with no _ARM_META entry and no R suffix keep alphabetical order, last."""
    present = ["ZzzOther", "AaaOther", "SimilarityReactive_R10"]
    arms = compare_arms._arms_sorted(pd.DataFrame({"arm": present}))
    assert arms == ["SimilarityReactive_R10", "AaaOther", "ZzzOther"]


def test_similarity_arms_get_distinct_colours():
    """
    Each R in the ladder gets its own colour.

    Indexing the palette by R itself collided: 20 % 4 == 0, the R=inf slot, so the
    R=20 and R=inf bars were drawn identically.
    """
    colors = [
        compare_arms._arm_style(f"SimilarityReactive_R{r}")[1]
        for r in ("5", "10", "20", "inf")
    ]
    assert len(set(colors)) == len(colors)


def test_similarity_variants_get_distinct_arm_keys(tmp_path: Path) -> None:
    """d_th-sweep runs stay separate arms; default runs keep their existing key."""
    keys = {}
    for name, params in {
        "default": {"recluster_every_R": 20, "d_th": 1.0},
        "dth": {"recluster_every_R": 20, "d_th": 0.5, "update_reference": "round_start"},
        "ref": {"recluster_every_R": None, "d_th": 1.0, "update_reference": "round_start"},
    }.items():
        run = _make_run(tmp_path / name, round_budget=2, max_round=1, ba=0.6)
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        cfg["fl"]["strategy"] = {"name": "similarity", "params": params}
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        df = pd.read_parquet(run / "metrics.parquet")
        df["strategy"] = "SimilarityReactive"
        df.to_parquet(run / "metrics.parquet")
        keys[name] = compare_arms._load_run_dir(run)["arm"].iloc[0]

    assert keys["default"] == "SimilarityReactive_R20"
    assert keys["dth"] == "SimilarityReactive_R20_dth0.5_round_start"
    assert keys["ref"] == "SimilarityReactive_Rinf_dth1_round_start"
    assert compare_arms._extra_arm_key(keys["dth"])[1] == 20.0
    assert "d_th=0.5" in compare_arms._arm_style(keys["dth"])[0]


# --- Number of clusters formed ---
# A similarity run can end up with a single cluster holding every satellite, in
# which case hierarchical aggregation is flat FedAvg and the re-cluster period
# has no effect. The n_clusters column of the metrics records the partition
# size; these tests check that it reaches the per-run table and the warning.


def _make_run_with_clusters(
    run_dir: Path,
    *,
    n_clusters_per_round: list[int],
    ba: float = 0.6,
    seed: int = 0,
    strategy: str = "EphemerisClustering",
    include_column: bool = True,
) -> Path:
    """Write a complete run dir whose n_clusters column follows the given per-round list."""
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = len(n_clusters_per_round)
    rows = []
    for r, k in enumerate(n_clusters_per_round):
        evaluated = r == budget - 1
        row = {
            "round": r,
            "ba_probe": ba if evaluated else float("nan"),
            "ba_personalized": ba if evaluated else float("nan"),
            "auc_probe": ba if evaluated else float("nan"),
            "comm_bytes_up": 1000,
            "comm_bytes_down": 1000,
            "overhead_rounds": 0,
            "ari_vs_regions": 0.0,
            "seed": seed,
            "strategy": strategy,
        }
        if include_column:
            row["n_clusters"] = k
        rows.append(row)
    pd.DataFrame(rows).to_parquet(run_dir / "metrics.parquet")
    cfg = {
        "fl": {
            "round_budget": budget,
            "sync_interval": budget,
            "strategy": {"name": "ephemeris", "params": {"use_meo_proximity": False}},
        },
        "world": {},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    return run_dir


def _table(paths: list[Path]):
    return compare_arms._per_run_table(compare_arms._load_all(paths), (0.58,))


def test_mean_and_max_clusters_recorded(tmp_path):
    """mean_clusters averages over rounds; max_clusters takes the per-run maximum."""
    run = _make_run_with_clusters(tmp_path / "r0", n_clusters_per_round=[1, 4, 4, 1])
    tbl = _table([run])
    assert tbl["mean_clusters"].iloc[0] == 2.5
    assert tbl["max_clusters"].iloc[0] == 4.0


def test_clusters_are_nan_when_column_absent(tmp_path):
    """Results dirs predating the n_clusters column must still load, reporting NaN."""
    run = _make_run_with_clusters(
        tmp_path / "r0", n_clusters_per_round=[4] * 4, include_column=False
    )
    tbl = _table([run])
    assert "mean_clusters" in tbl.columns
    assert pd.isna(tbl["mean_clusters"].iloc[0])
    assert pd.isna(tbl["max_clusters"].iloc[0])


def test_verdict_flags_single_cluster_arm(tmp_path, capsys):
    """An arm that never exceeds one cluster triggers the degenerate partition warning."""
    run = _make_run_with_clusters(tmp_path / "r0", n_clusters_per_round=[1] * 4)
    compare_arms._print_verdict(_table([run]), (0.58,))
    out = capsys.readouterr().out
    assert "Single-cluster partition" in out
    assert "identical to flat FedAvg" in out


def test_verdict_silent_when_partition_is_real(tmp_path, capsys):
    """An arm reaching >1 cluster at any round must not be flagged."""
    run = _make_run_with_clusters(tmp_path / "r0", n_clusters_per_round=[1, 4, 4, 4])
    compare_arms._print_verdict(_table([run]), (0.58,))
    assert "Single-cluster partition" not in capsys.readouterr().out


def test_verdict_reports_seed_std(tmp_path, capsys):
    """Across-seed std is printed; a single-seed cell prints '--' rather than NaN."""
    multi = [
        _make_run_with_clusters(tmp_path / f"m{s}", n_clusters_per_round=[4] * 4,
                                ba=0.60 + 0.01 * s, seed=s)
        for s in range(3)
    ]
    compare_arms._print_verdict(_table(multi), (0.58,))
    out = capsys.readouterr().out
    assert "sdBA" in out and "0.0100" in out  # std of 0.60/0.61/0.62

    one = _make_run_with_clusters(tmp_path / "s0", n_clusters_per_round=[4] * 4, seed=0)
    compare_arms._print_verdict(_table([one]), (0.58,))
    assert "--" in capsys.readouterr().out
