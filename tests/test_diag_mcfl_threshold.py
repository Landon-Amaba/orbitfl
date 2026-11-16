"""Tests for scripts/diag_mcfl_threshold.py.

The main test checks that a run with the observers attached has exactly the
same metrics as a run with the plain runner; the diagnostics are only
meaningful if observing does not change the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from orbitfl.config import Config, load_config
from orbitfl.fl.clustering.similarity import SimilarityReactive
from orbitfl.fl.loop import FederatedRunner
from orbitfl.metrics.forecast import auc, balanced_accuracy
from orbitfl.trace.reader import Trace
from orbitfl.utils.seeding import seed_everything
from orbitfl.world.occupancy import OccupancyWorld

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import diag_mcfl_threshold as diag  # noqa: E402

_GOLDEN = Path(__file__).parent / "golden" / "tiny_trace"
_BASE_CFG = Path(__file__).parent.parent / "configs" / "base.yaml"
_SETTINGS = {
    "ari_null_permutations": 20,
    "ari_null_seed": 0,
    "min_real_cluster_size": 2,
}
_SEED = 0
_T_INIT = 2
_D_TH = 0.5


def _config(round_budget: int = 6, recluster_every_r: int | None = 3) -> Config:
    """
    Small similarity Config over the golden tiny trace.

    Parameters
    ----------
    round_budget      : int        Rounds to run.
    recluster_every_r : int | None Re-clustering period.

    Returns
    -------
    Config  Fully resolved config.
    """
    return load_config(
        _BASE_CFG,
        override_dict={
            "fl": {
                "round_budget": round_budget,
                "round_duration_s": 300,
                "sync_interval": 2,
                "eval_interval": 2,
                "strategy": {
                    "name": "similarity",
                    "params": {
                        "T_init": _T_INIT,
                        "gamma": 0.5,
                        "d_th": _D_TH,
                        "recluster_every_R": recluster_every_r,
                        "update_reference": "round_start",
                    },
                },
            },
            "model": {"hidden": 16, "window": 16},
            "experiment": {"delta_steps": 4, "seeds": [_SEED]},
        },
    )


def _plain_metrics(cfg: Config, trace: Trace) -> pd.DataFrame:
    """Run the plain loop, seeding in the order experiments.runner.run_one uses.

    Parameters
    ----------
    cfg   : Config  Resolved config.
    trace : Trace   Loaded trace.

    Returns
    -------
    pd.DataFrame  Per-round metrics.
    """
    seed_everything(_SEED)
    world = OccupancyWorld(
        cfg.world, seed=_SEED, timestep_s=cfg.sim.timestep_s, n_steps=trace.n_timesteps
    )
    strategy = SimilarityReactive(
        n_leo=trace.n_leo,
        n_clusters=trace.n_meo,
        T_init=_T_INIT,
        recluster_every_R=cfg.fl.strategy.params["recluster_every_R"],
        gamma=0.5,
        d_th=_D_TH,
        seed=_SEED,
        update_reference="round_start",
    )
    return FederatedRunner(cfg, trace, world, strategy, device=torch.device("cpu")).run().metrics


@pytest.fixture(scope="module")
def trace() -> Trace:
    """Load the golden tiny trace, skipping when the fixture is absent."""
    if not _GOLDEN.exists():
        pytest.skip("golden tiny_trace fixture not found")
    return Trace.load(_GOLDEN)


@pytest.fixture(scope="module")
def observed(trace: Trace) -> tuple[diag.ObservingRunner, diag.RecordingSimilarity, pd.DataFrame]:
    """Run the observed replay once for the tests that read its records."""
    runner, strategy = diag.build_observed_runner(_config(), trace, _SEED, torch.device("cpu"))
    return runner, strategy, runner.run().metrics


def test_observers_leave_metrics_identical(trace: Trace, observed: tuple) -> None:
    """Every logged column must match the plain runner exactly, NaN positions included."""
    _, _, metrics = observed
    pd.testing.assert_frame_equal(metrics, _plain_metrics(_config(), trace), check_exact=True)


def test_recorded_labels_are_the_partition_the_loop_used(observed: tuple) -> None:
    """Re-clustering the recorded distances at the run's d_th must give its labels and K."""
    _, strategy, metrics = observed
    assert len(strategy.events) >= 2, "the fixture must re-cluster at least once"
    for event in strategy.events:
        labels = diag.partition_at_threshold(event["dist"], _D_TH)
        np.testing.assert_array_equal(labels, event["labels"])
        logged = metrics.loc[metrics["round"] == event["round"], "n_clusters"].item()
        assert np.unique(labels).size == logged


def test_recorded_window_tracks_each_bootstrap(observed: tuple) -> None:
    """The first window starts at 0, and later ones start after the previous clustering."""
    _, strategy, _ = observed
    assert strategy.events[0]["window_start"] == 0
    for previous, current in zip(strategy.events, strategy.events[1:]):
        assert current["window_start"] > previous["round"]


def test_operating_points_reproduce_logged_ba(observed: tuple) -> None:
    """BA recomputed from captured predictions must equal what the loop logged."""
    runner, _, metrics = observed
    points = pd.DataFrame(runner.operating_points)
    assert set(points["model"]) == {"global", "personalized"}
    np.testing.assert_allclose(points["ba_at_fixed"], points["ba"], atol=1e-12)
    expected_rounds = set(metrics.dropna(subset=["ba_probe"])["round"])
    assert set(points["round"]) == expected_rounds


def test_operating_point_matches_hand_computed_values() -> None:
    """Known labels and scores give known rates, BA and best threshold."""
    y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # (6,)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6])  # (6,) separable at 0.4
    point = diag.operating_point(y_true, y_prob)
    assert point["base_rate"] == pytest.approx(0.5)
    assert point["pred_pos_rate"] == pytest.approx(1 / 6)
    assert point["tpr"] == pytest.approx(1 / 3)
    assert point["tnr"] == pytest.approx(1.0)
    assert point["ba_at_fixed"] == pytest.approx(2 / 3)
    assert point["best_ba"] == pytest.approx(1.0)
    assert point["best_threshold"] == pytest.approx(0.4)


def test_best_ba_ignores_threshold_shift_but_fixed_ba_does_not() -> None:
    """
    A monotone rescaling keeps the ROC curve but moves the 0.5 operating point.

    This is the distinction the threshold check relies on.
    """
    rng = np.random.default_rng(0)
    y_true = (rng.random(4000) < 0.4).astype(float)  # (4000,)
    logits = rng.normal(size=4000) + y_true - 1.0  # (4000,) underconfident on positives
    prob = 1.0 / (1.0 + np.exp(-logits))  # (4000,)
    shifted = 1.0 / (1.0 + np.exp(-(logits + 0.5)))  # (4000,) same ranking
    before, after = diag.operating_point(y_true, prob), diag.operating_point(y_true, shifted)
    assert auc(y_true, prob) == pytest.approx(auc(y_true, shifted))
    assert after["best_ba"] == pytest.approx(before["best_ba"])
    assert after["ba_at_fixed"] != pytest.approx(before["ba_at_fixed"], abs=1e-3)
    assert before["ba_at_fixed"] == pytest.approx(
        balanced_accuracy(y_true, (prob >= 0.5).astype(float))
    )


def test_partition_summary_separates_real_clusters_from_singletons() -> None:
    """One big cluster plus singletons has high K but a single real cluster."""
    labels = np.array([0, 0, 0, 0, 1, 2, 3])  # (7,)
    summary = diag.partition_summary(labels, min_real_size=2)
    assert summary == {
        "k": 4,
        "n_small": 3,
        "n_real": 1,
        "largest": 4,
        "frac_in_real": pytest.approx(4 / 7),
    }


def test_ari_against_null_scores_a_true_partition_above_chance() -> None:
    """A partition equal to the reference scores ARI 1 and sits far above its null."""
    reference = np.repeat(np.arange(4), 25)  # (100,)
    scores = diag.ari_against_null(reference, reference, n_permutations=50, seed=0)
    assert scores["ari"] == pytest.approx(1.0)
    assert abs(scores["null_mean"]) < 0.05
    assert scores["z"] > 10


def test_ari_against_null_handles_one_cluster() -> None:
    """A single cluster has no null spread: z must be NaN, not a division error."""
    scores = diag.ari_against_null(
        np.zeros(10, dtype=int), np.arange(10) % 2, n_permutations=5, seed=0
    )
    assert scores["ari"] == pytest.approx(0.0)
    assert np.isnan(scores["z"])


def test_modal_labels_takes_each_satellites_most_frequent_label() -> None:
    """Mode is per column, and ties go to the lowest label."""
    window = np.array([[0, 2, 1], [0, 2, 3], [1, 0, 3], [1, 2, 1]])  # (4, 3)
    np.testing.assert_array_equal(diag.modal_labels(window), [0, 2, 1])


def test_score_event_returns_one_row_per_threshold(observed: tuple, trace: Trace) -> None:
    """Each threshold gets a row, and the run's own d_th reproduces the recorded K."""
    _, strategy, _ = observed
    cfg = _config()
    from orbitfl.experiments.runner import _region_schedule

    schedule = _region_schedule(cfg, trace)  # (R, N)
    event = strategy.events[0]
    rows = diag.score_event(event, schedule, trace.leo_plane_ids, [1.0, _D_TH], _SETTINGS)
    assert [row["d_th"] for row in rows] == [1.0, _D_TH]
    assert rows[1]["k"] == np.unique(event["labels"]).size


def test_compare_to_source_lines_up_shared_rounds() -> None:
    """A replay shorter than its source compares only the rounds both have."""
    base = {"ba_probe": 0.5, "auc_probe": 0.6, "ba_personalized": 0.5, "auc_personalized": 0.6}
    source = pd.DataFrame({"round": [0, 1, 2], "n_clusters": [4, 4, 1], **base})
    replay = pd.DataFrame({"round": [0, 1], "n_clusters": [4, 4], **base})
    merged = diag.compare_to_source(replay, source)
    assert merged["round"].tolist() == [0, 1]
    assert (merged["n_clusters_stored"] == merged["n_clusters_replay"]).all()


def test_build_observed_runner_rejects_other_strategies(trace: Trace) -> None:
    """Replaying a non-similarity run would record nothing useful, so it must fail."""
    plane = {"fl": {"strategy": {"name": "plane", "params": {}}}}
    cfg = load_config(_BASE_CFG, override_dict=plane)
    with pytest.raises(ValueError, match="similarity"):
        diag.build_observed_runner(cfg, trace, _SEED, torch.device("cpu"))
