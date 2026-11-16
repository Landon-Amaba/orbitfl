"""Tests for src/orbitfl/config.py.

YAML and dataclass round-trip; unknown keys raise; every config in
configs/experiments/ loads.
"""

import dataclasses
import tempfile
from pathlib import Path

import pytest
import yaml

from orbitfl.config import (
    AllocationConfig,
    Config,
    ConstellationConfig,
    ExperimentConfig,
    FLConfig,
    LeoConfig,
    MeoConfig,
    ModelConfig,
    RegionConfig,
    SimConfig,
    StrategyConfig,
    WorldConfig,
    _from_dict,
    load_config,
    to_dict,
)
from orbitfl.experiments.sweep import sweep_from_file

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
BASE_YAML = CONFIGS_DIR / "base.yaml"


# --- Basic loading ---


def test_base_yaml_loads():
    """base.yaml loads into a Config without error."""
    cfg = load_config(BASE_YAML)
    assert isinstance(cfg, Config)


def test_frozen_prevents_mutation():
    """All fields are immutable after construction."""
    cfg = Config()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.seed = 999  # type: ignore[misc]


def test_nested_types():
    """Nested configs have the correct dataclass types after loading base.yaml."""
    cfg = load_config(BASE_YAML)
    assert isinstance(cfg.sim, SimConfig)
    assert isinstance(cfg.constellation, ConstellationConfig)
    assert isinstance(cfg.constellation.leo, LeoConfig)
    assert isinstance(cfg.constellation.meo, MeoConfig)
    assert isinstance(cfg.world, WorldConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.fl, FLConfig)
    assert isinstance(cfg.fl.strategy, StrategyConfig)
    assert isinstance(cfg.allocation, AllocationConfig)
    assert isinstance(cfg.experiment, ExperimentConfig)


def test_regions_are_region_configs():
    """Each entry in world.regions is a RegionConfig, not a raw dict."""
    cfg = load_config(BASE_YAML)
    assert len(cfg.world.regions) > 0
    for r in cfg.world.regions:
        assert isinstance(r, RegionConfig)


# --- Default values ---


def test_defaults():
    """Config() has the documented dataclass defaults."""
    cfg = Config()
    assert cfg.seed == 0
    assert cfg.sim.timestep_s == 15
    assert cfg.sim.duration_h == 24
    assert cfg.constellation.source == "walker"
    assert cfg.constellation.leo.n_planes == 6
    assert cfg.constellation.leo.sats_per_plane == 5
    assert cfg.constellation.meo.n == 3
    assert cfg.world.n_channels == 32
    assert cfg.model.window == 16
    assert cfg.fl.round_budget == 300
    assert cfg.fl.strategy.name == "ephemeris"
    assert cfg.allocation.peering is True
    assert cfg.experiment.delta_steps == 4


def test_values_from_base_yaml():
    """Key values loaded from base.yaml."""
    cfg = load_config(BASE_YAML)
    assert cfg.sim.timestep_s == 15
    assert cfg.sim.duration_h == 24
    assert cfg.constellation.source == "walker"
    assert cfg.world.beam_half_cone_deg == pytest.approx(20.0)
    assert cfg.world.flip_noise == pytest.approx(0.05)
    assert cfg.fl.round_budget == 300
    assert cfg.allocation.threshold_db == pytest.approx(-6.0)


# --- Unknown-key validation ---


def test_unknown_top_level_key_raises():
    """An unknown key at the top level raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config"):
        _from_dict(Config, {"seed": 0, "bogus_key": 1})


def test_unknown_nested_key_raises():
    """An unknown key inside a nested config raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config"):
        _from_dict(Config, {"sim": {"timestep_s": 15, "not_a_field": True}})


def test_unknown_region_key_raises():
    """An unknown key inside a RegionConfig raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config"):
        _from_dict(
            Config,
            {
                "world": {
                    "regions": [{"name": "test", "mystery_field": 99}]
                }
            },
        )


# --- Round-trip ---


def test_round_trip_via_to_dict():
    """Config -> to_dict -> _from_dict -> to_dict is idempotent."""
    cfg = load_config(BASE_YAML)
    d = to_dict(cfg)
    cfg2 = _from_dict(Config, d)
    assert to_dict(cfg2) == d


def test_round_trip_via_yaml_file():
    """Config -> to_dict -> YAML file -> load_config reproduces equal field values."""
    cfg = load_config(BASE_YAML)
    d = to_dict(cfg)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(d, f)
        tmp_path = Path(f.name)
    try:
        cfg2 = load_config(tmp_path)
        assert cfg.seed == cfg2.seed
        assert cfg.sim.timestep_s == cfg2.sim.timestep_s
        assert cfg.fl.round_budget == cfg2.fl.round_budget
        assert cfg.constellation.leo.n_planes == cfg2.constellation.leo.n_planes
        assert len(cfg.world.regions) == len(cfg2.world.regions)
    finally:
        tmp_path.unlink()


# --- Override / deep-merge ---


def test_override_merges_correctly():
    """An override YAML changes only the keys it specifies."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump({"experiment": {"name": "override_test", "seeds": [42]}}, f)
        override_path = Path(f.name)
    try:
        cfg = load_config(BASE_YAML, override_path=override_path)
        assert cfg.experiment.name == "override_test"
        assert cfg.experiment.seeds == [42]
        # Unspecified fields should be unchanged from base
        assert cfg.sim.timestep_s == 15
        assert cfg.fl.round_budget == 300
    finally:
        override_path.unlink()


# --- All experiment configs load ---


def test_experiment_configs_load():
    """Every YAML in configs/experiments/ loads without error as a Config override."""
    exp_dir = CONFIGS_DIR / "experiments"
    yamls = sorted(exp_dir.glob("*.yaml"))
    assert yamls, f"No experiment configs found in {exp_dir}"
    for path in yamls:
        cfg = load_config(BASE_YAML, override_path=path)
        assert isinstance(cfg, Config), f"load_config failed for {path.name}"
        assert isinstance(cfg.experiment.name, str)


# --- All sweep configs load - fixed keys and every expanded sweep combination ---


def test_sweep_configs_load():
    """
    Every YAML in configs/sweeps/ is a valid Config override, and every per-run
    override its ``sweep:`` block expands to also produces a valid Config.

    load_config strips the top-level ``sweep:`` key (it is consumed by
    slurm_submit, not the schema), so the fixed portion of each sweep file must
    merge cleanly onto base.yaml; and each Cartesian combination the sweep
    expands to (the dicts slurm_submit feeds back as --override-json) must yield
    a schema-valid Config. This guards the new _fullv2 sweeps and every existing
    sweep against silent schema drift.
    """
    sweep_dir = CONFIGS_DIR / "sweeps"
    yamls = sorted(sweep_dir.glob("*.yaml"))
    assert yamls, f"No sweep configs found in {sweep_dir}"
    for path in yamls:
        # Fixed keys merge onto base (sweep: is stripped inside load_config).
        cfg = load_config(BASE_YAML, override_path=path)
        assert isinstance(cfg, Config), f"load_config failed for {path.name}"
        # Every expanded per-run override is itself a valid Config.
        runs = sweep_from_file(path)
        assert runs, f"sweep expansion empty for {path.name}"
        for override in runs:
            run_cfg = load_config(
                BASE_YAML, override_path=path, override_dict=override
            )
            assert isinstance(run_cfg, Config), (
                f"expanded override failed for {path.name}: {override}"
            )
