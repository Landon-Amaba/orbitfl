"""Configuration schema.

Frozen dataclasses mirror configs/base.yaml. load_config(base_path,
override_path=None) reads YAML, deep-merges an optional override and returns a
validated Config; unknown keys at any nesting level raise ValueError.
to_dict(cfg) converts back to a plain dict.
"""

import copy
import dataclasses
import typing
from pathlib import Path
from typing import Any

import yaml

# --- Leaf config dataclasses ---


@dataclasses.dataclass(frozen=True)
class SimConfig:
    """Simulation time parameters."""

    timestep_s: int = 15
    duration_h: int = 24
    epoch: str = "2026-09-01T00:00:00Z"


@dataclasses.dataclass(frozen=True)
class LeoConfig:
    """LEO Walker-delta shell parameters."""

    n_planes: int = 6
    sats_per_plane: int = 5
    alt_km: float = 550.0
    inc_deg: float = 53.0
    phasing_f: int = 1  # Walker-delta phasing factor


@dataclasses.dataclass(frozen=True)
class MeoConfig:
    """MEO tier parameters."""

    n: int = 3
    period_h: float = 5.8
    inc_deg: float = 45.0


@dataclasses.dataclass(frozen=True)
class ConstellationConfig:
    """Constellation source and geometry. source: 'walker' | 'tle'."""

    source: str = "walker"
    leo: LeoConfig = dataclasses.field(default_factory=LeoConfig)
    meo: MeoConfig = dataclasses.field(default_factory=MeoConfig)
    isl_max_range_km: float = 5000.0  # maximum distance for an inter-satellite link


@dataclasses.dataclass(frozen=True)
class RegionConfig:
    """One geographic region's radio-environment profile."""

    name: str = ""
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    occupancy: list[float] = dataclasses.field(default_factory=list)  # (C,) stationary occupancy
    mean_on_s: list[float] = dataclasses.field(default_factory=list)  # (C,) mean busy dwell s
    mean_off_s: list[float] = dataclasses.field(default_factory=list)  # (C,) mean idle dwell s


@dataclasses.dataclass(frozen=True)
class WorldConfig:
    """Ground-truth occupancy world parameters."""

    n_channels: int = 32
    flip_noise: float = 0.05
    beam_half_cone_deg: float = 20.0  # footprint half-cone angle
    # Inter-region predictability contrast in [0, 1]. 0.0 gives every region the same
    # lag-4 autocorrelation. Larger values spread the per-region persistence
    # (p_on + p_off), so some regions are easier to forecast than others. This
    # difference is visible in the observation window.
    predictability_contrast: float = 0.0
    # Hidden concept shift in [0, 1]: the fraction of channels whose delta-ahead
    # forecast target is complemented in odd-ranked regions, with the observation
    # window left unchanged. The conflict cannot be seen in the input and can only be
    # resolved from the region identity. 0.0 means no inversion. It changes targets
    # only, so the same trace is reused, and it can be combined with
    # predictability_contrast.
    concept_shift: float = 0.0
    regions: list[RegionConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Forecaster architecture parameters."""

    window: int = 16
    hidden: int = 64
    use_position: bool = True


@dataclasses.dataclass(frozen=True)
class StrategyConfig:
    """FL clustering strategy name and hyperparameters."""

    name: str = "ephemeris"
    params: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class FLConfig:
    """Federated learning loop parameters."""

    round_duration_s: int = 60
    round_budget: int = 300
    local_epochs: int = 1
    local_lr: float = 1e-3
    local_batch: int = 128
    sync_interval: int = 1
    participation_rate: float = 1.0
    eval_interval: int = 1  # evaluate every N rounds (1 = every round);
                            # the final round is always evaluated.
    # cluster_local_train=False: each round a client trains its cluster's model on all
    #   of its history [0, round_end), whichever cluster each sample was collected in.
    #   True: train only on the samples collected while the client belonged to its
    #   current cluster (from the per-round membership schedule). LEO clients cross
    #   every ground region each orbit, so without this a region-keyed model sees data
    #   from all regions and cannot specialise.
    cluster_local_train: bool = False
    # per_sample_eval=False: personalized BA scores each satellite's whole probe set
    #   under its final-round cluster model. True: each probe sample is scored under
    #   the model of the cluster the satellite belonged to at that sample's timestep,
    #   using the strategy's own membership schedule.
    # Both options need a strategy whose membership does not depend on training state
    # (ephemeris, plane, random_churn) so the schedule can be precomputed; they are
    # not supported for similarity.
    per_sample_eval: bool = False
    strategy: StrategyConfig = dataclasses.field(default_factory=StrategyConfig)


@dataclasses.dataclass(frozen=True)
class AllocationConfig:
    """Channel allocation and interference-graph parameters."""

    threshold_db: float = -6.0
    # This default (20.0) is not the value used in the thesis. configs/base.yaml sets
    # elevation_mask_deg = 12.0, and every experiment loads base.yaml first. A config
    # that does not build on base.yaml would get 20.0, which gives a much sparser
    # interference graph.
    elevation_mask_deg: float = 20.0
    theta: float = 0.5
    peering: bool = True
    n_channels: int = 32


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Experiment name, seeds and forecast horizon."""

    name: str = "base"
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0])
    delta_steps: int = 4


@dataclasses.dataclass(frozen=True)
class Config:
    """Top-level simulation config. Load via load_config(path)."""

    seed: int = 0
    sim: SimConfig = dataclasses.field(default_factory=SimConfig)
    constellation: ConstellationConfig = dataclasses.field(default_factory=ConstellationConfig)
    world: WorldConfig = dataclasses.field(default_factory=WorldConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    fl: FLConfig = dataclasses.field(default_factory=FLConfig)
    allocation: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)


# --- Dict <-> dataclass conversion ---


def _convert_value(hint: Any, val: Any) -> Any:
    """Convert val to the type described by hint, recursing into dataclasses."""
    if val is None:
        return val

    origin = getattr(hint, "__origin__", None)

    # Union / Optional[X]
    if origin is typing.Union:
        non_none = [a for a in hint.__args__ if a is not type(None)]
        if non_none:
            return _convert_value(non_none[0], val)
        return val

    # list[X]
    if origin is list:
        args = getattr(hint, "__args__", None)
        if args and dataclasses.is_dataclass(args[0]) and isinstance(val, list):
            return [_from_dict(args[0], item) for item in val]
        return val

    # tuple[X, ...]
    if origin is tuple:
        args = getattr(hint, "__args__", None)
        if (
            args
            and len(args) == 2
            and args[1] is Ellipsis
            and dataclasses.is_dataclass(args[0])
            and isinstance(val, (list, tuple))
        ):
            return tuple(_from_dict(args[0], item) for item in val)
        return val

    # Nested dataclass
    if dataclasses.is_dataclass(hint) and isinstance(val, dict):
        return _from_dict(hint, val)

    return val


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Convert a nested dict to a frozen dataclass, raising on unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {cls.__name__}: {sorted(unknown)}"
        )

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for f in dataclasses.fields(cls):
        if f.name not in data or data[f.name] is None:
            continue  # use dataclass default
        kwargs[f.name] = _convert_value(hints.get(f.name, Any), data[f.name])

    return cls(**kwargs)


def to_dict(cfg: Any) -> Any:
    """Recursively convert a frozen dataclass (or nested structure) to a plain dict."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {
            f.name: to_dict(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)
        }
    if isinstance(cfg, (list, tuple)):
        converted = [to_dict(item) for item in cfg]
        return type(cfg)(converted) if isinstance(cfg, tuple) else converted
    return cfg


# --- YAML loading ---


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(
    base_path: Path,
    override_path: Path | None = None,
    override_dict: dict | None = None,
) -> Config:
    """
    Load a Config from YAML.

    Merge order (each layer overrides the previous):
      1. base_path YAML
      2. override_path YAML  (if provided)
      3. override_dict       (if provided - used by sweep runner and tests)

    Raises ValueError if any YAML key is not in the schema.
    """
    with open(base_path) as f:
        data: dict = yaml.safe_load(f) or {}
    if override_path is not None:
        with open(override_path) as f:
            override: dict = yaml.safe_load(f) or {}
        override.pop("sweep", None)  # sweep files carry a 'sweep:' key for slurm_submit; ignore it
        data = _deep_merge(data, override)
    if override_dict is not None:
        data = _deep_merge(data, override_dict)
    return _from_dict(Config, data)
