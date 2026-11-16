"""Sweep expansion.

expand_sweep turns a sweep specification into one config override per run (the
Cartesian product of the swept values). scripts/slurm_submit.py turns that list
into a SLURM array job, or runs it locally with --serial.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml


def expand_sweep(sweep_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Cartesian expansion of a sweep specification into per-run config overrides.

    The sweep_spec has two top-level keys:
      base   : path to a base YAML config (str or Path)
      sweep  : dict mapping dotted config keys to lists of values to sweep

    Example sweep_spec:
      base: configs/base.yaml
      sweep:
        seed: [0, 1, 2, 3, 4]
        fl.strategy.name: [ephemeris, plane]
        experiment.delta_steps: [4, 8]

    This yields 5 x 2 x 2 = 20 per-run dicts, each with the base config
    deep-merged with one combination of the sweep values.

    Parameters
    ----------
    sweep_spec : dict  Sweep specification (see above).

    Returns
    -------
    list[dict]  One dict per run, ready to pass to load_config or write as YAML.
    """
    sweep: dict[str, list] = sweep_spec.get("sweep", {})
    if not sweep:
        return [{}]  # no sweep -> one run with no overrides

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]

    runs = []
    for combo in itertools.product(*value_lists):
        override: dict[str, Any] = {}
        for key, val in zip(keys, combo):
            _set_dotted(override, key, val)
        runs.append(override)

    return runs


def sweep_from_file(path: Path) -> list[dict[str, Any]]:
    """
    Load a sweep YAML file and return the expanded list of override dicts.

    The YAML file may contain any subset of the base.yaml schema plus a
    top-level 'sweep' key with lists of values to vary.

    Parameters
    ----------
    path : Path  Path to the sweep YAML file.

    Returns
    -------
    list[dict]  Expanded per-run overrides (excludes the 'sweep' key itself).
    """
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    return expand_sweep(spec)


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """
    Set a value in a nested dict using a dotted key path.

    _set_dotted(d, "fl.round_budget", 300) is equivalent to:
      d["fl"] = {"round_budget": 300}

    Parameters
    ----------
    d          : dict   Target dict (mutated in-place).
    dotted_key : str    Dot-separated key path (e.g. "fl.strategy.name").
    value      : Any    Value to set at that path.
    """
    parts = dotted_key.split(".")
    node = d
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
