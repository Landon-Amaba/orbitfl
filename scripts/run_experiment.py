r"""Run one FL experiment from the command line.

Usage:
  # One cell of a sweep, chosen with an inline override:
  python scripts/run_experiment.py configs/base.yaml \
      --override configs/sweeps/concept_shift_hidden_30_sweep.yaml \
      --override-json '{"world": {"concept_shift": 0.9},
                        "fl": {"strategy": {"name": "ephemeris"}},
                        "experiment": {"seeds": [0]}}' \
      --trace data/traces/leo30 --out runs/

  # One task of a sweep, by index into the runs file written by slurm_submit.py:
  python scripts/run_experiment.py configs/base.yaml \
      --override configs/sweeps/efficiency_sweep.yaml \
      --run-idx 7 --runs-file configs/sweeps/efficiency_sweep_runs.json

The positional argument is the base config. Pass configs/base.yaml unless you
want the built-in defaults of orbitfl.config, which describe a different,
smaller constellation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one FL experiment (one config, all seeds or one sweep task)."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Base config YAML or experiment override YAML.",
    )
    parser.add_argument(
        "--override",
        type=Path,
        default=None,
        dest="override_path",
        help="Experiment override YAML deep-merged on top of the base config.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/traces/leo180"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/"),
    )
    parser.add_argument(
        "--run-idx",
        type=int,
        default=None,
        metavar="N",
        help="Index into --runs-file for this SLURM array task.",
    )
    parser.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        metavar="JSON",
        help="JSON file of expanded override dicts produced by slurm_submit.py.",
    )
    parser.add_argument(
        "--override-json",
        type=str,
        default=None,
        metavar="JSON",
        help="Inline JSON override dict (alternative to --run-idx / --runs-file).",
    )
    parser.add_argument(
        "--dirty-ok",
        action="store_true",
    )
    args = parser.parse_args()

    # Per-run override: from the runs file, from inline JSON, or none
    override_dict: dict | None = None
    if args.run_idx is not None and args.runs_file is not None:
        with open(args.runs_file) as f:
            all_runs: list[dict] = json.load(f)
        if args.run_idx >= len(all_runs):
            print(
                f"ERROR: --run-idx {args.run_idx} out of range "
                f"(runs file has {len(all_runs)} entries).",
                file=sys.stderr,
            )
            sys.exit(1)
        override_dict = all_runs[args.run_idx]
    elif args.override_json:
        override_dict = json.loads(args.override_json)

    run_experiment(
        config_path=args.config,
        trace_path=args.trace,
        override_path=args.override_path,
        override_dict=override_dict,
        out_root=args.out,
        allow_dirty=args.dirty_ok,
    )


if __name__ == "__main__":
    main()
