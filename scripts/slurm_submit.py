r"""Turn a sweep config into a SLURM array job, or run it locally.

The defaults (partition names, storage paths in the help text) are those of the
University of Florida HiPerGator cluster; change them for another cluster.

Usage:
  # Print a SLURM script and submit it. Run this on the cluster:
  python scripts/slurm_submit.py configs/sweeps/efficiency_sweep.yaml \
      --trace data/traces/leo180 \
      --account mygroup --qos mygroup \
      --conda-env-path /path/to/conda/envs/orbitfl \
      --partition hpg-turin --gpus 1 --mem 32G --time 12:00:00 | sbatch

  # Run every task one after another on the local machine:
  python scripts/slurm_submit.py configs/sweeps/concept_shift_hidden_30_sweep.yaml \
      --serial --trace data/traces/leo30

  # Only print the expanded list of runs:
  python scripts/slurm_submit.py configs/sweeps/efficiency_sweep.yaml --dry-run

Steps:
  1. Load the sweep YAML and expand it into N override dicts.
  2. Write them to <sweep_stem>_runs.json next to the sweep file.
  3. Print a SLURM array script (indices 0..N-1) in which each task runs
       run_experiment.py --run-idx $SLURM_ARRAY_TASK_ID --runs-file <runs.json>

The project root written into the script is this file's location at the time
the script is generated. Generate the script on the cluster, or pass
--project-root, so that the path exists on the compute nodes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.experiments.sweep import sweep_from_file


def _resolve_root(args: argparse.Namespace) -> Path:
    """Return the project root: --project-root if given, else this file's grandparent."""
    if getattr(args, "project_root", None) is not None:
        return Path(args.project_root).resolve()
    return Path(__file__).parent.parent.resolve()


def _repo_relative(path: Path) -> Path:
    """Return path relative to the repo root.

    The generated SLURM script changes into the project root before calling
    run_experiment.py, so repo-relative paths work on any machine. A file outside
    the repo is returned as an absolute path.
    """
    repo_root = Path(__file__).parent.parent.resolve()
    try:
        return path.resolve().relative_to(repo_root)
    except ValueError:
        return path.resolve()


def _write_runs_json(sweep_path: Path, runs: list[dict]) -> Path:
    """Write the expanded run list next to the sweep YAML; return the path."""
    out = sweep_path.parent / f"{sweep_path.stem}_runs.json"
    with open(out, "w") as f:
        json.dump(runs, f, indent=2)
    return out


def _build_sbatch_header(
    sweep_path: Path,
    runs_json: Path,
    n_runs: int,
    args: argparse.Namespace,
    root: Path,
) -> str:
    """Build the #SBATCH directive block (no blank lines)."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sweep_path.stem}",
        f"#SBATCH --array=0-{n_runs - 1}",
        f"#SBATCH --output={root}/logs/{sweep_path.stem}_%A_%a.out",
        f"#SBATCH --error={root}/logs/{sweep_path.stem}_%A_%a.err",
        f"#SBATCH --account={args.account}",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH --partition={args.partition}",
    ]
    if args.gpus:
        lines.append(f"#SBATCH --gpus={args.gpus}")
    lines += [
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --cpus-per-task={args.cpus}",
    ]
    return "\n".join(lines)


def _build_module_block(args: argparse.Namespace) -> str:
    """Build the module-load and PATH block of the batch script."""
    lines = ["module load conda"]
    if args.gpus:
        lines.append("module load cuda")
    # Put the environment's bin directory on PATH instead of calling `conda activate`,
    # which needs shell initialisation that batch jobs may not have.
    lines.append(f"export PATH={args.conda_env_path}/bin:$PATH")
    # Unbuffered stdout, so progress lines reach the log file while the job runs.
    lines.append("export PYTHONUNBUFFERED=1")
    # Limit BLAS/OpenMP threads to the allocated CPUs; torch otherwise starts one
    # thread per physical core.
    lines.append(f"export OMP_NUM_THREADS={args.cpus}")
    lines.append(f"export MKL_NUM_THREADS={args.cpus}")
    return "\n".join(lines)


def _emit_slurm_script(
    sweep_path: Path,
    runs_json: Path,
    n_runs: int,
    args: argparse.Namespace,
) -> str:
    root = _resolve_root(args)
    trace_path = args.trace
    out_path = args.out

    # Repo-relative paths for the sweep and runs file (the script changes into the
    # project root first).
    sweep_rel = _repo_relative(sweep_path)
    runs_rel = _repo_relative(runs_json)

    sbatch_header = _build_sbatch_header(sweep_path, runs_json, n_runs, args, root)
    module_block = _build_module_block(args)

    return (
        f"{sbatch_header}\n"
        f"\n"
        f"{module_block}\n"
        f"\n"
        f"cd {root}\n"
        f"mkdir -p logs\n"
        f"\n"
        f"python scripts/run_experiment.py configs/base.yaml \\\n"
        f"    --override {sweep_rel} \\\n"
        f"    --trace {trace_path} \\\n"
        f"    --out {out_path} \\\n"
        f"    --run-idx $SLURM_ARRAY_TASK_ID \\\n"
        f"    --runs-file {runs_rel}\n"
    )


def _run_serial(
    sweep_path: Path,
    runs_json: Path,
    runs: list[dict],
    args: argparse.Namespace,
) -> None:
    root = _resolve_root(args)
    trace_path = args.trace
    out_path = args.out

    for idx in range(len(runs)):
        print(f"\n-- Run {idx + 1}/{len(runs)}: {runs[idx]} --")
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_experiment.py"),
            "configs/base.yaml",
            "--override", str(sweep_path),
            "--trace", str(trace_path),
            "--out", str(out_path),
            "--run-idx", str(idx),
            "--runs-file", str(runs_json),
        ]
        if getattr(args, "dirty_ok", False):
            cmd.append("--dirty-ok")

        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f"Run {idx} FAILED (exit {result.returncode})", file=sys.stderr)
            if not getattr(args, "keep_going", False):
                sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a SLURM array script or run a sweep serially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweep",
        type=Path,
        help="Sweep YAML (e.g. configs/sweeps/efficiency_sweep.yaml).",
    )
    parser.add_argument("--trace", type=Path, default=Path("data/traces/leo180"))
    parser.add_argument("--out", type=Path, default=Path("results/"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to the project root on the target machine. "
            "Defaults to the grandparent of this script (correct when run on the cluster). "
            "Example: --project-root /blue/mygroup/myuser/escfl"
        ),
    )

    # --- Scheduler account arguments (needed to print a SLURM script) ---
    parser.add_argument(
        "--account",
        default=None,
        metavar="GROUP",
        help=(
            "#SBATCH --account value. On HiPerGator this is your group name "
            "(e.g. 'mygroup'). Required to print a SLURM script; omitting defaults "
            "SLURM to 'ufhpc' and jobs may queue indefinitely or be rejected."
        ),
    )
    parser.add_argument(
        "--qos",
        default=None,
        metavar="QOS",
        help=(
            "#SBATCH --qos value. On HiPerGator typically matches --account "
            "(e.g. 'mygroup'); append '-b' for burst tier (e.g. 'mygroup-b'). "
            "Required for SLURM emission."
        ),
    )
    parser.add_argument(
        "--conda-env-path",
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to the conda environment directory on the cluster. "
            "Example: /blue/mygroup/share/conda/envs/orbitfl. "
            "Create with: conda create -p /blue/mygroup/share/conda/envs/orbitfl. "
            "Required for SLURM emission (not needed for --serial or --dry-run)."
        ),
    )

    # --- SLURM resources ---
    parser.add_argument(
        "--partition",
        default="hpg-turin",
        help=(
            "SLURM partition. HiPerGator options: "
            "hpg-turin (NVIDIA L4 GPUs, default tier) or "
            "hpg-b200 (NVIDIA B200s, requires explicit allocation request). "
            "Default: hpg-turin."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of GPUs per task. Emits '#SBATCH --gpus=N' and adds "
            "'module load cuda' to the job script. Use 0 (default) for CPU-only."
        ),
    )
    parser.add_argument(
        "--mem",
        default=None,
        metavar="SIZE",
        help=(
            "Memory per task, e.g. 32G. Required to print a SLURM script; "
            "SLURM's default is ~600MB which OOM-kills FL jobs immediately."
        ),
    )
    parser.add_argument("--time", default="04:00:00", metavar="HH:MM:SS")
    parser.add_argument("--cpus", type=int, default=4)

    # --- Execution modes ---
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run sweep locally in a loop instead of emitting a SLURM script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded run list without running or submitting.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="In --serial mode, continue past failing runs instead of aborting.",
    )
    parser.add_argument("--dirty-ok", action="store_true")
    args = parser.parse_args()

    runs = sweep_from_file(args.sweep)
    print(f"Sweep '{args.sweep.name}' -> {len(runs)} run(s)", file=sys.stderr)

    if args.dry_run:
        for i, r in enumerate(runs):
            print(f"  [{i:03d}] {json.dumps(r)}")
        return

    runs_json = _write_runs_json(args.sweep, runs)
    print(f"Runs JSON -> {runs_json}", file=sys.stderr)

    if args.serial:
        _run_serial(args.sweep, runs_json, runs, args)
        return

    # --- SLURM script: check that the required arguments are present ---
    errors: list[str] = []
    if args.account is None:
        errors.append(
            "--account is required for SLURM submission.\n"
            "  On HiPerGator this is your group name, e.g. --account mygroup"
        )
    if args.qos is None:
        errors.append(
            "--qos is required for SLURM submission.\n"
            "  On HiPerGator typically matches --account, e.g. --qos mygroup"
        )
    if args.conda_env_path is None:
        errors.append(
            "--conda-env-path is required for SLURM submission.\n"
            "  Example: --conda-env-path /blue/mygroup/share/conda/envs/orbitfl\n"
            "  Create with: conda create -p /blue/mygroup/share/conda/envs/orbitfl"
        )
    if args.mem is None:
        errors.append(
            "--mem is required for SLURM submission.\n"
            "  SLURM's default is ~600MB and will OOM-kill FL jobs immediately.\n"
            "  Specify e.g. --mem 32G"
        )
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    root = _resolve_root(args)
    logs_dir = root / "logs"
    if root.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs directory -> {logs_dir}", file=sys.stderr)
    else:
        # --project-root may point at a path that only exists on the cluster. The job
        # script creates the logs directory itself.
        print(
            f"Note: {root} not found locally; logs dir will be created by the job script.",
            file=sys.stderr,
        )

    script = _emit_slurm_script(args.sweep, runs_json, len(runs), args)
    print(script)  # pipe to sbatch


if __name__ == "__main__":
    main()
