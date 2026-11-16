"""Run directory management.

check_clean_git returns the short git hash of the working tree and refuses a
dirty tree unless told otherwise, so recorded results can be tied to a commit.
create_run_dir creates results/<exp_name>/<UTC-timestamp>_<git-hash>_<token>/.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def check_clean_git(allow_dirty: bool = False) -> str:
    """Return the short git hash of the current commit; raise if the tree is dirty.

    Parameters
    ----------
    allow_dirty : bool  If True, a dirty tree only prints a warning.

    Returns
    -------
    str  Short git hash (7 chars) or 'nogit' if not in a git repository.

    Raises
    ------
    RuntimeError  If the tree is dirty and allow_dirty is False.
    """
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        dirty_files = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        if dirty_files:
            if allow_dirty:
                print("[WARNING] Git tree is dirty. Results may not be reproducible.")
            else:
                raise RuntimeError(
                    "Git tree is dirty; commit changes before recording results, "
                    "or rerun with --dirty-ok."
                )
        return git_hash

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _run_uniquifier() -> str:
    """Token that keeps run directories distinct across concurrent tasks.

    Uses the SLURM array identifiers when present, otherwise a random hex token.
    Array tasks that start in the same UTC second and share a git hash (or 'nogit')
    would otherwise get the same directory name.
    """
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        return f"j{job or '0'}-{task}"
    return uuid.uuid4().hex[:8]


def create_run_dir(
    exp_name: str,
    results_root: Path,
    allow_dirty: bool = False,
) -> Path:
    """Create a timestamped run directory under results_root.

    Directory name: <UTC-timestamp>_<git-short-hash>_<token>
    e.g. results/efficiency/20260922T143055Z_abc1234_j12345-7/

    The token (SLURM array id, else random) keeps concurrent tasks apart. mkdir is
    called with exist_ok=False, so a name collision raises instead of mixing two
    runs in one directory.

    Parameters
    ----------
    exp_name     : str   Experiment name (sub-directory under results_root).
    results_root : Path  Root results directory.
    allow_dirty  : bool  Passed through to check_clean_git.

    Returns
    -------
    Path  The newly created run directory.
    """
    git_hash = check_clean_git(allow_dirty=allow_dirty)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(results_root) / exp_name / f"{ts}_{git_hash}_{_run_uniquifier()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
