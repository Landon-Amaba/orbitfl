"""Tests for src/orbitfl/utils/runs.py: every call gets its own run directory.

SLURM array tasks that start in the same UTC second and share the 'nogit' hash
would otherwise get the same directory name and overwrite each other's output.
"""

from __future__ import annotations

import orbitfl.utils.runs as runs
from orbitfl.utils.runs import _run_uniquifier, create_run_dir


def test_rapid_calls_same_second_nogit_are_unique(tmp_path, monkeypatch):
    """50 calls in the same second with the 'nogit' hash and no SLURM ids are all distinct."""
    monkeypatch.setattr(runs, "check_clean_git", lambda allow_dirty=False: "nogit")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    dirs = [create_run_dir("exp", tmp_path, allow_dirty=True) for _ in range(50)]
    assert len({str(d) for d in dirs}) == 50, "run directories collided"
    assert all(d.is_dir() for d in dirs)


def test_uniquifier_uses_slurm_array_id(monkeypatch):
    """Under a SLURM array task the token holds the job and task id."""
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert _run_uniquifier() == "j12345-7"


def test_uniquifier_random_without_slurm(monkeypatch):
    """Off SLURM the token is a distinct random hex each call."""
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    a, b = _run_uniquifier(), _run_uniquifier()
    assert a != b
    assert len(a) == 8
