r"""Golden-trace regression tests for src/orbitfl/trace/.

- Regenerating the tiny golden trace reproduces the stored tables.
- TRACE_VERSION in the metadata matches schema.TRACE_VERSION.
- Trace.load() raises TraceVersionError on a version mismatch.

The golden trace is in tests/golden/tiny_trace/ (four parquet files), generated
with:

    python scripts/make_trace.py \
        --config configs/base.yaml \
        --override configs/experiments/tiny_trace.yaml \
        --out tests/golden/tiny_trace

Tables are sorted by primary key before comparison. Integer columns must be
equal; float columns must agree within the tolerance below. Bit-exact equality
does not hold across machines: SGP4, libm and BLAS builds differ in the last
few bits (differences of about 2e-13 km on macOS and 2e-11 km on Linux x86_64
were observed in 2 of 1920 elements).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from skyfield.api import load

from orbitfl.config import load_config
from orbitfl.geometry.orbits import (
    Constellation,
    build_meo_tier,
    build_walker_delta,
)
from orbitfl.trace.generate import TraceGenerator
from orbitfl.trace.reader import Trace, TraceVersionError
from orbitfl.trace.schema import (
    SCHEMA_META,
    TRACE_VERSION,
)

_TS = load.timescale(builtin=True)

# --- Paths ---

_TESTS_DIR = Path(__file__).parent
_GOLDEN_DIR = _TESTS_DIR / "golden" / "tiny_trace"
_CONFIGS_DIR = _TESTS_DIR.parent / "configs"

# Float tolerances for comparing a regenerated trace with the golden one. They
# only absorb floating-point differences between platforms:
#   * The largest value in the tables is the MEO semi-major axis, about 1.64e4 km,
#     where one float64 step is about 3.6e-12 km. _FLOAT_RTOL = 1e-12 allows for
#     accumulated library differences.
#   * _FLOAT_ATOL = 1e-9 km (1 micrometre) covers coordinates that pass through 0,
#     where a relative test fails.
#   * A real change in geometry or schema moves values by metres (1e-3 km) or
#     more, far above these tolerances, and still fails the test.
_FLOAT_RTOL = 1e-12
_FLOAT_ATOL = 1e-9  # km
_BASE_YAML = _CONFIGS_DIR / "base.yaml"
_TINY_YAML = _CONFIGS_DIR / "experiments" / "tiny_trace.yaml"


# --- Helpers ---


def _build_constellation(cfg):
    """Build the tiny constellation from a Config (mirrors make_trace.py logic)."""
    from datetime import datetime

    leo_cfg = cfg.constellation.leo
    meo_cfg = cfg.constellation.meo
    epoch_dt = datetime.fromisoformat(cfg.sim.epoch.replace("Z", "+00:00"))
    epoch_sf = _TS.from_datetime(epoch_dt)

    leos = build_walker_delta(
        n_planes=leo_cfg.n_planes,
        sats_per_plane=leo_cfg.sats_per_plane,
        alt_km=leo_cfg.alt_km,
        inc_deg=leo_cfg.inc_deg,
        phasing_f=leo_cfg.phasing_f,
        epoch=epoch_sf,
    )
    meos = build_meo_tier(
        n=meo_cfg.n,
        period_hours=meo_cfg.period_h,
        inc_deg=meo_cfg.inc_deg,
        epoch=epoch_sf,
    )
    plane_id = np.repeat(np.arange(leo_cfg.n_planes, dtype=np.int32), leo_cfg.sats_per_plane)
    in_plane_idx = np.tile(np.arange(leo_cfg.sats_per_plane, dtype=np.int32), leo_cfg.n_planes)
    return Constellation(leos=leos, meos=meos, plane_id=plane_id, in_plane_idx=in_plane_idx)


def _load_table(path: Path, name: str) -> pd.DataFrame:
    return pq.read_table(path / f"{name}.parquet").to_pandas()


def _assert_tables_equal(
    golden_df: pd.DataFrame, regen_df: pd.DataFrame, sort_cols: list, name: str
) -> None:
    """Sort both DataFrames by sort_cols, then compare content column by column."""
    g = golden_df.sort_values(sort_cols).reset_index(drop=True)
    r = regen_df.sort_values(sort_cols).reset_index(drop=True)

    assert list(g.columns) == list(r.columns), f"{name}: column mismatch"
    assert len(g) == len(r), f"{name}: row count {len(g)} ≠ {len(r)}"

    for col in g.columns:
        gc = g[col].to_numpy()
        rc = r[col].to_numpy()
        if gc.dtype.kind in ("i", "u"):  # integer
            np.testing.assert_array_equal(gc, rc, err_msg=f"{name}.{col} integer mismatch")
        elif gc.dtype.kind == "f":       # float
            np.testing.assert_allclose(gc, rc, rtol=_FLOAT_RTOL, atol=_FLOAT_ATOL,
                                       err_msg=f"{name}.{col} float mismatch")
        else:                            # string / object
            assert list(gc) == list(rc), f"{name}.{col} string mismatch"


# --- Fixtures ---


@pytest.fixture(scope="module")
def cfg():
    return load_config(_BASE_YAML, override_path=_TINY_YAML)


@pytest.fixture(scope="module")
def regen_trace_dir(cfg, tmp_path_factory):
    """Generate a fresh tiny trace in a temp directory (once per test session)."""
    out = tmp_path_factory.mktemp("regen_trace")
    con = _build_constellation(cfg)
    gen = TraceGenerator(cfg, con)
    gen.run(out)
    return out


# --- Tests ---


def test_golden_dir_exists():
    """Golden trace directory must exist (generate with make_trace.py if missing)."""
    assert _GOLDEN_DIR.exists(), (
        f"Golden trace not found at {_GOLDEN_DIR}. "
        "Generate it with:\n"
        "  python scripts/make_trace.py "
        "--config configs/base.yaml "
        "--override configs/experiments/tiny_trace.yaml "
        "--out tests/golden/tiny_trace"
    )


def test_trace_version_matches_schema():
    """TRACE_VERSION in meta.parquet matches schema.TRACE_VERSION."""
    meta = _load_table(_GOLDEN_DIR, "meta")
    assert meta["trace_version"].iloc[0] == TRACE_VERSION


def test_positions_table_identical(regen_trace_dir):
    """Regenerated positions table is content-identical to the golden."""
    g = _load_table(_GOLDEN_DIR, "positions")
    r = _load_table(regen_trace_dir, "positions")
    _assert_tables_equal(g, r, sort_cols=["t_idx", "sat_id"], name="positions")


def test_assignment_table_identical(regen_trace_dir):
    """Regenerated assignment table is content-identical to the golden."""
    g = _load_table(_GOLDEN_DIR, "assignment")
    r = _load_table(regen_trace_dir, "assignment")
    _assert_tables_equal(g, r, sort_cols=["t_idx", "leo_id"], name="assignment")


def test_visibility_table_identical(regen_trace_dir):
    """Regenerated visibility table is content-identical to the golden."""
    g = _load_table(_GOLDEN_DIR, "visibility")
    r = _load_table(regen_trace_dir, "visibility")
    _assert_tables_equal(g, r, sort_cols=["t_idx", "a", "b"], name="visibility")


def test_meta_fields_match(regen_trace_dir):
    """The regenerated metadata matches the golden one, except git_hash and config_hash.

    git_hash changes with every commit. config_hash is not compared because the
    golden trace was generated from an earlier revision of configs/base.yaml, so
    its stored hash differs from the current one. The tiny-trace geometry is the
    same: the assignment table matches exactly and the positions and visibility
    tables agree within the float tolerance. config_hash is still checked for
    presence and format below.
    """
    g = _load_table(_GOLDEN_DIR, "meta")
    r = _load_table(regen_trace_dir, "meta")
    for col in ["trace_version", "timestep_s", "duration_s", "epoch", "seed"]:
        assert g[col].iloc[0] == r[col].iloc[0], (
            f"meta.{col} mismatch: {g[col].iloc[0]!r} ≠ {r[col].iloc[0]!r}"
        )

    # config_hash is not compared for equality (see the docstring), but both tables
    # must hold a 16-digit hexadecimal value.
    for label, df in (("golden", g), ("regenerated", r)):
        h = df["config_hash"].iloc[0]
        assert isinstance(h, str) and len(h) == 16 and all(c in "0123456789abcdef" for c in h), (
            f"{label} meta.config_hash is not a 16-hex-digit stamp: {h!r}"
        )


def test_loader_rejects_version_mismatch(tmp_path):
    """Trace.load() raises TraceVersionError when trace_version != TRACE_VERSION."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Write a meta table with a wrong version
    wrong_version = "0.0"
    assert wrong_version != TRACE_VERSION
    meta_table = pa.table(
        {
            "trace_version": pa.array([wrong_version], type=pa.string()),
            "config_hash": pa.array(["deadbeef"], type=pa.string()),
            "git_hash": pa.array(["abc1234"], type=pa.string()),
            "timestep_s": pa.array([15], type=pa.int32()),
            "duration_s": pa.array([3600], type=pa.int64()),
            "epoch": pa.array(["2026-09-01T00:00:00Z"], type=pa.string()),
            "seed": pa.array([0], type=pa.int32()),
        },
        schema=SCHEMA_META,
    )
    pq.write_table(meta_table, tmp_path / "meta.parquet")

    # Trace.load() must raise before reading the other (absent) tables
    with pytest.raises(TraceVersionError, match="version mismatch"):
        Trace.load(tmp_path)


def test_reader_api(regen_trace_dir):
    """Basic Trace reader API: shapes, types, and range constraints."""
    trace = Trace.load(regen_trace_dir)

    T = trace.n_timesteps
    N_leo = trace.n_leo

    # positions_at returns a DataFrame with all satellites at one timestep
    pos = trace.positions_at(0)
    assert len(pos) == N_leo + 2, "positions_at(0) should have N_leo+N_meo rows"  # N_meo=2

    # subpoints returns (T, 2) lat/lon for one satellite
    sp = trace.subpoints(0)
    assert sp.shape == (T, 2)
    assert np.all(np.abs(sp[:, 0]) <= 90.0),  "latitudes must be in [-90, 90]"

    # assignment_at returns (N_leo,) int with values in [-1, N_meo)
    asgn = trace.assignment_at(0)
    assert asgn.shape == (N_leo,)
    assert asgn.dtype == np.int32
    assert np.all((asgn >= -1) & (asgn < 2))  # N_meo=2

    # membership_schedule returns (R, N_leo) matching assignment_at for round r=0
    round_times = np.array([0.0, 60.0, 120.0])
    sched = trace.membership_schedule(round_times)
    assert sched.shape == (3, N_leo)
    np.testing.assert_array_equal(sched[0], trace.assignment_at(0))

    # churn_rate_per_orbit is non-negative
    assert trace.churn_rate_per_orbit() >= 0.0
