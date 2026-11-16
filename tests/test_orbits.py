"""Tests of the orbital mechanics in src/orbitfl/geometry/orbits.py.

T1  The period of a 550 km LEO is about 95 min (tolerance 2 min).
T2  The MEO period is about 5.8 h (tolerance 0.1 h).
T3  build_walker_delta gives the right plane count and RAAN phasing.
T4  The ground track of one LEO moves west by about 22.5 deg per orbit
    (tolerance 5 deg).
"""

import math
from pathlib import Path

import numpy as np
import pytest
from skyfield.api import load

from orbitfl.geometry.orbits import (
    _R_EARTH_KM,
    _SECONDS_PER_DAY,
    Ephemeris,
    _sma_km_from_period,
    build_meo_tier,
    build_walker_delta,
    load_tles,
    propagate,
)

_TS = load.timescale(builtin=True)
_EPOCH = _TS.utc(2026, 9, 1, 0, 0, 0)


# --- T1 - 550 km LEO orbital period ---

def test_leo_period():
    """550 km circular orbit period must be 95 +/- 2 min."""
    sats = build_walker_delta(
        n_planes=1, sats_per_plane=1,
        alt_km=550.0, inc_deg=53.0,
        phasing_f=0, epoch=_EPOCH,
    )
    assert len(sats) == 1
    sat = sats[0]
    mm_rpd = float(sat.model.no_kozai) * 60.0 * 24.0 / (2.0 * math.pi)  # rad/min -> rev/day
    period_min = _SECONDS_PER_DAY / mm_rpd / 60.0
    assert abs(period_min - 95.0) < 2.0, f"LEO period {period_min:.2f} min outside 95 ± 2"


# --- T2 - MEO period from build_meo_tier ---

def test_meo_period():
    """MEO built with period_hours=5.8 must have ~ 5.8 h period (+/-0.1 h)."""
    target_h = 5.8
    sats = build_meo_tier(n=1, period_hours=target_h, inc_deg=45.0, epoch=_EPOCH)
    assert len(sats) == 1
    sat = sats[0]
    mm_rpd = float(sat.model.no_kozai) * 60.0 * 24.0 / (2.0 * math.pi)
    period_h = _SECONDS_PER_DAY / mm_rpd / 3600.0
    assert abs(period_h - target_h) < 0.1, f"MEO period {period_h:.3f} h outside 5.8 ± 0.1"


def test_meo_altitude():
    """5.8 h MEO altitude must be near 10 000 km (+/-500 km)."""
    a_km = _sma_km_from_period(5.8)
    alt_km = a_km - _R_EARTH_KM
    assert 9500 < alt_km < 10500, f"MEO altitude {alt_km:.0f} km outside expected ~10 000 km"


# --- T3 - Walker-delta builder: plane count and RAAN spacing ---

def test_walker_plane_count():
    """build_walker_delta with n_planes=6, sats_per_plane=5 yields 30 satellites."""
    sats = build_walker_delta(6, 5, 550.0, 53.0, 1, _EPOCH)
    assert len(sats) == 30


def test_walker_raan_spacing():
    """Each plane's first satellite must have RAAN spaced 360/n_planes degrees apart."""
    n_planes, sps = 6, 5
    sats = build_walker_delta(n_planes, sps, 550.0, 53.0, 1, _EPOCH)
    expected_spacing = 360.0 / n_planes

    # Read RAAN from the sgp4 model (nodeo is in radians)
    raans_deg = sorted(
        math.degrees(sats[p * sps].model.nodeo) % 360
        for p in range(n_planes)
    )
    spacings = []
    for i in range(n_planes):
        diff = raans_deg[(i + 1) % n_planes] - raans_deg[i]
        spacings.append(diff % 360)

    for sp in spacings:
        assert abs(sp - expected_spacing) < 0.01, (
            f"RAAN spacing {sp:.4f}° ≠ expected {expected_spacing:.4f}°"
        )


# --- T4 - Ground track advances westward ~22.5 deg/orbit ---

def test_ground_track_westward():
    """
    Sub-satellite longitude must advance westward by 17-30 deg per orbit.

    Earth rotates 360 deg/day; LEO period ~ 95.6 min.
    Earth rotation per orbit ~ 360 x (95.6/1440) ~ 23.9 deg.
    """
    sats = build_walker_delta(1, 1, 550.0, 53.0, 0, _EPOCH)
    sat = sats[0]

    mm_rpd = float(sat.model.no_kozai) * 60.0 * 24.0 / (2.0 * math.pi)
    period_s = _SECONDS_PER_DAY / mm_rpd
    period_days = period_s / _SECONDS_PER_DAY

    t0 = _EPOCH
    t1 = _TS.tt_jd(_EPOCH.tt + period_days)

    eph = propagate([sat], _TS.tt_jd(np.array([t0.tt, t1.tt])))
    lon0 = eph.subpoint_lon[0, 0]
    lon1 = eph.subpoint_lon[1, 0]

    advance = (lon1 - lon0 + 180) % 360 - 180   # signed, wrapped to [-180, 180]
    assert advance < 0, f"Track moved east ({advance:.1f}°); expected westward"
    assert abs(advance) < 30.0, f"|advance| = {abs(advance):.1f}° > 30°"
    assert abs(advance) > 17.0, f"|advance| = {abs(advance):.1f}° < 17°"


# --- T5 - load_tles reads committed file without fetching ---

def test_load_tles_count():
    """load_tles with max_sats=50 returns exactly 50 satellites."""
    tle_path = Path(__file__).parents[1] / "data" / "tle" / "starlink_2026-09-02.txt"
    if not tle_path.exists():
        pytest.skip("Starlink TLE snapshot not present")
    sats = load_tles(tle_path, max_sats=50)
    assert len(sats) == 50


def test_load_tles_shell_filter():
    """shell_filter by inclination returns only in-band satellites."""
    tle_path = Path(__file__).parents[1] / "data" / "tle" / "starlink_2026-09-02.txt"
    if not tle_path.exists():
        pytest.skip("Starlink TLE snapshot not present")
    sats = load_tles(
        tle_path,
        max_sats=200,
        shell_filter={"inc_min_deg": 52.0, "inc_max_deg": 54.0},
    )
    assert len(sats) > 0
    for s in sats:
        inc = math.degrees(s.model.inclo)
        assert 52.0 <= inc <= 54.0, f"Sat {s.name} inc {inc:.2f}° outside filter"


# --- T6 - propagate output shapes and value ranges ---

def test_propagate_shapes():
    """propagate must return Ephemeris with shapes (T, N, 3), (T, N), (T, N)."""
    sats = build_walker_delta(2, 3, 550.0, 53.0, 0, _EPOCH)  # N=6
    T = 10
    times = _TS.tt_jd(_EPOCH.tt + np.arange(T) * 15.0 / _SECONDS_PER_DAY)
    eph = propagate(sats, times)
    assert isinstance(eph, Ephemeris)
    assert eph.ecef.shape == (T, 6, 3)
    assert eph.subpoint_lat.shape == (T, 6)
    assert eph.subpoint_lon.shape == (T, 6)


def test_propagate_lat_lon_range():
    """Sub-satellite lat/lon must stay in valid ranges."""
    sats = build_walker_delta(1, 1, 550.0, 53.0, 0, _EPOCH)
    no_kozai = float(sats[0].model.no_kozai)
    period_days = _SECONDS_PER_DAY / (no_kozai * 60 * 24 / (2 * math.pi)) / _SECONDS_PER_DAY
    times = _TS.tt_jd(_EPOCH.tt + np.linspace(0, period_days, 100))
    eph = propagate(sats, times)
    assert np.all(np.abs(eph.subpoint_lat) <= 90.0)
    assert np.all(np.abs(eph.subpoint_lon) <= 180.0)
