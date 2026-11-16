"""Constellation construction and SGP4 propagation.

Satellites are skyfield EarthSatellite objects, either read from a TLE file
(load_tles) or synthesised as circular orbits (build_walker_delta,
build_meo_tier). propagate turns them into positions and sub-satellite points.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
from sgp4.api import WGS84, Satrec
from sgp4.exporter import export_tle
from skyfield.api import EarthSatellite, load, wgs84

from orbitfl.geometry.constants import MU_KM3_S2 as _MU_KM3_S2
from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM

_SECONDS_PER_DAY: float = 86400.0

_TS = load.timescale(builtin=True)


# --- Public dataclasses ---


@dataclasses.dataclass
class Constellation:
    """LEO + MEO EarthSatellite objects with orbital plane metadata."""

    leos: list           # list[EarthSatellite], length N_leo
    meos: list           # list[EarthSatellite], length N_meo
    plane_id: np.ndarray         # (N_leo,) int - orbital plane index per LEO
    in_plane_idx: np.ndarray     # (N_leo,) int - seat index within plane


@dataclasses.dataclass
class Ephemeris:
    """Propagated satellite positions over T timesteps.

    ecef holds the Cartesian position returned by skyfield (GCRS, an inertial
    frame), despite its name. The simulator only uses distances and angles between
    these vectors, which are the same in any Earth-centred frame. The sub-satellite
    latitude and longitude are Earth-fixed.
    """

    ecef: np.ndarray  # (T, N, 3) km, Cartesian position (see class docstring)
    subpoint_lat: np.ndarray  # (T, N) degrees latitude
    subpoint_lon: np.ndarray  # (T, N) degrees longitude


# --- Internal helpers ---


def _period_s(alt_km: float) -> float:
    """Orbital period in seconds for a circular orbit at alt_km above surface."""
    a_km = _R_EARTH_KM + alt_km
    return 2.0 * math.pi * math.sqrt(a_km**3 / _MU_KM3_S2)


def _mean_motion_rpd(alt_km: float) -> float:
    """Mean motion in revolutions per day for a circular orbit."""
    return _SECONDS_PER_DAY / _period_s(alt_km)


def _sma_km_from_period(period_hours: float) -> float:
    """Semi-major axis in km from orbital period in hours."""
    T_s = period_hours * 3600.0
    return (_MU_KM3_S2 * (T_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)


def _sgp4_epoch(t) -> float:
    """Days since 1949-12-31 00:00 UT for a skyfield Time (uses TT, close enough to UTC here)."""
    return float(t.tt) - 2433281.5


def _satrec_circular(
    sat_num: int,
    sgp4_epoch: float,
    inc_rad: float,
    raan_rad: float,
    mean_anom_rad: float,
    alt_km: float,
) -> Satrec:
    """Build a Satrec for a zero-drag, zero-eccentricity circular orbit."""
    a_km = _R_EARTH_KM + alt_km
    T_s = 2.0 * math.pi * math.sqrt(a_km**3 / _MU_KM3_S2)
    n_rad_per_min = 2.0 * math.pi / (T_s / 60.0)

    sat = Satrec()
    sat.sgp4init(
        WGS84,
        "i",           # opsmode: improved
        sat_num,
        sgp4_epoch,
        0.0,           # BSTAR drag
        0.0,           # ndot
        0.0,           # nddot
        1e-7,          # eccentricity (~ 0)
        0.0,           # argument of perigee (radians)
        inc_rad,
        mean_anom_rad,
        n_rad_per_min,
        raan_rad,
    )
    return sat


def _to_earth_satellite(satrec: Satrec, name: str) -> EarthSatellite:
    l1, l2 = export_tle(satrec)
    return EarthSatellite(l1, l2, name, _TS)


def _tle_alt_km(line2: str) -> float:
    """Approximate altitude in km from a TLE line 2 (via mean motion)."""
    mm_rpd = float(line2[52:63])            # revs per day
    T_s = _SECONDS_PER_DAY / mm_rpd
    a_km = (_MU_KM3_S2 * (T_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    return a_km - _R_EARTH_KM


def _tle_inc_deg(line2: str) -> float:
    return float(line2[8:16])              # inclination field, cols 9-16


# --- Public API ---


def load_tles(
    path: Path,
    max_sats: int | None = None,
    shell_filter: dict | None = None,
) -> list:
    """Read satellites from a local three-line-element file. Nothing is downloaded.

    shell_filter keys (all optional):
      alt_min_km, alt_max_km    altitude band in km
      inc_min_deg, inc_max_deg  inclination band in degrees

    Returns list[EarthSatellite].
    """
    lines = Path(path).read_text().splitlines()
    lines = [ln for ln in lines if ln.strip()]

    sats: list = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()

        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue

        if shell_filter:
            inc = _tle_inc_deg(l2)
            alt = _tle_alt_km(l2)
            if "inc_min_deg" in shell_filter and inc < shell_filter["inc_min_deg"]:
                i += 3
                continue
            if "inc_max_deg" in shell_filter and inc > shell_filter["inc_max_deg"]:
                i += 3
                continue
            if "alt_min_km" in shell_filter and alt < shell_filter["alt_min_km"]:
                i += 3
                continue
            if "alt_max_km" in shell_filter and alt > shell_filter["alt_max_km"]:
                i += 3
                continue

        sats.append(EarthSatellite(l1, l2, name, _TS))
        i += 3

        if max_sats is not None and len(sats) >= max_sats:
            break

    return sats


def build_walker_delta(
    n_planes: int,
    sats_per_plane: int,
    alt_km: float,
    inc_deg: float,
    phasing_f: int,
    epoch,
) -> list:
    """
    Synthesize EarthSatellite objects for a Walker-delta constellation.

    RAAN spacing: 360 / n_planes degrees.
    Mean anomaly for satellite (plane p, seat j):
        M = j * (360/sats_per_plane) + p * phasing_f * (360 / (n_planes * sats_per_plane))

    Returns list[EarthSatellite] of length n_planes * sats_per_plane.
    """
    ep = _sgp4_epoch(epoch)
    sats: list = []

    for p in range(n_planes):
        raan_rad = math.radians(p * 360.0 / n_planes)
        for j in range(sats_per_plane):
            mean_anom_deg = (
                j * (360.0 / sats_per_plane)
                + p * phasing_f * (360.0 / (n_planes * sats_per_plane))
            )
            sat_num = p * sats_per_plane + j + 1
            satrec = _satrec_circular(
                sat_num, ep,
                math.radians(inc_deg),
                raan_rad,
                math.radians(mean_anom_deg % 360.0),
                alt_km,
            )
            sats.append(_to_earth_satellite(satrec, f"WLK-P{p:02d}S{j:02d}"))

    return sats


def build_meo_tier(
    n: int,
    period_hours: float,
    inc_deg: float,
    epoch,
) -> list:
    """Synthesize EarthSatellite objects for a circular MEO tier.

    Semi-major axis: a = (mu * (T / 2pi)^2)^(1/3). A 5.8 h period gives
    a ~ 16,400 km, i.e. about 10,000 km altitude. Satellites are evenly spaced
    in RAAN (360/n degrees apart) and all start at mean anomaly 0.

    Returns list[EarthSatellite] of length n.
    """
    a_km = _sma_km_from_period(period_hours)
    alt_km = a_km - _R_EARTH_KM
    ep = _sgp4_epoch(epoch)
    sats: list = []

    for k in range(n):
        raan_rad = math.radians(k * 360.0 / n)
        sat_num = 90000 + k + 1
        satrec = _satrec_circular(
            sat_num, ep,
            math.radians(inc_deg),
            raan_rad,
            0.0,  # mean anomaly: all start at the ascending node
            alt_km,
        )
        sats.append(_to_earth_satellite(satrec, f"MEO-{k:02d}"))

    return sats


def propagate(sats: list, times) -> Ephemeris:
    """Propagate a list of EarthSatellite objects over a skyfield Time vector.

    Returns Ephemeris:
      ecef          (T, N, 3) km      Cartesian positions (see Ephemeris)
      subpoint_lat  (T, N) degrees    sub-satellite latitude
      subpoint_lon  (T, N) degrees    sub-satellite longitude
    """
    N = len(sats)
    T = int(np.atleast_1d(times.tt).shape[0])

    ecef = np.empty((T, N, 3))
    lat = np.empty((T, N))
    lon = np.empty((T, N))

    for i, sat in enumerate(sats):
        geo = sat.at(times)
        pos = geo.position.km              # (3, T) or (3,) for scalar time
        if pos.ndim == 1:
            pos = pos[:, np.newaxis]       # (3, 1)
        ecef[:, i, :] = pos.T             # (T, 3)

        sub = wgs84.subpoint_of(geo)
        lat[:, i] = np.atleast_1d(sub.latitude.degrees)
        lon[:, i] = np.atleast_1d(sub.longitude.degrees)

    return Ephemeris(ecef=ecef, subpoint_lat=lat, subpoint_lon=lon)
