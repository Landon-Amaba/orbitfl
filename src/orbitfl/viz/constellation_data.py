"""Build the constellation state that the figures draw.

This is the only module in viz/ that deals with orbital mechanics, and it does
so only through orbitfl.geometry:
  build_walker_delta / build_meo_tier      satellite objects
  propagate                                positions and sub-satellite points
  leo_meo_assignment / isl_pairs           topology
  rivalry_radius_km / footprint_radius_km  radii

The figure modules and check_constellation.py all use the single
ConstellationSnapshot built here, so the checks apply to the objects that are
actually plotted.

Positions are in GCRS (inertial); see viz/earth.py.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from skyfield.api import load

from orbitfl.config import Config, load_config
from orbitfl.geometry.constants import MU_KM3_S2 as _MU_KM3_S2
from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM
from orbitfl.geometry.footprint import footprint_radius_km
from orbitfl.geometry.linkbudget import rivalry_radius_km
from orbitfl.geometry.orbits import (
    Constellation,
    build_meo_tier,
    build_walker_delta,
    propagate,
)
from orbitfl.geometry.visibility import isl_pairs, leo_meo_assignment
from orbitfl.viz.earth import itrf_to_gcrs_rotation

_TS = load.timescale(builtin=True)

# Parameters of the thesis constellation. The figure is meant to show this
# design, so build_snapshot compares the config against these values and raises
# on a mismatch instead of drawing a different constellation.
FROZEN_DESIGN: dict[str, float | int] = {
    "leo.n_planes": 12,
    "leo.sats_per_plane": 15,
    "leo.alt_km": 550.0,
    "leo.inc_deg": 53.0,
    "leo.phasing_f": 1,
    "meo.n": 4,
    "meo.period_h": 5.8,
    "meo.inc_deg": 45.0,
}


@dataclasses.dataclass(frozen=True)
class ConstellationSnapshot:
    """Everything the constellation figures draw and the checks test.

    Array axes: N_leo = n_planes * sats_per_plane (180 in the thesis), N_meo (4),
    K_* = samples over one orbital period, M = samples of the trailing ground track.
    """

    # --- source ---
    config_path: Path
    override_path: Path | None
    epoch_iso: str
    epoch: object                  # skyfield.timelib.Time, scalar

    # --- parameters read from the config ---
    n_planes: int
    sats_per_plane: int
    leo_alt_km: float
    leo_inc_deg: float
    phasing_f: int
    n_meo: int
    meo_period_h: float
    meo_inc_deg: float
    isl_max_range_km: float
    elevation_mask_deg: float
    beam_half_cone_deg: float

    # --- parameters read from the built satellite objects ---
    # (build_meo_tier sets mean anomaly 0 for every MEO)
    meo_mean_anom_deg: tuple[float, ...]       # (N_meo,) degrees
    meo_raan_deg: tuple[float, ...]            # (N_meo,) degrees
    leo_raan_deg: tuple[float, ...]            # (N_leo,) degrees
    leo_period_s: float
    meo_period_s: float
    meo_sma_km: float

    # --- objects ---
    constellation: Constellation
    rot_itrf_to_gcrs: np.ndarray               # (3, 3) ITRF -> GCRS at epoch

    # --- state at the epoch ---
    leo_pos: np.ndarray                        # (N_leo, 3) km GCRS
    meo_pos: np.ndarray                        # (N_meo, 3) km GCRS
    leo_lat: np.ndarray                        # (N_leo,) degrees
    leo_lon: np.ndarray                        # (N_leo,) degrees
    meo_lat: np.ndarray                        # (N_meo,) degrees
    meo_lon: np.ndarray                        # (N_meo,) degrees
    assignment: np.ndarray                     # (N_leo,) int, MEO index or -1
    isl_edges: tuple[tuple[int, int], ...]     # (E,) pairs (i, j) with i < j

    # --- one full orbital period ---
    leo_orbit_pos: np.ndarray                  # (K_leo, N_leo, 3) km GCRS
    leo_orbit_lat: np.ndarray                  # (K_leo, N_leo) degrees
    leo_orbit_lon: np.ndarray                  # (K_leo, N_leo) degrees
    meo_orbit_pos: np.ndarray                  # (K_meo, N_meo, 3) km GCRS
    meo_orbit_lat: np.ndarray                  # (K_meo, N_meo) degrees
    meo_orbit_lon: np.ndarray                  # (K_meo, N_meo) degrees

    # --- trailing ground track, ending at the epoch ---
    trail_minutes: float
    trail_leo_lat: np.ndarray                  # (M, N_leo) degrees
    trail_leo_lon: np.ndarray                  # (M, N_leo) degrees
    trail_meo_lat: np.ndarray                  # (M, N_meo) degrees
    trail_meo_lon: np.ndarray                  # (M, N_meo) degrees

    # --- radii ---
    rivalry_radius_km: float
    footprint_radius_km: float

    @property
    def n_leo(self) -> int:
        """Number of LEO satellites (180 in the thesis constellation)."""
        return len(self.constellation.leos)

    @property
    def n_sats(self) -> int:
        """Total number of satellites (184 in the thesis constellation)."""
        return self.n_leo + len(self.constellation.meos)

    @property
    def plane_id(self) -> np.ndarray:
        """(N_leo,) int, orbital plane index of each LEO."""
        return self.constellation.plane_id

    @property
    def in_plane_idx(self) -> np.ndarray:
        """(N_leo,) int, seat index of each LEO within its plane."""
        return self.constellation.in_plane_idx


# --- Helpers ---


def parse_epoch(epoch_iso: str):
    """ISO-8601 UTC string to a scalar skyfield Time.

    Same convention as scripts/make_trace.py, so figures and traces share an epoch.

    Parameters
    ----------
    epoch_iso : str  e.g. "2026-09-01T00:00:00Z"

    Returns
    -------
    skyfield.timelib.Time  scalar time
    """
    dt = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00"))
    return _TS.from_datetime(dt)


def assert_frozen_design(cfg: Config) -> None:
    """Raise if the config does not describe the thesis constellation.

    Parameters
    ----------
    cfg : Config  loaded configuration

    Raises
    ------
    ValueError  Listing every parameter that differs from FROZEN_DESIGN.
    """
    leo, meo = cfg.constellation.leo, cfg.constellation.meo
    actual: dict[str, float | int] = {
        "leo.n_planes": leo.n_planes,
        "leo.sats_per_plane": leo.sats_per_plane,
        "leo.alt_km": leo.alt_km,
        "leo.inc_deg": leo.inc_deg,
        "leo.phasing_f": leo.phasing_f,
        "meo.n": meo.n,
        "meo.period_h": meo.period_h,
        "meo.inc_deg": meo.inc_deg,
    }
    bad = [
        f"  {k}: config={actual[k]!r}  expected={v!r}"
        for k, v in FROZEN_DESIGN.items()
        if actual[k] != v
    ]
    if bad:
        raise ValueError(
            "configs/base.yaml does not match the thesis constellation:\n"
            + "\n".join(bad)
            + "\nRefusing to draw a figure that would misrepresent the design. Resolve the "
              "discrepancy (and, if the constellation really changed, re-freeze it and bump "
              "TRACE_VERSION) before regenerating figures."
        )


def _circular_period_s(alt_km: float) -> float:
    """Keplerian period in seconds of a circular orbit at alt_km above the surface.

    Used only to decide how long a track to draw and how many samples to take; the
    plotted positions come from propagate.

    Parameters
    ----------
    alt_km : float  altitude above the WGS-84 equatorial radius in km

    Returns
    -------
    float  period in seconds
    """
    a_km = _R_EARTH_KM + alt_km
    return 2.0 * math.pi * math.sqrt(a_km**3 / _MU_KM3_S2)


def _satrec_period_s(sat) -> float:
    """
    Orbital period of a built EarthSatellite, from its SGP4 mean motion.

    Parameters
    ----------
    sat : skyfield.sgp4lib.EarthSatellite

    Returns
    -------
    float  period in seconds
    """
    return 2.0 * math.pi / sat.model.no_kozai * 60.0


def _times_from(epoch, offsets_s: np.ndarray):
    """
    Build a skyfield Time vector at second offsets from a scalar epoch.

    Parameters
    ----------
    epoch     : skyfield.timelib.Time  scalar
    offsets_s : np.ndarray, (K,)  offsets in seconds (may be negative)

    Returns
    -------
    skyfield.timelib.Time  vector of length K
    """
    return _TS.tt_jd(float(epoch.tt) + np.asarray(offsets_s, dtype=float) / 86400.0)


# --- Public API ---


def build_snapshot(
    config_path: Path,
    override_path: Path | None = None,
    epoch_iso: str | None = None,
    orbit_samples: int = 241,
    trail_minutes: float = 20.0,
    trail_samples: int = 41,
    check_frozen: bool = True,
) -> ConstellationSnapshot:
    """Build and propagate the constellation with the simulator's geometry code.

    Parameters
    ----------
    config_path   : Path   base config YAML (configs/base.yaml)
    override_path : Path | None  optional override YAML, merged over the base
    epoch_iso     : str | None   snapshot epoch; defaults to sim.epoch
    orbit_samples : int    samples per orbital period for the track lines and for
                           the maximum-latitude check
    trail_minutes : float  length of the trailing ground track before the epoch
    trail_samples : int    samples in the trailing ground track
    check_frozen  : bool   if True, raise when the config differs from
                           FROZEN_DESIGN

    Returns
    -------
    ConstellationSnapshot
    """
    cfg = load_config(config_path, override_path=override_path)
    if check_frozen:
        assert_frozen_design(cfg)

    leo_cfg = cfg.constellation.leo
    meo_cfg = cfg.constellation.meo
    epoch_iso = epoch_iso or cfg.sim.epoch
    epoch = parse_epoch(epoch_iso)

    # --- satellites ---
    leos = build_walker_delta(
        n_planes=leo_cfg.n_planes,
        sats_per_plane=leo_cfg.sats_per_plane,
        alt_km=leo_cfg.alt_km,
        inc_deg=leo_cfg.inc_deg,
        phasing_f=leo_cfg.phasing_f,
        epoch=epoch,
    )
    meos = build_meo_tier(
        n=meo_cfg.n,
        period_hours=meo_cfg.period_h,
        inc_deg=meo_cfg.inc_deg,
        epoch=epoch,
    )

    # plane_id / in_plane_idx: built as in scripts/make_trace.py
    plane_id = np.repeat(
        np.arange(leo_cfg.n_planes, dtype=np.int32), leo_cfg.sats_per_plane
    )                                                   # (N_leo,)
    in_plane_idx = np.tile(
        np.arange(leo_cfg.sats_per_plane, dtype=np.int32), leo_cfg.n_planes
    )                                                   # (N_leo,)
    constellation = Constellation(
        leos=leos, meos=meos, plane_id=plane_id, in_plane_idx=in_plane_idx
    )

    # --- state at the epoch ---
    eph_leo = propagate(leos, epoch)                    # ecef (1, N_leo, 3)
    eph_meo = propagate(meos, epoch)                    # ecef (1, N_meo, 3)
    leo_pos = eph_leo.ecef[0]                           # (N_leo, 3) km GCRS
    meo_pos = eph_meo.ecef[0]                           # (N_meo, 3) km GCRS

    # Topology functions take (T, N, 3) arrays; here T = 1.
    assignment = leo_meo_assignment(
        eph_leo.ecef, eph_meo.ecef
    )[0]                                                # (N_leo,) int, -1 = blind
    isl_edges = tuple(
        isl_pairs(eph_leo.ecef, cfg.constellation.isl_max_range_km)[0]
    )                                                   # (E,) (i, j), i < j

    # --- one full orbital period ---
    leo_period_s = _satrec_period_s(leos[0])
    meo_period_s = _satrec_period_s(meos[0])
    t_leo = _times_from(epoch, np.linspace(0.0, leo_period_s, orbit_samples))
    t_meo = _times_from(epoch, np.linspace(0.0, meo_period_s, orbit_samples))
    orbit_leo = propagate(leos, t_leo)                  # (K_leo, N_leo, ...)
    orbit_meo = propagate(meos, t_meo)                  # (K_meo, N_meo, ...)

    # --- trailing ground track, ending at the epoch ---
    t_trail = _times_from(
        epoch, np.linspace(-trail_minutes * 60.0, 0.0, trail_samples)
    )
    trail_leo = propagate(leos, t_trail)                # (M, N_leo, ...)
    trail_meo = propagate(meos, t_trail)                # (M, N_meo, ...)

    return ConstellationSnapshot(
        config_path=Path(config_path),
        override_path=Path(override_path) if override_path else None,
        epoch_iso=epoch_iso,
        epoch=epoch,
        n_planes=leo_cfg.n_planes,
        sats_per_plane=leo_cfg.sats_per_plane,
        leo_alt_km=leo_cfg.alt_km,
        leo_inc_deg=leo_cfg.inc_deg,
        phasing_f=leo_cfg.phasing_f,
        n_meo=meo_cfg.n,
        meo_period_h=meo_cfg.period_h,
        meo_inc_deg=meo_cfg.inc_deg,
        isl_max_range_km=cfg.constellation.isl_max_range_km,
        elevation_mask_deg=cfg.allocation.elevation_mask_deg,
        beam_half_cone_deg=cfg.world.beam_half_cone_deg,
        meo_mean_anom_deg=tuple(math.degrees(s.model.mo) % 360.0 for s in meos),
        meo_raan_deg=tuple(math.degrees(s.model.nodeo) % 360.0 for s in meos),
        leo_raan_deg=tuple(math.degrees(s.model.nodeo) % 360.0 for s in leos),
        leo_period_s=leo_period_s,
        meo_period_s=meo_period_s,
        meo_sma_km=(_MU_KM3_S2 * (meo_period_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0),
        constellation=constellation,
        rot_itrf_to_gcrs=itrf_to_gcrs_rotation(epoch),
        leo_pos=leo_pos,
        meo_pos=meo_pos,
        leo_lat=eph_leo.subpoint_lat[0],
        leo_lon=eph_leo.subpoint_lon[0],
        meo_lat=eph_meo.subpoint_lat[0],
        meo_lon=eph_meo.subpoint_lon[0],
        assignment=assignment,
        isl_edges=isl_edges,
        leo_orbit_pos=orbit_leo.ecef,
        leo_orbit_lat=orbit_leo.subpoint_lat,
        leo_orbit_lon=orbit_leo.subpoint_lon,
        meo_orbit_pos=orbit_meo.ecef,
        meo_orbit_lat=orbit_meo.subpoint_lat,
        meo_orbit_lon=orbit_meo.subpoint_lon,
        trail_minutes=trail_minutes,
        trail_leo_lat=trail_leo.subpoint_lat,
        trail_leo_lon=trail_leo.subpoint_lon,
        trail_meo_lat=trail_meo.subpoint_lat,
        trail_meo_lon=trail_meo.subpoint_lon,
        rivalry_radius_km=rivalry_radius_km(
            leo_cfg.alt_km, cfg.allocation.elevation_mask_deg
        ),
        footprint_radius_km=footprint_radius_km(
            leo_cfg.alt_km, cfg.world.beam_half_cone_deg
        ),
    )


def propagate_window(
    snap: ConstellationSnapshot, offsets_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the snapshot's satellites at arbitrary offsets from its epoch.

    Used by the animation, which needs positions at times the snapshot does not
    hold.

    Parameters
    ----------
    snap      : ConstellationSnapshot
    offsets_s : np.ndarray, (F,)  seconds relative to snap.epoch

    Returns
    -------
    leo_pos : np.ndarray, (F, N_leo, 3) km GCRS
    meo_pos : np.ndarray, (F, N_meo, 3) km GCRS
    """
    times = _times_from(snap.epoch, offsets_s)
    return (
        propagate(snap.constellation.leos, times).ecef,
        propagate(snap.constellation.meos, times).ecef,
    )
