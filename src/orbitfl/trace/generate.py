"""Trace generation.

TraceGenerator(cfg, constellation).run(out_path) propagates the constellation,
computes visibility, LEO-to-MEO assignment and link rates, and writes the four
parquet tables described in schema.py.

Satellite ids are the same in every table:
  LEO k  ->  sat_id = k             (k in [0, N_leo))
  MEO k  ->  sat_id = N_leo + k     (k in [0, N_meo))
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from skyfield.api import load

from orbitfl.config import Config, to_dict
from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM
from orbitfl.geometry.linkbudget import BW_HZ, EIRP_DBW, FREQ_GHZ, G_RX_DBI, NOISE_TEMP_K
from orbitfl.geometry.orbits import Constellation, Ephemeris, propagate
from orbitfl.geometry.visibility import isl_pairs, leo_meo_assignment, los_visible
from orbitfl.trace.schema import (
    SCHEMA_ASSIGNMENT,
    SCHEMA_META,
    SCHEMA_POSITIONS,
    SCHEMA_VISIBILITY,
    TRACE_VERSION,
)

# Boltzmann constant (exact SI value)
_K_BOLTZMANN: float = 1.380649e-23  # J K^-1

_TS = load.timescale(builtin=True)


class TraceGenerator:
    """
    Propagates the constellation and writes all four trace parquet files.

    Parameters
    ----------
    cfg          : Config        Full simulation configuration.
    constellation: Constellation Pre-built LEO + MEO EarthSatellite objects.
    """

    def __init__(self, cfg: Config, constellation: Constellation) -> None:
        self._cfg = cfg
        self._constellation = constellation

    def run(self, out_path: Path) -> None:
        """
        Generate the full trace and write four parquet files to out_path.

        Files produced:
          positions.parquet   (T*(N_leo+N_meo), 11) satellite positions at every step
          visibility.parquet  (n_vis, 6)            all visible links with Shannon rates
          assignment.parquet  (T*N_leo, 3)          LEO->MEO assignments per timestep
          meta.parquet        (1, 7)                run provenance

        Parameters
        ----------
        out_path : Path  Directory to write trace files; created if absent.
        """
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        cfg = self._cfg
        con = self._constellation
        N_leo: int = len(con.leos)
        N_meo: int = len(con.meos)

        # --- Build time vector ---
        sim = cfg.sim
        T: int = sim.duration_h * 3600 // sim.timestep_s  # number of timesteps

        epoch_dt = datetime.fromisoformat(sim.epoch.replace("Z", "+00:00"))
        epoch_sf = _TS.from_datetime(epoch_dt)

        # TT Julian day vector: epoch_sf.tt is scalar; arange offsets in days
        jd_step: float = sim.timestep_s / 86400.0
        jd_vector: np.ndarray = float(epoch_sf.tt) + np.arange(T) * jd_step  # (T,)
        times_sf = _TS.tt_jd(jd_vector)

        # numpy UTC timestamps for the positions table (Unix seconds as int64)
        epoch_unix: float = epoch_dt.timestamp()
        unix_times: np.ndarray = (epoch_unix + np.arange(T) * sim.timestep_s).astype(
            np.int64
        )  # (T,) seconds since 1970-01-01T00:00:00Z

        # --- Propagation ---
        leo_eph: Ephemeris = propagate(con.leos, times_sf)  # ecef (T, N_leo, 3)
        meo_eph: Ephemeris = propagate(con.meos, times_sf)  # ecef (T, N_meo, 3)

        # --- Assignment: (T, N_leo) int, MEO index or -1 ---
        asgn_matrix: np.ndarray = leo_meo_assignment(
            leo_eph.ecef, meo_eph.ecef
        )  # (T, N_leo)

        # --- Write tables ---
        pq.write_table(
            _build_positions(leo_eph, meo_eph, con, T, N_leo, N_meo, unix_times),
            out_path / "positions.parquet",
        )
        pq.write_table(
            _build_assignment(asgn_matrix, T, N_leo),
            out_path / "assignment.parquet",
        )
        pq.write_table(
            _build_visibility(leo_eph, meo_eph, cfg, T, N_leo, N_meo),
            out_path / "visibility.parquet",
        )
        pq.write_table(
            _build_meta(cfg, T),
            out_path / "meta.parquet",
        )


# --- Table builders ---


def _build_positions(
    leo_eph: Ephemeris,
    meo_eph: Ephemeris,
    con: Constellation,
    T: int,
    N_leo: int,
    N_meo: int,
    unix_times: np.ndarray,  # (T,) int64 Unix seconds
) -> pa.Table:
    """
    Construct the positions pyarrow Table.

    Row count: T * (N_leo + N_meo).
    sat_id scheme: LEO k -> k, MEO k -> N_leo + k.
    alt_km is the geocentric norm minus R_EARTH_KM (adequate for circular orbits).
    """
    # t_idx column - repeated per satellite in each group
    t_idx_leo: np.ndarray = np.repeat(np.arange(T, dtype=np.int32), N_leo)  # (T*N_leo,)
    t_idx_meo: np.ndarray = np.repeat(np.arange(T, dtype=np.int32), N_meo)  # (T*N_meo,)

    # Unix timestamp column - same shape as t_idx (int64 seconds)
    time_utc_leo: np.ndarray = np.repeat(unix_times, N_leo)   # (T*N_leo,)
    time_utc_meo: np.ndarray = np.repeat(unix_times, N_meo)   # (T*N_meo,)

    # sat_id
    sat_id_leo: np.ndarray = np.tile(np.arange(N_leo, dtype=np.int32), T)          # (T*N_leo,)
    sat_id_meo: np.ndarray = np.tile(
        np.arange(N_leo, N_leo + N_meo, dtype=np.int32), T
    )  # (T*N_meo,)

    # ECEF coords: (T, N, 3) -> (T*N, 3)
    leo_xyz: np.ndarray = leo_eph.ecef.reshape(-1, 3)   # (T*N_leo, 3) km
    meo_xyz: np.ndarray = meo_eph.ecef.reshape(-1, 3)   # (T*N_meo, 3) km

    # Altitude: geocentric norm - R_EARTH_KM, valid for near-circular orbits
    leo_alt: np.ndarray = np.linalg.norm(leo_eph.ecef, axis=-1).ravel() - _R_EARTH_KM  # (T*N_leo,)
    meo_alt: np.ndarray = np.linalg.norm(meo_eph.ecef, axis=-1).ravel() - _R_EARTH_KM  # (T*N_meo,)

    # plane_id: LEO from constellation, MEO = -1
    plane_id_leo: np.ndarray = np.tile(con.plane_id.astype(np.int32), T)   # (T*N_leo,)
    plane_id_meo: np.ndarray = np.full(T * N_meo, -1, dtype=np.int32)      # (T*N_meo,)

    # Concatenate LEO then MEO rows
    all_t = np.concatenate([t_idx_leo, t_idx_meo])                              # (T*(N_leo+N_meo),)
    all_time = np.concatenate([time_utc_leo, time_utc_meo])                     # int64 Unix s
    all_sat_id = np.concatenate([sat_id_leo, sat_id_meo])
    all_kind = (["leo"] * (T * N_leo)) + (["meo"] * (T * N_meo))
    all_xyz = np.concatenate([leo_xyz, meo_xyz], axis=0)                        # (T*(N+M), 3)
    all_lat = np.concatenate([leo_eph.subpoint_lat.ravel(), meo_eph.subpoint_lat.ravel()])
    all_lon = np.concatenate([leo_eph.subpoint_lon.ravel(), meo_eph.subpoint_lon.ravel()])
    all_alt = np.concatenate([leo_alt, meo_alt])
    all_plane = np.concatenate([plane_id_leo, plane_id_meo])

    return pa.table(
        {
            "t_idx": pa.array(all_t, type=pa.int32()),
            "time_utc": pa.array(all_time, type=pa.timestamp("s", tz="UTC")),
            "sat_id": pa.array(all_sat_id, type=pa.int32()),
            "kind": pa.array(all_kind, type=pa.string()),
            "x": pa.array(all_xyz[:, 0], type=pa.float64()),
            "y": pa.array(all_xyz[:, 1], type=pa.float64()),
            "z": pa.array(all_xyz[:, 2], type=pa.float64()),
            "lat": pa.array(all_lat, type=pa.float64()),
            "lon": pa.array(all_lon, type=pa.float64()),
            "alt_km": pa.array(all_alt, type=pa.float64()),
            "plane_id": pa.array(all_plane, type=pa.int32()),
        },
        schema=SCHEMA_POSITIONS,
    )


def _build_assignment(
    asgn_matrix: np.ndarray,  # (T, N_leo) int
    T: int,
    N_leo: int,
) -> pa.Table:
    """
    Construct the assignment pyarrow Table.

    Row count: T * N_leo (one row per (timestep, LEO) pair).
    meo_id is the MEO's sat_id = N_leo + k, or -1 for unassigned.
    """
    t_idx: np.ndarray = np.repeat(np.arange(T, dtype=np.int32), N_leo)   # (T*N_leo,)
    leo_id: np.ndarray = np.tile(np.arange(N_leo, dtype=np.int32), T)    # (T*N_leo,)
    meo_id: np.ndarray = asgn_matrix.ravel().astype(np.int32)             # (T*N_leo,)

    return pa.table(
        {
            "t_idx": pa.array(t_idx, type=pa.int32()),
            "leo_id": pa.array(leo_id, type=pa.int32()),
            "meo_id": pa.array(meo_id, type=pa.int32()),
        },
        schema=SCHEMA_ASSIGNMENT,
    )


def _build_visibility(
    leo_eph: Ephemeris,
    meo_eph: Ephemeris,
    cfg: Config,
    T: int,
    N_leo: int,
    N_meo: int,
) -> pa.Table:
    """Construct the visibility pyarrow Table (leo_meo rows, then ISL rows).

    Link rates use the placeholder link-budget constants from linkbudget.py.
    Visibility here is line of sight only (80 km grazing altitude); the elevation
    mask of the interference graph is not applied.

    Row count: n_visible_leo_meo_pairs + n_isl_pairs (both vary by timestep).
    """
    # Noise power, shared by every link
    noise_dbw: float = 10.0 * math.log10(_K_BOLTZMANN * NOISE_TEMP_K * BW_HZ)
    fspl_freq_term: float = 92.44 + 20.0 * math.log10(FREQ_GHZ)   # constant part of FSPL (dB)
    snr_base: float = EIRP_DBW + G_RX_DBI - 0.0 - noise_dbw       # FSPL not yet subtracted

    # --- LEO-MEO visibility ---
    # Broadcast to (T, N_leo, N_meo, 3) for all-pairs computation
    leo_b: np.ndarray = leo_eph.ecef[:, :, np.newaxis, :]   # (T, N_leo, 1,     3) km
    meo_b: np.ndarray = meo_eph.ecef[:, np.newaxis, :, :]   # (T, 1,     N_meo, 3) km

    vis_lm: np.ndarray = los_visible(leo_b, meo_b)           # (T, N_leo, N_meo) bool
    diff_lm: np.ndarray = leo_b - meo_b                       # (T, N_leo, N_meo, 3) km
    dist_lm: np.ndarray = np.sqrt(np.sum(diff_lm**2, axis=-1))  # (T, N_leo, N_meo) km

    safe_dist_lm: np.ndarray = np.where(dist_lm > 0.0, dist_lm, 1e-3)
    fspl_lm: np.ndarray = fspl_freq_term + 20.0 * np.log10(safe_dist_lm)   # (T, N_leo, N_meo) dB
    rate_lm: np.ndarray = BW_HZ * np.log2(
        1.0 + 10.0 ** ((snr_base - fspl_lm) / 10.0)
    )  # (T, N_leo, N_meo) bps

    # Extract visible (t, leo, meo) triples
    t_idx_lm, leo_idx_lm, meo_idx_lm = np.where(vis_lm)     # each: (n_vis_lm,)
    a_lm: np.ndarray = leo_idx_lm.astype(np.int32)           # leo sat_id = leo index
    b_lm: np.ndarray = (N_leo + meo_idx_lm).astype(np.int32) # meo sat_id = N_leo + meo index
    dist_lm_vis: np.ndarray = dist_lm[t_idx_lm, leo_idx_lm, meo_idx_lm]
    rate_lm_vis: np.ndarray = rate_lm[t_idx_lm, leo_idx_lm, meo_idx_lm]

    # --- ISL visibility ---
    isls: list = isl_pairs(leo_eph.ecef, cfg.constellation.isl_max_range_km)

    isl_t_list: list = []
    isl_a_list: list = []
    isl_b_list: list = []

    for t_i, pairs in enumerate(isls):
        for (i, j) in pairs:
            isl_t_list.append(t_i)
            isl_a_list.append(i)
            isl_b_list.append(j)

    if isl_t_list:
        isl_t_np = np.array(isl_t_list, dtype=np.int32)    # (n_isl,)
        isl_a_np = np.array(isl_a_list, dtype=np.int32)    # (n_isl,) - leo sat_id
        isl_b_np = np.array(isl_b_list, dtype=np.int32)    # (n_isl,) - leo sat_id (j > i)

        # Distances by fancy indexing
        pos_a: np.ndarray = leo_eph.ecef[isl_t_np, isl_a_np]  # (n_isl, 3) km
        pos_b: np.ndarray = leo_eph.ecef[isl_t_np, isl_b_np]  # (n_isl, 3) km
        isl_dist: np.ndarray = np.linalg.norm(pos_a - pos_b, axis=-1)  # (n_isl,) km

        safe_isl_dist = np.maximum(isl_dist, 1e-3)
        fspl_isl: np.ndarray = fspl_freq_term + 20.0 * np.log10(safe_isl_dist)
        rate_isl: np.ndarray = BW_HZ * np.log2(
            1.0 + 10.0 ** ((snr_base - fspl_isl) / 10.0)
        )  # (n_isl,) bps

        # Merge leo_meo and ISL rows
        t_idx_all = np.concatenate(
            [t_idx_lm.astype(np.int32), isl_t_np]
        )  # (n_vis_lm + n_isl,)
        a_all = np.concatenate([a_lm, isl_a_np])
        b_all = np.concatenate([b_lm, isl_b_np])
        dist_all = np.concatenate([dist_lm_vis, isl_dist])
        rate_all = np.concatenate([rate_lm_vis, rate_isl])
        kind_all = (["leo_meo"] * len(t_idx_lm)) + (["isl"] * len(isl_t_list))
    else:
        t_idx_all = t_idx_lm.astype(np.int32)
        a_all = a_lm
        b_all = b_lm
        dist_all = dist_lm_vis
        rate_all = rate_lm_vis
        kind_all = ["leo_meo"] * len(t_idx_lm)

    return pa.table(
        {
            "t_idx": pa.array(t_idx_all, type=pa.int32()),
            "a": pa.array(a_all, type=pa.int32()),
            "b": pa.array(b_all, type=pa.int32()),
            "kind": pa.array(kind_all, type=pa.string()),
            "distance_km": pa.array(dist_all, type=pa.float64()),
            "rate_bps": pa.array(rate_all, type=pa.float64()),
        },
        schema=SCHEMA_VISIBILITY,
    )


def _build_meta(cfg: Config, T: int) -> pa.Table:
    """Construct the meta pyarrow Table (single row)."""
    cfg_yaml = yaml.dump(to_dict(cfg), default_flow_style=False, sort_keys=True)
    config_hash: str = hashlib.sha256(cfg_yaml.encode()).hexdigest()[:16]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_hash: str = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "nogit"

    sim = cfg.sim
    duration_s: int = sim.duration_h * 3600

    return pa.table(
        {
            "trace_version": pa.array([TRACE_VERSION], type=pa.string()),
            "config_hash": pa.array([config_hash], type=pa.string()),
            "git_hash": pa.array([git_hash], type=pa.string()),
            "timestep_s": pa.array([sim.timestep_s], type=pa.int32()),
            "duration_s": pa.array([duration_s], type=pa.int64()),
            "epoch": pa.array([sim.epoch], type=pa.string()),
            "seed": pa.array([cfg.seed], type=pa.int32()),
        },
        schema=SCHEMA_META,
    )
