"""Trace loader and accessors.

Trace.load(path) reads a trace directory (four parquet files) and raises
TraceVersionError if its version does not match schema.TRACE_VERSION.

Accessors
---------
positions_at(t_idx)         positions of all satellites at one timestep
subpoints(sat_id)           (T, 2) lat/lon track of one satellite
assignment_at(t_idx)        (N_leo,) MEO sat_id per LEO, or -1
churn_events()              every MEO-to-MEO handover
churn_rate_per_orbit()      mean handovers per satellite per LEO orbit
membership_schedule(times)  (R, N_leo) MEO id per satellite per FL round

Satellite ids follow generate.py:
  LEO k  ->  sat_id = k             (k in [0, N_leo))
  MEO k  ->  sat_id = N_leo + k     (k in [0, N_meo))
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orbitfl.geometry.constants import MU_KM3_S2 as _MU_KM3_S2
from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM
from orbitfl.trace.schema import TRACE_VERSION


class TraceVersionError(ValueError):
    """Raised when a trace directory has an incompatible TRACE_VERSION."""


class Trace:
    """Accessor for a generated trace directory.

    Use Trace.load rather than the constructor. The parquet files are read once at
    load time; the accessors are in-memory pandas / numpy operations.
    """

    # --- Construction ---

    @classmethod
    def load(cls, path: Path) -> "Trace":
        """
        Load a trace directory and verify its TRACE_VERSION.

        Parameters
        ----------
        path : Path  Directory produced by TraceGenerator.run().

        Returns
        -------
        Trace

        Raises
        ------
        TraceVersionError  If the stored trace_version != TRACE_VERSION.
        FileNotFoundError  If path or any expected parquet file is absent.
        """
        path = Path(path)

        meta = pq.read_table(path / "meta.parquet").to_pandas()
        version: str = str(meta["trace_version"].iloc[0])
        if version != TRACE_VERSION:
            raise TraceVersionError(
                f"Trace version mismatch: file has '{version}', "
                f"code expects '{TRACE_VERSION}'. "
                "Bump TRACE_VERSION and re-run make_trace.py to update."
            )

        positions = pq.read_table(path / "positions.parquet").to_pandas()
        visibility = pq.read_table(path / "visibility.parquet").to_pandas()
        assignment = pq.read_table(path / "assignment.parquet").to_pandas()

        trace = cls.__new__(cls)
        trace._path = path
        trace._meta = meta
        trace._positions = positions
        trace._visibility = visibility
        trace._assignment = assignment

        # Precompute the (T, N_leo) assignment matrix for constant-time lookups
        T: int = int(meta["duration_s"].iloc[0]) // int(meta["timestep_s"].iloc[0])
        N_leo: int = int(assignment["leo_id"].max()) + 1
        asgn_matrix = np.full((T, N_leo), -1, dtype=np.int32)
        asgn_matrix[
            assignment["t_idx"].to_numpy(),
            assignment["leo_id"].to_numpy(),
        ] = assignment["meo_id"].to_numpy()
        trace._asgn_matrix = asgn_matrix  # (T, N_leo) int

        return trace

    # --- Accessors ---

    def positions_at(self, t_idx: int) -> pd.DataFrame:
        """
        All satellite positions (LEO + MEO) at a single timestep.

        Parameters
        ----------
        t_idx : int  Timestep index in [0, T).

        Returns
        -------
        pd.DataFrame  Subset of the positions table with the schema from schema.py.
        """
        return self._positions[self._positions["t_idx"] == t_idx].reset_index(drop=True)

    def subpoints(self, sat_id: int) -> np.ndarray:
        """Sub-satellite lat/lon track for one satellite.

        OccupancyWorld uses this track to decide which ground region the satellite
        observes at each timestep.

        Parameters
        ----------
        sat_id : int  Satellite id (LEO: 0..N_leo-1; MEO: N_leo..N_leo+N_meo-1).

        Returns
        -------
        np.ndarray, (T, 2)  Columns: [lat_deg, lon_deg], one row per timestep.
        """
        rows = (
            self._positions[self._positions["sat_id"] == sat_id]
            .sort_values("t_idx")[["lat", "lon"]]
            .to_numpy()
        )  # (T, 2)
        return rows

    def assignment_at(self, t_idx: int) -> np.ndarray:
        """
        MEO assignment for every LEO at a single timestep.

        Parameters
        ----------
        t_idx : int  Timestep index in [0, T).

        Returns
        -------
        np.ndarray, (N_leo,) int  MEO sat_id or -1 for unassigned LEOs.
        """
        return self._asgn_matrix[t_idx]  # (N_leo,)

    def churn_events(self) -> pd.DataFrame:
        """Every MEO-to-MEO handover in the trace.

        A handover is a timestep where a LEO's meo_id changes from one valid MEO
        (meo_id >= 0) to a different valid MEO. Transitions into or out of a coverage
        gap (meo_id = -1) are not counted.

        Returns
        -------
        pd.DataFrame  Columns: leo_id, t_idx, meo_id_from, meo_id_to.
        """
        df = (
            self._assignment.sort_values(["leo_id", "t_idx"])
            .copy()
            .reset_index(drop=True)
        )
        df["meo_id_from"] = df.groupby("leo_id")["meo_id"].shift(1)

        handovers = df[
            (df["meo_id_from"] >= 0)
            & (df["meo_id"] >= 0)
            & (df["meo_id"] != df["meo_id_from"])
        ].copy()

        return (
            handovers[["leo_id", "t_idx", "meo_id_from", "meo_id"]]
            .rename(columns={"meo_id": "meo_id_to"})
            .reset_index(drop=True)
        )

    def churn_rate_per_orbit(self) -> float:
        """Mean handovers per satellite per LEO orbital period.

        The LEO period is derived from the mean LEO altitude at t_idx = 0, assuming a
        circular orbit:
            T = 2pi * sqrt(a^3 / mu),   a = R_earth + alt_km

        Returns
        -------
        float  Handovers per satellite per orbit (>= 0).
        """
        # Estimate LEO altitude from the positions at t = 0
        leo_pos_t0 = self._positions[
            (self._positions["t_idx"] == 0) & (self._positions["kind"] == "leo")
        ]
        mean_alt_km: float = float(leo_pos_t0["alt_km"].mean())
        a_km: float = _R_EARTH_KM + mean_alt_km
        leo_period_s: float = 2.0 * math.pi * math.sqrt(a_km**3 / _MU_KM3_S2)

        duration_s: int = int(self._meta["duration_s"].iloc[0])
        N_leo: int = self._asgn_matrix.shape[1]

        # Total orbit-equivalents across all LEOs over the full trace
        total_orbit_equiv: float = N_leo * (duration_s / leo_period_s)

        total_handovers: int = len(self.churn_events())
        return total_handovers / total_orbit_equiv if total_orbit_equiv > 0 else 0.0

    def membership_schedule(self, round_times: np.ndarray) -> np.ndarray:
        """Cluster membership by serving MEO for every FL round.

        The schedule is computed from the trace alone, before any training, so using it
        costs no measurement or communication.

        Parameters
        ----------
        round_times : np.ndarray, (R,) float
            Seconds from trace epoch for each FL round (e.g. r * round_duration_s).

        Returns
        -------
        np.ndarray, (R, N_leo) int
            MEO cluster id (= meo_id) per satellite per round; -1 for unassigned.
        """
        timestep_s: int = int(self._meta["timestep_s"].iloc[0])
        T: int = self._asgn_matrix.shape[0]

        t_indices: np.ndarray = np.clip(
            np.round(np.asarray(round_times) / timestep_s).astype(int),
            0,
            T - 1,
        )  # (R,)
        return self._asgn_matrix[t_indices]  # (R, N_leo)

    # --- Properties ---

    @property
    def meta(self) -> pd.DataFrame:
        """Single-row DataFrame with trace provenance metadata."""
        return self._meta

    @property
    def n_leo(self) -> int:
        """Number of LEO satellites in this trace."""
        return self._asgn_matrix.shape[1]

    @property
    def n_meo(self) -> int:
        """Number of MEO satellites in this trace."""
        meo_rows = self._positions[self._positions["kind"] == "meo"]
        return int(meo_rows["sat_id"].nunique())

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps T in this trace."""
        return self._asgn_matrix.shape[0]

    @property
    def leo_plane_ids(self) -> np.ndarray:
        """Orbital plane id for every LEO satellite.

        Returns
        -------
        np.ndarray  (N_leo,) int, plane_id[k] for LEO k, read from the positions at t = 0.
        """
        t0 = self._positions[
            (self._positions["t_idx"] == 0) & (self._positions["kind"] == "leo")
        ].sort_values("sat_id")
        return t0["plane_id"].to_numpy(dtype=np.int32)
