"""Line-of-sight visibility, elevation angles, LEO-to-MEO assignment and ISL pairs.

Functions
---------
los_visible         whether the segment between two points clears the Earth
elevation_deg       elevation of a satellite above a ground point's horizon
leo_meo_assignment  nearest visible MEO for every LEO and timestep (-1 if none)
isl_pairs           LEO pairs within range and line of sight, per timestep
"""

from __future__ import annotations

import numpy as np

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM


def los_visible(
    p1_ecef,
    p2_ecef,
    grazing_alt_km: float = 80.0,
) -> np.ndarray:
    """
    Line-of-sight test: True if the segment p1->p2 clears Earth.

    The segment is P(t) = p1 + t*(p2-p1), t in [0,1]. Its closest approach to
    the Earth centre (origin) is found analytically by differentiating ||P(t)||^2,
    clamping t to [0,1], and comparing the resulting distance to R_earth + grazing_alt_km.

    Fully broadcastable: p1_ecef and p2_ecef may be any shapes that broadcast
    together, with the last axis being the 3-D spatial dimension.

    Parameters
    ----------
    p1_ecef : array_like, (..., 3) km
    p2_ecef : array_like, (..., 3) km
    grazing_alt_km : float
        Minimum clearance above Earth's surface along the segment.

    Returns
    -------
    np.ndarray, (...) bool
    """
    p1 = np.asarray(p1_ecef, dtype=float)
    p2 = np.asarray(p2_ecef, dtype=float)
    min_r = _R_EARTH_KM + grazing_alt_km

    d = p2 - p1                                           # (..., 3)
    dd = np.sum(d * d, axis=-1)                           # (...) squared length of segment

    # t* = -(p1 * d) / (d * d); guard against degenerate p1==p2 (dd=0) -> t*=0
    p1d = np.sum(p1 * d, axis=-1)                         # (...)
    t = np.where(dd > 0.0, -p1d / np.where(dd > 0.0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)                              # (...)

    closest = p1 + t[..., np.newaxis] * d                 # (..., 3)
    dist_sq = np.sum(closest * closest, axis=-1)          # (...) squared distance to origin

    return dist_sq > min_r * min_r                        # avoids sqrt for speed


def elevation_deg(ground_ecef, sat_ecef) -> np.ndarray:
    """
    Elevation angle (degrees) of a satellite as seen from a ground point.

    Elevation is the angle above the local horizontal plane:
        sin(el) = g_hat * r_hat
    where g_hat = ground_ecef / |ground_ecef| is the local vertical and
    r_hat = (sat - ground) / |sat - ground| is the unit slant vector.

    Parameters
    ----------
    ground_ecef : array_like, (..., 3) km
    sat_ecef    : array_like, (..., 3) km

    Returns
    -------
    np.ndarray, (...) degrees in [-90, 90]
    """
    g = np.asarray(ground_ecef, dtype=float)
    s = np.asarray(sat_ecef, dtype=float)

    g_norm = np.linalg.norm(g, axis=-1, keepdims=True)
    g_hat = g / np.where(g_norm > 0.0, g_norm, 1.0)      # (..., 3)

    r = s - g
    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / np.where(r_norm > 0.0, r_norm, 1.0)      # (..., 3)

    sin_el = np.clip(np.sum(g_hat * r_hat, axis=-1), -1.0, 1.0)
    return np.degrees(np.arcsin(sin_el))


def leo_meo_assignment(leo_ecef: np.ndarray, meo_ecef: np.ndarray) -> np.ndarray:
    """Assign each LEO to its nearest visible MEO at every timestep.

    Uses los_visible with the default grazing altitude (80 km). When no MEO is in
    view the entry is -1. Callers must treat -1 as "no aggregator" and never use
    it as an index; the FL loop skips such clients for that round.

    Parameters
    ----------
    leo_ecef : np.ndarray, (T, N_leo, 3) km  ECEF positions of LEO satellites
    meo_ecef : np.ndarray, (T, N_meo, 3) km  ECEF positions of MEO satellites

    Returns
    -------
    np.ndarray, (T, N_leo) int
        MEO index in [0, N_meo) where a MEO is visible, -1 otherwise.
    """
    leo = np.asarray(leo_ecef, dtype=float)   # (T, N_leo, 3)
    meo = np.asarray(meo_ecef, dtype=float)   # (T, N_meo, 3)

    # Broadcast to (T, N_leo, N_meo) for all-pairs visibility and distance
    leo_b = leo[:, :, np.newaxis, :]          # (T, N_leo, 1,     3)
    meo_b = meo[:, np.newaxis, :, :]          # (T, 1,     N_meo, 3)

    vis = los_visible(leo_b, meo_b)           # (T, N_leo, N_meo) bool

    diff = leo_b - meo_b                      # (T, N_leo, N_meo, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=-1))   # (T, N_leo, N_meo)

    # Mask invisible MEOs with inf so argmin always picks a visible one first
    dist_masked = np.where(vis, dist, np.inf)     # (T, N_leo, N_meo)

    nearest = np.argmin(dist_masked, axis=-1)     # (T, N_leo)  - index in [0, N_meo)
    any_vis = np.any(vis, axis=-1)                # (T, N_leo) bool

    return np.where(any_vis, nearest, -1).astype(int)


def isl_pairs(leo_ecef: np.ndarray, max_range_km: float) -> list:
    """Inter-satellite link pairs for each timestep.

    A pair (i, j) with i < j is included iff the distance between LEO i and LEO j
    is <= max_range_km and los_visible is True (the Earth does not block the link).

    Parameters
    ----------
    leo_ecef     : np.ndarray, (T, N_leo, 3) km
    max_range_km : float  maximum ISL range (constellation.isl_max_range_km)

    Returns
    -------
    list[list[tuple[int, int]]]
        Outer list: one entry per timestep.
        Inner list: (i, j) pairs with i < j, in row-major order.
    """
    leo = np.asarray(leo_ecef, dtype=float)
    T, N_leo = leo.shape[:2]

    result: list = []
    for t in range(T):
        pos = leo[t]                                      # (N_leo, 3)

        # Pairwise vectors and distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N_leo, N_leo, 3)
        dist = np.sqrt(np.sum(diff * diff, axis=-1))          # (N_leo, N_leo)

        # Visibility matrix
        vis = los_visible(
            pos[:, np.newaxis, :],   # (N_leo, 1,     3)
            pos[np.newaxis, :, :],   # (1,     N_leo, 3)
        )                            # (N_leo, N_leo) bool

        # Upper triangle: within range and line-of-sight clear
        mask = np.triu((dist <= max_range_km) & vis, k=1)
        rows, cols = np.where(mask)
        result.append(list(zip(rows.tolist(), cols.tolist())))

    return result
