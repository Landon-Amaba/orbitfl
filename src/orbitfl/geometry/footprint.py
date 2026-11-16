"""Satellite footprint geometry on the Earth's surface.

Functions
---------
footprint_radius_km   radius of a nadir-pointing beam's footprint
footprint_center      footprint centre (the sub-satellite point)
footprints_overlap    whether two circular footprints overlap
"""

from __future__ import annotations

import math

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM


def footprint_radius_km(alt_km: float, half_cone_deg: float) -> float:
    """
    Footprint radius in km for a nadir-pointing beam.

    The beam subtends half-cone angle theta from nadir at the satellite. Using the
    spherical law of sines in the triangle (Earth-centre O, satellite S, footprint
    edge P) with |OS| = R + h, |OP| = R, angle at S = theta:

        rho = arcsin((R + h) * sin(theta) / R) - theta   [radians, Earth central angle]

    which gives arc-length radius R * rho on the surface.

    Parameters
    ----------
    alt_km        : float  satellite altitude above the surface in km
    half_cone_deg : float  beam half-cone angle from nadir in degrees

    Returns
    -------
    float  footprint radius in km (arc-length)
    """
    R = _R_EARTH_KM
    a = R + alt_km
    theta = math.radians(half_cone_deg)

    # Argument clamped to [-1, 1] to guard against floating-point rounding
    # when the beam just grazes Earth's limb (a*sin(theta)/R -> 1).
    arg = min(a * math.sin(theta) / R, 1.0)
    rho = math.asin(arg) - theta      # Earth central angle, radians >= 0

    return R * rho                    # arc-length in km


def footprint_center(
    subpoint_lat: float, subpoint_lon: float
) -> tuple[float, float]:
    """
    Footprint centre for a nadir-pointing beam.

    For a nadir-pointing antenna the footprint centre coincides with the
    sub-satellite point.

    Parameters
    ----------
    subpoint_lat : float  sub-satellite latitude in degrees
    subpoint_lon : float  sub-satellite longitude in degrees

    Returns
    -------
    tuple[float, float]  (latitude_deg, longitude_deg)
    """
    return (subpoint_lat, subpoint_lon)


def footprints_overlap(
    c1: tuple[float, float],
    r1: float,
    c2: tuple[float, float],
    r2: float,
) -> bool:
    """
    True iff the two circular footprints overlap.

    Footprints are modelled as spherical caps. They overlap when the
    great-circle distance between their centres is less than the sum of their
    radii (in km of arc-length).  Uses the haversine formula for numerical
    stability at small angular separations.

    Parameters
    ----------
    c1 : tuple[float, float]  (lat_deg, lon_deg) centre of footprint 1
    r1 : float                radius of footprint 1 in km
    c2 : tuple[float, float]  (lat_deg, lon_deg) centre of footprint 2
    r2 : float                radius of footprint 2 in km

    Returns
    -------
    bool  True iff great-circle distance(c1, c2) < r1 + r2
    """
    lat1 = math.radians(c1[0])
    lon1 = math.radians(c1[1])
    lat2 = math.radians(c2[0])
    lon2 = math.radians(c2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    gc_dist_km = _R_EARTH_KM * 2 * math.asin(math.sqrt(min(a, 1.0)))

    return gc_dist_km < r1 + r2
