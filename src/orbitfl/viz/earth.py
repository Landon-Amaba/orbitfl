"""Earth rendering helpers: WGS-84 ellipsoid surface, texture sampling, frames.

Reference frame
---------------
Ephemeris.ecef (orbitfl.geometry.orbits) holds skyfield's
EarthSatellite.at(t).position.km, which is in GCRS, an inertial frame, and not
in an Earth-fixed frame as the field name suggests. The array equals the
Earth-fixed (ITRF) vector rotated about the z axis by the sidereal angle.

The simulator is unaffected: visibility, elevation, assignment, ISL and link
budget code use only distances and angles between these vectors, which do not
change under that rotation, and the sub-satellite latitude and longitude are
computed by skyfield in the correct frame. The frame matters for a figure,
which combines satellite vectors with an Earth texture fixed to Greenwich.

The figures therefore plot satellites in GCRS and rotate the Earth into GCRS
with skyfield's ITRS-to-GCRS rotation at the epoch. Orbit tracks are inertial
and close into rings.

Earth texture
-------------
assets/earth_bluemarble_2048x1024.jpg is NASA Visible Earth "Blue Marble: Next
Generation" (December 2004, topography and bathymetry), Reto Stöckli / NASA
Earth Observatory, downsampled from 5400x2700 to 2048x1024. See
assets/LICENSE.md.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image
from skyfield.framelib import itrs

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM

# WGS-84 flattening (NIMA TR 8350.2, 3rd ed.).
WGS84_FLATTENING: float = 1.0 / 298.257223563

# First eccentricity squared, e^2 = 2f - f^2.
WGS84_E2: float = 2.0 * WGS84_FLATTENING - WGS84_FLATTENING**2

# Semi-minor axis b = a(1 - f) ~ 6356.752 km.
B_EARTH_KM: float = _R_EARTH_KM * (1.0 - WGS84_FLATTENING)

TEXTURE_PATH: Path = Path(__file__).parent / "assets" / "earth_bluemarble_2048x1024.jpg"


@functools.lru_cache(maxsize=4)
def load_texture(max_width: int = 2048) -> np.ndarray:
    """Load the Blue Marble texture as a float RGB array.

    Parameters
    ----------
    max_width : int
        Downsample so the returned width is at most this many pixels; the height
        follows, keeping the 2:1 aspect of the equirectangular map.

    Returns
    -------
    np.ndarray, (H, W, 3) float in [0, 1]
        Row 0 is latitude +90 deg, column 0 is longitude -180 deg.
    """
    img = Image.open(TEXTURE_PATH).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, max_width // 2), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def sample_texture(tex: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lookup of the texture at given geodetic coordinates.

    Parameters
    ----------
    tex     : np.ndarray, (H, W, 3) float  texture from load_texture
    lat_deg : np.ndarray, (...)   geodetic latitude in degrees, [-90, 90]
    lon_deg : np.ndarray, (...)   longitude in degrees, [-180, 180]

    Returns
    -------
    np.ndarray, (..., 3) float in [0, 1]  RGB per input coordinate
    """
    h, w = tex.shape[0], tex.shape[1]
    row = np.clip(((90.0 - lat_deg) / 180.0 * h).astype(int), 0, h - 1)      # (...)
    col = np.clip(((lon_deg + 180.0) / 360.0 * w).astype(int), 0, w - 1)     # (...)
    return tex[row, col]


def geodetic_to_itrf(lat_deg, lon_deg, alt_km=0.0) -> np.ndarray:
    """Geodetic (lat, lon, altitude) to Earth-fixed Cartesian on the WGS-84 ellipsoid.

    Uses the prime-vertical radius of curvature N(phi) = a / sqrt(1 - e^2 sin^2 phi),
    so surface points lie on the ellipsoid rather than on a sphere.

    Parameters
    ----------
    lat_deg : array_like, (...)  geodetic latitude in degrees
    lon_deg : array_like, (...)  longitude in degrees
    alt_km  : array_like, (...) or float  height above the ellipsoid in km

    Returns
    -------
    np.ndarray, (..., 3) km  ITRF Cartesian coordinates
    """
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(alt_km, dtype=float)

    n = _R_EARTH_KM / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)   # (...)
    x = (n + h) * np.cos(phi) * np.cos(lam)                        # (...)
    y = (n + h) * np.cos(phi) * np.sin(lam)                        # (...)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(phi)                   # (...)
    return np.stack([x, y, z], axis=-1)                            # (..., 3)


def itrf_to_gcrs_rotation(epoch) -> np.ndarray:
    """ITRS-to-GCRS rotation matrix at an epoch.

    skyfield's itrs.rotation_at(t) returns the GCRS-to-ITRS matrix; its transpose
    maps an Earth-fixed vector into the inertial frame of Ephemeris.ecef (see the
    module docstring).

    Parameters
    ----------
    epoch : skyfield.timelib.Time  scalar time

    Returns
    -------
    np.ndarray, (3, 3)  R such that v_gcrs = R @ v_itrf
    """
    return np.asarray(itrs.rotation_at(epoch), dtype=float).T   # (3, 3)


def itrf_to_gcrs(v_itrf: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Rotate ITRF vectors into GCRS.

    Parameters
    ----------
    v_itrf : np.ndarray, (..., 3) km  ITRF vectors
    rot    : np.ndarray, (3, 3)       matrix from itrf_to_gcrs_rotation

    Returns
    -------
    np.ndarray, (..., 3) km  the same vectors expressed in GCRS
    """
    return np.einsum("ij,...j->...i", rot, np.asarray(v_itrf, dtype=float))


def globe_surface(
    rot: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 361,
    texture_width: int = 1024,
    brightness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A textured WGS-84 ellipsoid mesh, rotated into GCRS.

    Parameters
    ----------
    rot           : np.ndarray, (3, 3)  ITRF-to-GCRS rotation at the epoch
    n_lat         : int   latitude samples (rows of the mesh)
    n_lon         : int   longitude samples (columns of the mesh)
    texture_width : int   texture width used for the colour lookup
    brightness    : float multiplier applied to the sampled RGB, clipped to
                    [0, 1]; values below 1 darken the globe so that satellite
                    markers stand out.

    Returns
    -------
    x, y, z : np.ndarray, each (n_lat, n_lon) km  GCRS mesh coordinates
    rgb     : np.ndarray, (n_lat, n_lon, 3) float in [0, 1]  per-vertex colour
    """
    lat = np.linspace(90.0, -90.0, n_lat)      # (n_lat,)
    lon = np.linspace(-180.0, 180.0, n_lon)    # (n_lon,)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")   # (n_lat, n_lon) each

    v_itrf = geodetic_to_itrf(lat_g, lon_g, 0.0)          # (n_lat, n_lon, 3)
    v_gcrs = itrf_to_gcrs(v_itrf, rot)                    # (n_lat, n_lon, 3)

    tex = load_texture(texture_width)                     # (H, W, 3)
    rgb = np.clip(sample_texture(tex, lat_g, lon_g) * brightness, 0.0, 1.0)

    return v_gcrs[..., 0], v_gcrs[..., 1], v_gcrs[..., 2], rgb


def texture_colorscale(
    n_colors: int = 255, texture_width: int = 512
) -> tuple[np.ndarray, list[list]]:
    """Quantise the Earth texture into an indexed colourscale for Plotly.

    Plotly's Surface colours vertices through a 1-D colourscale and cannot take an
    RGB image. The texture is reduced to n_colors palette entries; the palette
    becomes a piecewise-constant colourscale and the per-vertex palette index
    becomes surfacecolor.

    Parameters
    ----------
    n_colors      : int  palette size (<= 255)
    texture_width : int  texture width used for quantisation

    Returns
    -------
    indexed    : np.ndarray, (H, W) uint8  palette index per texture pixel,
                 same orientation as load_texture (row 0 = +90 deg lat)
    colorscale : list[list]  Plotly colourscale, 2 stops per palette entry
    """
    img = Image.open(TEXTURE_PATH).convert("RGB")
    if img.width > texture_width:
        img = img.resize((texture_width, texture_width // 2), Image.LANCZOS)

    pal_img = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    indexed = np.asarray(pal_img, dtype=np.uint8)                 # (H, W)
    palette = np.asarray(pal_img.getpalette()[: 3 * n_colors]).reshape(n_colors, 3)

    colorscale: list[list] = []
    for i in range(n_colors):
        rgb = f"rgb({palette[i, 0]},{palette[i, 1]},{palette[i, 2]})"
        colorscale.append([i / n_colors, rgb])
        colorscale.append([(i + 1) / n_colors, rgb])
    return indexed, colorscale


def great_circle_km(
    lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray
) -> np.ndarray:
    """Haversine great-circle distance on a sphere of radius R_EARTH_KM.

    Same convention as geometry.footprint.footprints_overlap, so the footprint
    rings drawn here agree with the overlap test used by the simulator.

    Parameters
    ----------
    lat1_deg, lon1_deg : array_like, (...)  first point(s), degrees
    lat2_deg, lon2_deg : array_like, (...)  second point(s), degrees

    Returns
    -------
    np.ndarray, (...)  arc length in km
    """
    p1 = np.radians(np.asarray(lat1_deg, dtype=float))
    l1 = np.radians(np.asarray(lon1_deg, dtype=float))
    p2 = np.radians(np.asarray(lat2_deg, dtype=float))
    l2 = np.radians(np.asarray(lon2_deg, dtype=float))

    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2.0) ** 2
    return _R_EARTH_KM * 2.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def small_circle_latlon(
    center_lat_deg: float, center_lon_deg: float, radius_km: float, n_points: int = 121
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of a spherical cap boundary (a footprint ring).

    The ring is the set of points at great-circle distance radius_km from the
    centre. It is traced by rotating the centre's unit vector, which is exact on
    the sphere.

    Parameters
    ----------
    center_lat_deg : float  cap centre geodetic latitude, degrees
    center_lon_deg : float  cap centre longitude, degrees
    radius_km      : float  cap radius as surface arc length in km
    n_points       : int    samples around the ring

    Returns
    -------
    lat, lon : np.ndarray, each (n_points,) degrees; lon wrapped to [-180, 180]
    """
    rho = radius_km / _R_EARTH_KM                       # cap angular radius, rad
    phi = np.radians(center_lat_deg)
    lam = np.radians(center_lon_deg)

    # Local orthonormal frame at the cap centre: up (radial), east, north.
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.cross(up, east)                          # (3,)

    az = np.linspace(0.0, 2.0 * np.pi, n_points)        # (n_points,)
    # Point at angular distance rho from `up` along azimuth `az`.
    pts = (
        np.cos(rho) * up[None, :]
        + np.sin(rho) * (np.cos(az)[:, None] * north[None, :] + np.sin(az)[:, None] * east[None, :])
    )                                                   # (n_points, 3)

    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))       # (n_points,)
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))               # (n_points,)
    return lat, lon
