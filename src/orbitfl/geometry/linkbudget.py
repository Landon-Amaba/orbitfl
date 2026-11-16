"""Free-space path loss, link budget and the pairwise interference rule.

The link-budget constants below (carrier frequency, EIRP, receive gain, noise
temperature, bandwidth) are illustrative placeholder values. They have not been
checked against a published link budget and should be replaced before any
quantitative use of the SNR or INR figures. In the thesis experiments they only
enter through the interference graph, whose topology is set by the elevation
mask rather than by these values (see is_interference_edge).

Functions
---------
fspl_db                free-space path loss
snr_db                 received SNR from a Friis link budget
achievable_rate_bps    Shannon capacity
elevation_from_ground  elevation of a satellite above a ground point's horizon
rivalry_radius_km      ground distance at which a neighbour sits on the elevation mask
interference_db        co-channel INR at one satellite's footprint from another
is_interference_edge   the edge rule of the interference graph
"""

from __future__ import annotations

import math

import numpy as np

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM

# Placeholder link-budget constants (not taken from a reference; see module docstring).

FREQ_GHZ: float = 20.0  # carrier frequency, GHz
EIRP_DBW: float = 34.0  # transmit EIRP, dBW
G_RX_DBI: float = 34.0  # receive antenna gain, dBi
NOISE_TEMP_K: float = 300.0  # system noise temperature, K
BW_HZ: float = 250e6  # channel bandwidth, Hz

# Boltzmann constant (exact SI value)
_K_BOLTZMANN: float = 1.380649e-23   # J K^-1


def fspl_db(d_km: float, f_ghz: float) -> float:
    """
    Free-space path loss in dB.

    FSPL = 20*log10(4pi*d*f / c)
         = 20*log10(d_km) + 20*log10(f_ghz) + 92.44

    The constant 92.44 dB absorbs unit conversions (km->m, GHz->Hz) and
    4pi/c.  Valid for d_km > 0 and f_ghz > 0.

    Parameters
    ----------
    d_km  : float  slant distance in km
    f_ghz : float  carrier frequency in GHz

    Returns
    -------
    float  FSPL in dB
    """
    return 92.44 + 20.0 * math.log10(d_km) + 20.0 * math.log10(f_ghz)


def snr_db(
    eirp_dbw: float,
    g_rx_dbi: float,
    fspl: float,
    losses_db: float,
    noise_temp_k: float,
    bw_hz: float,
) -> float:
    """
    Received SNR in dB using the Friis link budget.

    SNR = EIRP + G_rx - FSPL - losses - N0
    where N0 = 10*log10(k_B * T * B) is the noise power spectral density
    integrated over bandwidth B.

    Parameters
    ----------
    eirp_dbw    : float  transmitter EIRP in dBW
    g_rx_dbi    : float  receive antenna gain in dBi
    fspl        : float  free-space path loss in dB (from fspl_db)
    losses_db   : float  additional link losses in dB (pointing, atmosphere, ...)
    noise_temp_k: float  system noise temperature in K
    bw_hz       : float  noise bandwidth in Hz

    Returns
    -------
    float  SNR in dB
    """
    noise_power_dbw = (
        10.0 * math.log10(_K_BOLTZMANN)
        + 10.0 * math.log10(noise_temp_k)
        + 10.0 * math.log10(bw_hz)
    )
    return eirp_dbw + g_rx_dbi - fspl - losses_db - noise_power_dbw


def achievable_rate_bps(snr_db_val: float, bw_hz: float) -> float:
    """
    Shannon channel capacity in bits per second.

    C = B * log2(1 + SNR)

    Parameters
    ----------
    snr_db_val : float  SNR in dB
    bw_hz      : float  bandwidth in Hz

    Returns
    -------
    float  capacity in bps
    """
    snr_linear = 10.0 ** (snr_db_val / 10.0)
    return bw_hz * math.log2(1.0 + snr_linear)


def elevation_from_ground(g_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """
    Elevation angle in degrees above the local horizon at ground point g_ecef.

    The local zenith at g_ecef is the outward radial direction (g_ecef normalised).
    Elevation = arcsin( dot(unit_vec_to_sat, local_zenith) ).

    Parameters
    ----------
    g_ecef   : np.ndarray (3,)  ground point in ECEF km
    sat_ecef : np.ndarray (3,)  satellite position in ECEF km

    Returns
    -------
    float  elevation in degrees; negative means below horizon
    """
    g_hat = np.asarray(g_ecef, dtype=float)
    g_hat = g_hat / float(np.linalg.norm(g_hat))         # local zenith unit vector
    vec = np.asarray(sat_ecef, dtype=float) - np.asarray(g_ecef, dtype=float)
    d = float(np.linalg.norm(vec))
    if d < 1e-3:
        return 90.0                                        # satellite directly overhead
    sin_el = float(np.dot(vec / d, g_hat))
    sin_el = max(-1.0, min(1.0, sin_el))                   # clamp for float safety
    return math.degrees(math.asin(sin_el))


def rivalry_radius_km(alt_km: float, elevation_mask_deg: float) -> float:
    """Ground-arc distance at which a co-altitude neighbour sits exactly on the mask.

    Two satellites at the same altitude are separated by this surface distance when
    the elevation of one, seen from the other's footprint centre, equals
    elevation_mask_deg. The central angle phi satisfies

        cos(phi + theta) = R * cos(theta) / (R + h),    D = R * phi

    Reference values for h = 550 km, R = 6378.137 km:
        mask  0 deg: 2557 km    mask 10 deg: 1664 km    mask 15 deg: 1360 km
        mask 20 deg: 1125 km    mask 25 deg:  941 km

    Parameters
    ----------
    alt_km             : float  satellite altitude in km
    elevation_mask_deg : float  elevation mask in degrees

    Returns
    -------
    float  surface arc distance in km
    """
    theta = math.radians(elevation_mask_deg)
    cos_arg = (_R_EARTH_KM * math.cos(theta)) / (_R_EARTH_KM + alt_km)
    cos_arg = max(-1.0, min(1.0, cos_arg))   # clamp for float safety
    phi = math.acos(cos_arg) - theta          # central angle in radians
    return _R_EARTH_KM * phi


def interference_db(sat_i: int, sat_j: int, geometry: dict) -> float:
    """
    Co-channel interference-to-noise ratio (INR) at sat_i's footprint from sat_j.

    Models sat_j as an uncoordinated co-channel transmitter. The INR is the
    received power from sat_j at the centre of sat_i's footprint (on the ground),
    divided by the noise floor.

    Returns float('-inf') when Earth occludes the path (sat_j is below sat_i's
    horizon), so is_interference_edge correctly returns False in that case.

    Assumes full boresight gain on both ends and zero additional losses (losses_db=0).
    This is a worst-case / optimistic-for-interference simplification; link margins
    are not the intent here, only graph topology.

    geometry dict keys (all required):
      'ecef' : np.ndarray (N, 3) km  - ECEF positions of all satellites
      'lat'  : np.ndarray (N,)       - sub-satellite latitudes in degrees
      'lon'  : np.ndarray (N,)       - sub-satellite longitudes in degrees

    Parameters
    ----------
    sat_i    : int   index of the victim satellite
    sat_j    : int   index of the interfering satellite
    geometry : dict  (see above)

    Returns
    -------
    float  INR in dB at sat_i's footprint centre due to sat_j's transmission,
           or float('-inf') if Earth blocks the path
    """
    # Footprint centre of sat_i, obtained by scaling its position vector down to the
    # Earth's surface. This keeps g_i in the same Cartesian frame as geometry["ecef"];
    # building it from lat/lon instead would mix an Earth-fixed point with positions
    # in the propagator's inertial frame and give wrong elevations.
    p_i = np.asarray(geometry["ecef"][sat_i], dtype=float)
    g_i = p_i * (_R_EARTH_KM / float(np.linalg.norm(p_i)))   # (3,) km, surface nadir

    p_j = np.asarray(geometry["ecef"][sat_j], dtype=float)   # (3,) km

    # sat_j must be above the local horizon of g_i, i.e. g_hat_i . (p_j - g_i) > 0.
    # Below the horizon the Earth blocks the path, so return -inf.
    g_i_hat = g_i / float(np.linalg.norm(g_i))
    if float(np.dot(g_i_hat, p_j - g_i)) <= 0.0:
        return float("-inf")

    d_km = float(np.linalg.norm(p_j - g_i))
    if d_km < 1e-3:
        d_km = 1e-3  # collocated satellites

    fspl = fspl_db(d_km, FREQ_GHZ)
    received_power_dbw = EIRP_DBW + G_RX_DBI - fspl

    noise_power_dbw = (
        10.0 * math.log10(_K_BOLTZMANN)
        + 10.0 * math.log10(NOISE_TEMP_K)
        + 10.0 * math.log10(BW_HZ)
    )

    return received_power_dbw - noise_power_dbw


def is_interference_edge(
    sat_i: int,
    sat_j: int,
    geometry: dict,
    threshold_db: float,
    elevation_mask_deg: float,
) -> bool:
    """Edge rule of the interference graph.

    True iff sat_j is at or above elevation_mask_deg as seen from sat_i's footprint
    centre and the co-channel INR at that footprint exceeds threshold_db. The mask
    is tested first; pairs below it return False without computing the INR.

    With the 12 deg mask and -6 dB threshold used in the thesis, the INR threshold
    never rejects a pair that passes the mask: at the largest above-horizon slant
    range the INR is still about 6.7 dB above the threshold. The graph topology is
    therefore set by the mask alone.

    The footprint centre g_i is derived from satellite i's own position vector, so
    both operands stay in the frame of geometry["ecef"].

    Parameters
    ----------
    sat_i              : int    victim satellite index
    sat_j              : int    interferer satellite index
    geometry           : dict   see interference_db
    threshold_db       : float  INR threshold (allocation.threshold_db)
    elevation_mask_deg : float  minimum elevation in degrees (allocation.elevation_mask_deg)

    Returns
    -------
    bool
    """
    # Footprint centre of sat_i in the frame of geometry["ecef"] (see interference_db).
    p_i_vec = np.asarray(geometry["ecef"][sat_i], dtype=float)
    g_i = p_i_vec * (_R_EARTH_KM / float(np.linalg.norm(p_i_vec)))
    p_j = np.asarray(geometry["ecef"][sat_j], dtype=float)  # (3,) km

    if elevation_from_ground(g_i, p_j) < elevation_mask_deg:
        return False
    return interference_db(sat_i, sat_j, geometry) >= threshold_db
