"""Geographic region profiles for the occupancy world model.

RegionProfile holds one region's bounding box and its per-channel occupancy and
dwell times. build_region_map converts a WorldConfig into a list of profiles,
and region_at maps a (lat, lon) point to a region index. region_at is the single
place where region boundaries are evaluated, so the occupancy model and the
region-based clustering strategy always agree on them. A point that falls in no
box is assigned to the last region, which acts as a catch-all.

The module also holds the two world knobs studied in the thesis:
contrast_dwell_times (predictability contrast) and region_polarity /
concept_shift_channel_mask (hidden concept shift).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from orbitfl.config import WorldConfig


@dataclasses.dataclass(frozen=True)
class RegionProfile:
    """Per-region radio-environment profile. Arrays broadcast to shape (C,)."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    occupancy: np.ndarray   # (C,) stationary per-channel occupancy in [0, 1]
    mean_on_s: np.ndarray   # (C,) mean busy-dwell seconds
    mean_off_s: np.ndarray  # (C,) mean idle-dwell seconds

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegionProfile):
            return NotImplemented
        return (
            self.name == other.name
            and self.lat_min == other.lat_min
            and self.lat_max == other.lat_max
            and self.lon_min == other.lon_min
            and self.lon_max == other.lon_max
            and np.array_equal(self.occupancy, other.occupancy)
            and np.array_equal(self.mean_on_s, other.mean_on_s)
            and np.array_equal(self.mean_off_s, other.mean_off_s)
        )

    def __hash__(self) -> int:
        return hash((
            self.name, self.lat_min, self.lat_max, self.lon_min, self.lon_max,
        ))


RegionMap = list[RegionProfile]

# At contrast = 1 the per-region p_sum multipliers span [0.1, 1.9], so the lag-4
# autocorrelation r4 = (1 - p_sum)^4 ranges from about 0.75 down to near 0 across
# the regions.
_CONTRAST_MAX_SPREAD: float = 0.9


def contrast_dwell_times(
    mean_on_s: np.ndarray,
    mean_off_s: np.ndarray,
    dt: float,
    region_idx: int,
    n_regions: int,
    contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale dwell times to spread per-region predictability, holding occupancy fixed.

    Predictability is governed by p_sum = p_on + p_off, with r4 = (1 - p_sum)^4.
    The stationary occupancy pi = p_off / p_sum does not change when both
    probabilities are scaled by the same factor. Region `region_idx` gets a p_sum
    multiplier that ramps linearly with its rank in [0, n_regions - 1]: region 0 is
    the most predictable (smallest p_sum) and the last region the least. The spread
    is proportional to `contrast`. Dwell times are then rebuilt from the scaled
    probabilities. contrast <= 0, or a single region, returns the inputs unchanged.

    Parameters
    ----------
    mean_on_s  : (C,) float  Busy-dwell seconds per channel.
    mean_off_s : (C,) float  Idle-dwell seconds per channel.
    dt         : float       Markov step duration in seconds.
    region_idx : int         Region index (0-based).
    n_regions  : int         Total number of regions.
    contrast   : float       Contrast in [0, 1]; 0 = no change.

    Returns
    -------
    (mean_on_s', mean_off_s') : tuple of (C,) float  Rescaled dwell times.
    """
    if contrast <= 0.0 or n_regions <= 1:
        return mean_on_s, mean_off_s

    p_on = dt / mean_on_s      # (C,) P(busy -> idle)
    p_off = dt / mean_off_s    # (C,) P(idle -> busy)
    p_sum = p_on + p_off       # (C,) lag-1 flip probability; sets r4
    pi = p_off / p_sum         # (C,) stationary occupancy (preserved)

    rank = region_idx / (n_regions - 1)  # 0 (most predictable) .. 1 (least)
    mult = 1.0 + contrast * (2.0 * rank - 1.0) * _CONTRAST_MAX_SPREAD
    new_p_sum = np.clip(p_sum * mult, 1e-6, 0.999)  # keep a valid transition probability

    new_p_off = pi * new_p_sum
    new_p_on = (1.0 - pi) * new_p_sum
    return dt / new_p_on, dt / new_p_off


def region_polarity(region_idx: int) -> int:
    """Concept-shift polarity of a region: +1 for even rank, -1 for odd rank.

    Splits the region map into two groups with opposite targets. The even/odd rule
    is arbitrary but fixed and uses no random numbers. With a four-region map it
    gives two regions per polarity.

    Parameters
    ----------
    region_idx : int  Region rank (0-based) in the region map.

    Returns
    -------
    int  +1 (target unchanged) or -1 (target inverted on the conflicting channels).
    """
    return 1 if region_idx % 2 == 0 else -1


def concept_shift_channel_mask(n_channels: int, concept_shift: float) -> np.ndarray:
    """Boolean mask of the channels affected by the concept shift.

    The first round(concept_shift * n_channels) channels are marked True. On those
    channels the delta-ahead forecast target is complemented in regions of polarity
    -1 (see region_polarity), while the observation window is left untouched. More
    channels conflict as concept_shift grows. The flip is deterministic, so it adds
    no label noise. concept_shift <= 0 returns an all-False mask.

    Parameters
    ----------
    n_channels    : int    Number of spectrum channels C.
    concept_shift : float  Fraction of channels in conflict, in [0, 1]; 0 = none.

    Returns
    -------
    np.ndarray  (C,) bool  True where the target is subject to polarity inversion.
    """
    mask = np.zeros(n_channels, dtype=bool)  # (C,) bool
    if concept_shift <= 0.0:
        return mask
    k = int(round(min(concept_shift, 1.0) * n_channels))
    mask[:k] = True
    return mask


def _broadcast(values: list[float], n_channels: int) -> np.ndarray:
    """Broadcast a 1-element or C-element list to shape (C,) float64."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (1,):
        return np.full(n_channels, float(arr[0]))
    if arr.shape == (n_channels,):
        return arr.copy()
    raise ValueError(
        f"Expected 1 or {n_channels} values; got shape {arr.shape}"
    )


def build_region_map(cfg: WorldConfig) -> RegionMap:
    """
    Convert a WorldConfig into a list of RegionProfile objects.

    Single-element occupancy and dwell-time lists are broadcast to all
    cfg.n_channels channels so downstream code can always index by channel.

    Parameters
    ----------
    cfg : WorldConfig

    Returns
    -------
    RegionMap  same ordering as cfg.regions; last entry used as catch-all
    """
    C = cfg.n_channels
    profiles: RegionMap = []
    for r in cfg.regions:
        profiles.append(RegionProfile(
            name=r.name,
            lat_min=r.lat_min,
            lat_max=r.lat_max,
            lon_min=r.lon_min,
            lon_max=r.lon_max,
            occupancy=_broadcast(r.occupancy, C),
            mean_on_s=_broadcast(r.mean_on_s, C),
            mean_off_s=_broadcast(r.mean_off_s, C),
        ))
    return profiles


def region_at(lat: float, lon: float, region_map: RegionMap) -> int:
    """
    Index of the first RegionProfile whose bounding box contains (lat, lon).

    Boxes are half-open: [lat_min, lat_max) x [lon_min, lon_max).
    Falls back to len(region_map) - 1 if no box matches (global catch-all).

    Parameters
    ----------
    lat        : float  sub-satellite latitude  in degrees (-90 ... 90)
    lon        : float  sub-satellite longitude in degrees (-180 ... 180)
    region_map : RegionMap

    Returns
    -------
    int  index into region_map
    """
    for idx, r in enumerate(region_map):
        if r.lat_min <= lat < r.lat_max and r.lon_min <= lon < r.lon_max:
            return idx
    return len(region_map) - 1
