"""Tests for src/orbitfl/geometry/visibility.py.

V1  los_visible is symmetric: los_visible(a, b) == los_visible(b, a).
V2  Satellites on opposite sides of the Earth at the same altitude are never visible.
V3  Identical or nearly identical points are always visible to each other.
V4  leo_meo_assignment returns -1 at exactly the timesteps with no MEO in view,
    checked on a synthetic fixture.
V5  isl_pairs lists each pair once and respects max_range_km.
V6  elevation_deg is 90 deg directly overhead and about 0 deg at the horizon.
"""

from __future__ import annotations

import math

import numpy as np

from orbitfl.geometry.visibility import (
    _R_EARTH_KM,
    elevation_deg,
    isl_pairs,
    leo_meo_assignment,
    los_visible,
)

# Shared constants matching test_orbits.py fixtures
_LEO_ALT_KM: float = 550.0
_MEO_ALT_KM: float = 10_035.0   # ~ period 5.8 h semi-major axis - R_earth
_R_LEO: float = _R_EARTH_KM + _LEO_ALT_KM
_R_MEO: float = _R_EARTH_KM + _MEO_ALT_KM


# --- V1 - symmetry of los_visible ---

class TestLosVisibleSymmetry:
    def test_symmetric_two_leos_same_side(self):
        """Two LEOs on the same hemisphere: visibility is symmetric."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = np.array([0.0, _R_LEO, 0.0])
        assert bool(los_visible(a, b)) == bool(los_visible(b, a))

    def test_symmetric_leo_meo(self):
        """LEO-MEO pair: order of arguments does not matter."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = np.array([_R_MEO, 0.0, 0.0])
        assert bool(los_visible(a, b)) == bool(los_visible(b, a))

    def test_symmetric_blocked_pair(self):
        """Blocked pair (antipodal) is symmetric in its blocking."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = np.array([-_R_LEO, 0.0, 0.0])
        assert bool(los_visible(a, b)) == bool(los_visible(b, a))

    def test_symmetric_batch(self):
        """Batch of random pairs: swap gives identical result element-wise."""
        rng = np.random.default_rng(0)
        N = 20
        p1 = rng.standard_normal((N, 3))
        scale = _R_LEO + rng.uniform(0, 500, (N, 1))
        p1 = p1 / np.linalg.norm(p1, axis=-1, keepdims=True) * scale
        p2 = rng.standard_normal((N, 3))
        scale2 = _R_LEO + rng.uniform(0, 500, (N, 1))
        p2 = p2 / np.linalg.norm(p2, axis=-1, keepdims=True) * scale2
        np.testing.assert_array_equal(los_visible(p1, p2), los_visible(p2, p1))


# --- V2 - antipodal satellites are never visible to each other ---

class TestAntipodal:
    def test_exact_antipodal_leo(self):
        """Satellites at (R_leo, 0, 0) and (-R_leo, 0, 0) are blocked by Earth."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = -a
        assert not los_visible(a, b)

    def test_exact_antipodal_meo(self):
        """Same test for MEO altitude - still blocked."""
        a = np.array([0.0, _R_MEO, 0.0])
        b = -a
        assert not los_visible(a, b)

    def test_antipodal_batch(self):
        """Any satellite and its antipode are mutually blocked."""
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((10, 3))
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True) * _R_LEO
        assert not np.any(los_visible(pts, -pts))

    def test_near_antipodal_still_blocked(self):
        """Slightly perturbed antipodal points remain blocked."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = np.array([-_R_LEO + 10.0, 0.0, 0.0])   # 10 km off exact antipode
        assert not los_visible(a, b)


# --- V3 - co-located points are always visible ---

class TestColocated:
    def test_identical_point(self):
        """A point can always see itself."""
        a = np.array([_R_LEO, 100.0, 200.0])
        assert los_visible(a, a)

    def test_near_identical_points(self):
        """Points 1 m apart at LEO altitude are visible."""
        a = np.array([_R_LEO, 0.0, 0.0])
        b = a + np.array([0.001, 0.0, 0.0])   # 1 m separation
        assert los_visible(a, b)

    def test_batch_identical(self):
        """Batch of self-pairs: all visible."""
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((15, 3))
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True) * _R_LEO
        assert np.all(los_visible(pts, pts))


# --- V4: leo_meo_assignment returns -1 exactly where no MEO is in view ---

class TestLeoMeoAssignment:
    """
    Synthetic fixture (no real TLEs needed):
      T=2, N_leo=2, N_meo=1.
      MEO fixed on +x axis at MEO radius.

      t=0:  LEO-0 at (+x, R_leo) - visible to MEO.
            LEO-1 at (-x, R_leo) - Earth blocks line to MEO.
      t=1:  LEO-0 at (+x, R_leo) - visible.
            LEO-1 at (+x, R_leo) - visible (same x-side as MEO).

    Expected assignment matrix:
      result[0, 0] = 0   (LEO-0 sees MEO-0 at t=0)
      result[0, 1] = -1  (LEO-1 blind at t=0)
      result[1, 0] = 0   (LEO-0 sees MEO-0 at t=1)
      result[1, 1] = 0   (LEO-1 sees MEO-0 at t=1)
    """

    def _fixture(self):
        leo = np.array([
            # t=0
            [[_R_LEO,  0.0, 0.0],    # LEO-0: +x, visible
             [-_R_LEO, 0.0, 0.0]],   # LEO-1: -x, blocked
            # t=1
            [[_R_LEO,  0.0, 0.0],    # LEO-0: +x, visible
             [_R_LEO,  0.0, 0.0]],   # LEO-1: +x, visible
        ], dtype=float)               # (2, 2, 3)

        meo = np.array([
            [[_R_MEO, 0.0, 0.0]],    # t=0
            [[_R_MEO, 0.0, 0.0]],    # t=1
        ], dtype=float)               # (2, 1, 3)

        return leo, meo

    def test_blind_slot_is_minus_one(self):
        leo, meo = self._fixture()
        result = leo_meo_assignment(leo, meo)
        assert result[0, 1] == -1, f"Expected -1 for blind LEO-1 at t=0, got {result[0,1]}"

    def test_visible_slots_get_correct_index(self):
        leo, meo = self._fixture()
        result = leo_meo_assignment(leo, meo)
        assert result[0, 0] == 0, f"Expected 0 for LEO-0 at t=0, got {result[0,0]}"
        assert result[1, 0] == 0, f"Expected 0 for LEO-0 at t=1, got {result[1,0]}"
        assert result[1, 1] == 0, f"Expected 0 for LEO-1 at t=1, got {result[1,1]}"

    def test_no_spill_to_adjacent_timestep(self):
        """The -1 at (t=0, LEO-1) must not contaminate (t=1, LEO-1)."""
        leo, meo = self._fixture()
        result = leo_meo_assignment(leo, meo)
        assert result[1, 1] != -1, "Blind-spot -1 spilled into adjacent visible timestep"

    def test_output_shape(self):
        leo, meo = self._fixture()
        result = leo_meo_assignment(leo, meo)
        assert result.shape == (2, 2), f"Expected (2,2), got {result.shape}"
        assert result.dtype == int or np.issubdtype(result.dtype, np.integer)

    def test_nearest_when_multiple_meos(self):
        """With two MEOs, each LEO is assigned the closer visible one."""
        R_close = _R_LEO + 200.0   # closer MEO (still in space)
        R_far = _R_MEO

        leo = np.array([[[_R_LEO, 0.0, 0.0]]], dtype=float)  # (1, 1, 3)
        meo = np.array([[[R_far, 0.0, 0.0],
                          [R_close, 0.0, 0.0]]], dtype=float)  # (1, 2, 3)

        result = leo_meo_assignment(leo, meo)
        # MEO-1 (index 1) is closer; both visible -> expect index 1
        assert result[0, 0] == 1, f"Expected MEO-1 (closer), got {result[0,0]}"

    def test_all_meos_blocked_returns_minus_one(self):
        """LEO with all MEOs on opposite side -> all -1."""
        leo = np.array([[[_R_LEO, 0.0, 0.0]]], dtype=float)   # (1, 1, 3)
        meo = np.array([[[-_R_MEO, 0.0, 0.0],
                          [-_R_MEO, 100.0, 0.0]]], dtype=float)  # (1, 2, 3)

        result = leo_meo_assignment(leo, meo)
        assert result[0, 0] == -1


# --- V5 - isl_pairs: range cap and symmetry ---

class TestIslPairs:
    def test_no_pairs_when_range_zero(self):
        """max_range_km=0 means no pairs (satellites are not coincident)."""
        leo = np.array([[[_R_LEO, 0.0, 0.0],
                          [_R_LEO, 100.0, 0.0]]], dtype=float)
        pairs = isl_pairs(leo, max_range_km=0.0)
        assert pairs[0] == []

    def test_all_pairs_when_range_large(self):
        """
        Nearby LEOs (30 deg separation) see each other; antipodal are blocked by Earth.

        At 550 km altitude the maximum visible separation is ~42.5 deg (chord midpoint
        distance = R_leo*cos(theta/2) must exceed R_earth + 80 km grazing altitude).
        30 deg is within that limit; 180 deg is far outside it.
        """
        r = _R_LEO
        a = math.radians(30)
        leo = np.array([[[r, 0.0, 0.0],
                          [r * math.cos(a), r * math.sin(a), 0.0],  # 30 deg away - visible
                          [-r, 0.0, 0.0]]], dtype=float)             # 180 deg away - blocked
        pairs = isl_pairs(leo, max_range_km=np.inf)
        pair_set = set(pairs[0])
        assert (0, 1) in pair_set,     "30°-separated LEOs should be linked"
        assert (0, 2) not in pair_set, "Antipodal LEOs should be blocked by Earth"
        assert (1, 2) not in pair_set, "150°-separated LEOs should be blocked"

    def test_pairs_respect_range_cap(self):
        """Two close LEOs are linked; a distant third is not."""
        near = np.array([_R_LEO, 0.0, 0.0])
        # 500 km away in the y-direction (same altitude ~ same radius)
        offset = np.array([_R_LEO, 500.0, 0.0])
        offset = offset / np.linalg.norm(offset) * _R_LEO
        # Far: 5000 km away
        far = np.array([0.0, _R_LEO, 0.0])   # 90 deg away ~ 7000 km arc

        leo = np.array([[near, offset, far]], dtype=float)  # (1, 3, 3)
        dist_near_offset = float(np.linalg.norm(near - offset))

        pairs_within = isl_pairs(leo, max_range_km=dist_near_offset + 1.0)
        pairs_outside = isl_pairs(leo, max_range_km=dist_near_offset - 1.0)

        assert (0, 1) in pairs_within[0], "Near pair should be linked"
        assert (0, 1) not in pairs_outside[0], "Near pair outside cap should not link"

    def test_upper_triangle_only(self):
        """All returned pairs (i, j) have i < j - no duplicates."""
        rng = np.random.default_rng(42)
        N = 6
        pts = rng.standard_normal((N, 3))
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True) * _R_LEO
        leo = pts[np.newaxis]   # (1, N, 3)
        pairs = isl_pairs(leo, max_range_km=2000.0)
        for i, j in pairs[0]:
            assert i < j, f"Pair ({i},{j}) violates i < j"


# --- V6 - elevation_deg ---

class TestElevationDeg:
    def test_zenith_is_90(self):
        """Satellite directly above observer -> elevation = 90 deg."""
        ground = np.array([_R_EARTH_KM, 0.0, 0.0])
        sat = np.array([_R_EARTH_KM + 550.0, 0.0, 0.0])
        el = float(elevation_deg(ground, sat))
        assert abs(el - 90.0) < 1e-9, f"Zenith elevation = {el}°, expected 90°"

    def test_horizon_is_near_zero(self):
        """Satellite at the geometric horizon has exactly 0 deg elevation.

        The horizon angle satisfies cos(theta) = R_earth / (R_earth + h).
        At that angle g_hat * r_hat = 0 exactly, so elevation_deg returns 0 deg.
        (90 deg separation is well below the horizon at 550 km - ~ -43 deg.)
        """
        h = 550.0
        r_sat = _R_EARTH_KM + h
        theta_h = math.acos(_R_EARTH_KM / r_sat)       # geometric horizon angle
        ground = np.array([_R_EARTH_KM, 0.0, 0.0])
        sat = np.array([r_sat * math.cos(theta_h), r_sat * math.sin(theta_h), 0.0])
        el = float(elevation_deg(ground, sat))
        assert abs(el) < 1e-6, f"Horizon elevation = {el}°, expected 0°"

    def test_negative_below_horizon(self):
        """Satellite behind Earth from observer's perspective -> negative elevation."""
        ground = np.array([_R_EARTH_KM, 0.0, 0.0])
        sat = np.array([-_R_LEO, 0.0, 0.0])   # directly behind Earth
        el = float(elevation_deg(ground, sat))
        assert el < 0.0, f"Behind-Earth elevation = {el}°, should be negative"

    def test_batch_shape(self):
        """elevation_deg handles (T, 3) inputs and returns (T,) output."""
        T = 5
        ground = np.tile([_R_EARTH_KM, 0.0, 0.0], (T, 1))
        sat = np.tile([_R_LEO, 0.0, 0.0], (T, 1))
        el = elevation_deg(ground, sat)
        assert el.shape == (T,)
        assert np.all(el > 89.0)
