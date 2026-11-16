"""Tests for src/orbitfl/viz/: the constellation figure and its checks.

Every assertion is made on the ConstellationSnapshot that the figure modules
draw. The expected values are those of the thesis constellation; tolerances
come from check_constellation.TOLERANCES, which explains each one.
"""

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Path3DCollection  # noqa: E402

from orbitfl.config import load_config  # noqa: E402
from orbitfl.geometry.constants import R_EARTH_KM  # noqa: E402
from orbitfl.viz import palettes  # noqa: E402
from orbitfl.viz.check_constellation import TOLERANCES, run_checks  # noqa: E402
from orbitfl.viz.constellation_data import (  # noqa: E402
    FROZEN_DESIGN,
    assert_frozen_design,
    build_snapshot,
)
from orbitfl.viz.constellation_figure import (  # noqa: E402
    FigureOptions,
    draw_3d,
    draw_groundtrack,
    occluded,
    split_lon_wrap,
    view_direction,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
BASE_YAML = CONFIGS_DIR / "base.yaml"


@pytest.fixture(scope="module")
def snap():
    """The constellation snapshot the figures draw, built once per module."""
    return build_snapshot(BASE_YAML)


# --- The full verification table ---


def test_every_verification_check_passes(snap):
    """check_constellation reports no failures for the base config."""
    failures = [
        f"{c.name}: measured {c.measured}, expected {c.expected} ({c.tolerance})"
        for c in run_checks(snap)
        if not c.passed
    ]
    assert not failures, "verification failures:\n  " + "\n  ".join(failures)


# --- Counts and plane structure ---


def test_satellite_counts(snap):
    """Exactly 180 LEOs and 4 MEOs are built."""
    assert snap.n_leo == 180
    assert len(snap.constellation.meos) == 4
    assert snap.n_sats == 184


def test_twelve_planes_of_fifteen(snap):
    """12 distinct orbital planes, 15 satellites each, measured from the RAANs."""
    uniq, counts = np.unique(np.round(np.asarray(snap.leo_raan_deg), 4), return_counts=True)
    assert uniq.size == 12
    assert set(counts.tolist()) == {15}
    # plane_id must agree with the geometric grouping
    for p in range(12):
        assert int((snap.plane_id == p).sum()) == 15


def test_walker_delta_not_star(snap):
    """RAANs are spaced 30 deg over the full 360 deg, not 180 deg (a Walker star)."""
    uniq = np.unique(np.round(np.asarray(snap.leo_raan_deg), 4))
    spacing = np.diff(uniq)
    assert np.allclose(spacing, 30.0, atol=TOLERANCES["raan_deg"])
    assert uniq.max() - uniq.min() == pytest.approx(330.0, abs=TOLERANCES["raan_deg"])


# --- Orbit geometry ---


def test_leo_max_sublatitude_is_inclination(snap):
    """Peak |sub-satellite latitude| over a full orbit is ~53 deg for the LEOs."""
    measured = float(np.abs(snap.leo_orbit_lat).max())
    assert measured == pytest.approx(53.0, abs=TOLERANCES["max_sublat_deg"])


def test_meo_max_sublatitude_is_inclination(snap):
    """Peak |sub-satellite latitude| over a full orbit is ~45 deg for the MEOs."""
    measured = float(np.abs(snap.meo_orbit_lat).max())
    assert measured == pytest.approx(45.0, abs=TOLERANCES["max_sublat_deg"])


def test_leo_altitude(snap):
    """Orbit-mean geocentric radius minus R_e is ~550 km."""
    alt = float(np.linalg.norm(snap.leo_orbit_pos, axis=-1).mean()) - R_EARTH_KM
    assert alt == pytest.approx(550.0, abs=TOLERANCES["leo_alt_km"])


def test_meo_altitude_and_period(snap):
    """MEO altitude is ~10 011 km within 1 km and the period is 5.8 h."""
    alt = float(np.linalg.norm(snap.meo_orbit_pos, axis=-1).mean()) - R_EARTH_KM
    assert alt == pytest.approx(10011.0, abs=TOLERANCES["meo_alt_km"])
    # and consistent with the semi-major axis implied by the configured period
    assert alt == pytest.approx(
        snap.meo_sma_km - R_EARTH_KM, abs=TOLERANCES["meo_alt_km"]
    )
    assert snap.meo_period_s / 3600.0 == pytest.approx(5.8, abs=TOLERANCES["period_h"])


def test_meo_raans_are_90_deg_apart(snap):
    """The four MEO RAANs are 0/90/180/270 deg."""
    raans = np.sort(np.asarray(snap.meo_raan_deg))
    assert np.allclose(raans, [0.0, 90.0, 180.0, 270.0], atol=TOLERANCES["raan_deg"])
    assert np.allclose(np.diff(raans), 90.0, atol=TOLERANCES["raan_deg"])


def test_in_plane_ground_track_separation(snap):
    """Adjacent seats' sub-satellite points are ~2669 km apart, within 1%."""
    from orbitfl.viz.earth import great_circle_km

    seps = []
    for p in range(snap.n_planes):
        idx = np.flatnonzero(snap.plane_id == p)
        order = idx[np.argsort(snap.in_plane_idx[idx])]
        nxt = np.roll(order, -1)
        seps.extend(
            great_circle_km(
                snap.leo_lat[order], snap.leo_lon[order],
                snap.leo_lat[nxt], snap.leo_lon[nxt],
            ).tolist()
        )
    mean_sep = float(np.mean(seps))
    expected = 2.0 * math.pi * R_EARTH_KM / snap.sats_per_plane   # 2671.67 km
    assert mean_sep == pytest.approx(expected, rel=TOLERANCES["ground_sep_frac"])
    # the value quoted in the thesis text, to the same 1% tolerance
    assert mean_sep == pytest.approx(2669.0, rel=TOLERANCES["ground_sep_frac"])


# --- Frame handling (the figure-specific failure mode) ---


def test_globe_orientation_matches_subpoints(snap):
    """
    Each satellite sits over its own plotted sub-satellite point.

    This is the check that fails if the Earth texture is placed with the wrong
    rotation, which is the live risk because Ephemeris.ecef is GCRS, not ITRF.
    """
    from orbitfl.viz.earth import geodetic_to_itrf, itrf_to_gcrs

    sub = itrf_to_gcrs(
        geodetic_to_itrf(snap.leo_lat, snap.leo_lon, 0.0), snap.rot_itrf_to_gcrs
    )
    u = snap.leo_pos / np.linalg.norm(snap.leo_pos, axis=-1, keepdims=True)
    v = sub / np.linalg.norm(sub, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))
    assert float(ang.max()) < TOLERANCES["globe_orientation_deg"]


def test_ephemeris_ecef_field_is_inertial_not_itrf(snap):
    """
    Pin the documented frame finding: `Ephemeris.ecef` is GCRS despite its name.

    The longitude implied by the array differs from the (correct) subpoint
    longitude by the Greenwich apparent sidereal angle, while the norms agree  -
    i.e. a pure rotation about +z. viz/earth.py relies on this; if the
    simulator is ever corrected to emit true ITRF, this test fails and the
    figure's frame handling must be revisited.
    """
    from skyfield.framelib import itrs

    true_itrf = np.array(
        [sat.at(snap.epoch).frame_xyz(itrs).km for sat in snap.constellation.leos]
    )                                                        # (N_leo, 3) km, true ITRF

    # Same length, different direction => the stored array is the ITRF vector
    # rotated, not the ITRF vector.
    assert np.allclose(
        np.linalg.norm(snap.leo_pos, axis=-1),
        np.linalg.norm(true_itrf, axis=-1),
        rtol=1e-9,
    ), "norms differ: the stored array is not a rotation of ITRF"
    assert np.linalg.norm(snap.leo_pos - true_itrf, axis=-1).min() > 1000.0, (
        "Ephemeris.ecef now matches ITRF: the simulator was corrected, so "
        "viz/earth.py's frame handling must be revisited"
    )

    # The rotation is dominated by the Greenwich apparent sidereal angle; the
    # ~0.4 deg residual spread is precession/nutation/polar motion, which is
    # exactly why viz/earth.py uses Skyfield's full ITRS->GCRS matrix rather
    # than a GMST spin.
    offset = (
        np.degrees(np.arctan2(snap.leo_pos[:, 1], snap.leo_pos[:, 0])) - snap.leo_lon
    ) % 360.0
    gast_deg = (snap.epoch.gast * 15.0) % 360.0
    assert offset.max() - offset.min() < 1.0
    assert abs(((offset.mean() - gast_deg + 180.0) % 360.0) - 180.0) < 1.0


# --- Simulator overlays ---


def test_assignment_is_the_simulator_s_own(snap):
    """Assignment values are valid MEO indices or the -1 blind marker."""
    assert snap.assignment.shape == (180,)
    assert np.all((snap.assignment >= -1) & (snap.assignment < 4))


def test_rivalry_radius_matches_the_closed_form(snap):
    """The footprint-ring radius equals the 12 deg elevation-mask arc distance."""
    theta = math.radians(snap.elevation_mask_deg)
    phi = math.acos(R_EARTH_KM * math.cos(theta) / (R_EARTH_KM + snap.leo_alt_km)) - theta
    assert snap.rivalry_radius_km == pytest.approx(
        R_EARTH_KM * phi, abs=TOLERANCES["rivalry_km"]
    )
    assert snap.rivalry_radius_km == pytest.approx(1533.6, abs=0.1)


# --- Design guard ---


def test_frozen_design_guard_accepts_base_config():
    """configs/base.yaml describes the thesis constellation."""
    assert_frozen_design(load_config(BASE_YAML))


@pytest.mark.parametrize("key", sorted(FROZEN_DESIGN))
def test_frozen_design_guard_rejects_each_change(key, tmp_path):
    """Changing any guarded parameter raises instead of drawing a different constellation."""
    import dataclasses

    cfg = load_config(BASE_YAML)
    tier, field = key.split(".")
    current = getattr(getattr(cfg.constellation, tier), field)
    changed = dataclasses.replace(
        getattr(cfg.constellation, tier), **{field: type(current)(current) + type(current)(1)}
    )
    bad_constellation = dataclasses.replace(cfg.constellation, **{tier: changed})
    bad = dataclasses.replace(cfg, constellation=bad_constellation)
    with pytest.raises(ValueError, match="does not match the thesis constellation"):
        assert_frozen_design(bad)


# --- Plotting helpers ---


def test_occlusion_hides_only_the_far_side():
    """A point behind the Earth on the view axis is occluded; in front, not."""
    view = view_direction(0.0, 0.0)                     # camera on +x
    behind = np.array([[-7000.0, 0.0, 0.0]])
    front = np.array([[7000.0, 0.0, 0.0]])
    beside = np.array([[-7000.0, 9000.0, 0.0]])         # far side but off the disc
    assert occluded(behind, view)[0]
    assert not occluded(front, view)[0]
    assert not occluded(beside, view)[0]


def test_lon_wrap_split():
    """A track crossing the antimeridian is split rather than drawn across the map."""
    lon = np.array([170.0, 179.0, -179.0, -170.0])
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    segs = split_lon_wrap(lon, lat)
    assert len(segs) == 2
    assert all(np.all(np.abs(np.diff(s[:, 0])) < 180.0) for s in segs)


def test_palette_refuses_to_cycle_colours():
    """More planes than colour-blind-safe colours is an error, not a silent reuse."""
    assert len(set(palettes.plane_colors(12))) == 12
    with pytest.raises(ValueError):
        palettes.plane_colors(13)


# --- The figures themselves ---


def test_3d_figure_plots_every_satellite(snap):
    """
    The 3-D figure contains exactly one marker per satellite.

    Satellites are split across a "behind the Earth" and an "in front" artist,
    so this sums the offsets of every 3-D scatter in the axes.
    """
    opts = FigureOptions(globe_lon=61, texture_width=256, isl=True, footprints=True)
    fig = draw_3d(snap, opts)
    try:
        ax = fig.axes[0]
        plotted = sum(
            len(c._offsets3d[0])
            for c in ax.collections
            if isinstance(c, Path3DCollection)
        )
        n_blind = int((snap.assignment < 0).sum())
        # LEOs + MEOs, plus one extra marker for each unassigned LEO
        assert plotted == snap.n_leo + len(snap.constellation.meos) + n_blind
    finally:
        plt.close(fig)


def test_groundtrack_figure_plots_every_satellite(snap):
    """The ground-track panel contains exactly one marker per satellite."""
    fig = draw_groundtrack(snap, FigureOptions())
    try:
        ax = fig.axes[0]
        plotted = sum(
            len(c.get_offsets())
            for c in ax.collections
            if isinstance(c, PathCollection)
        )
        n_blind = int((snap.assignment < 0).sum())
        assert plotted == snap.n_leo + len(snap.constellation.meos) + n_blind
        assert ax.get_xlim() == (-180.0, 180.0)
        assert ax.get_ylim() == (-90.0, 90.0)
    finally:
        plt.close(fig)


def test_figure_cli_writes_outputs(snap, tmp_path):
    """End-to-end: the CLI writes PNG, PDF, verification table and caption."""
    from orbitfl.viz.constellation_figure import main

    rc = main([
        "--config", str(BASE_YAML),
        "--outdir", str(tmp_path),
        "--globe-lon", "61",
        "--texture-width", "256",
        "--no-html",
        "--dpi", "80",
    ])
    assert rc == 0
    stamp = "20260901T000000Z"
    for name in (
        f"constellation_3d_{stamp}.png",
        f"constellation_3d_{stamp}.pdf",
        f"constellation_groundtrack_{stamp}.png",
        f"constellation_groundtrack_{stamp}.pdf",
        f"constellation_verification_{stamp}.txt",
        f"constellation_caption_{stamp}.md",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"
    table = (tmp_path / f"constellation_verification_{stamp}.txt").read_text()
    assert "FAIL" not in table
    caption = (tmp_path / f"constellation_caption_{stamp}.md").read_text()
    assert "2026-09-01T00:00:00Z" in caption
    assert "Walker-delta 12/180/1" in caption
