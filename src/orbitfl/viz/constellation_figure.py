"""Figures of the constellation.

Outputs, written to --outdir (default figures/constellation/):
  constellation_3d_<epoch>.png / .pdf           3-D snapshot, Earth to scale
  constellation_groundtrack_<epoch>.png / .pdf  equirectangular sub-satellite map
  constellation_3d_<epoch>.html                 interactive version (Plotly)
  constellation_orbit_<epoch>.gif|mp4           orbit animation (--animate)
  constellation_verification_<epoch>.txt        table from check_constellation
  constellation_caption_<epoch>.md              caption with epoch and parameters

All positions, sub-satellite points, assignments, ISL edges and footprint radii
come from orbitfl.geometry through constellation_data.build_snapshot.

Frame and occlusion
-------------------
Positions are in GCRS (inertial), and the WGS-84 ellipsoid with its texture is
rotated into GCRS at the epoch (viz/earth.py), so orbit tracks close into
rings. Matplotlib's 3-D axes sort whole artists by a single depth value, which
would draw satellites on the far side of the Earth on top of the globe. The
axes are therefore created with computed_zorder=False, and every drawable is
split into a part behind the Earth, drawn faintly below the globe, and a part
in front, using an orthographic occlusion test against a sphere of Earth radius.

Run:
    python -m orbitfl.viz.constellation_figure --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM  # noqa: E402
from orbitfl.viz import earth, palettes  # noqa: E402
from orbitfl.viz.check_constellation import Check, format_table, run_checks  # noqa: E402
from orbitfl.viz.constellation_data import (  # noqa: E402
    ConstellationSnapshot,
    build_snapshot,
    propagate_window,
)

# Figure sizes for a page with a 6.5 in text width; the ground-track map keeps a
# 2:1 aspect.
_FIG3D_IN: tuple[float, float] = (7.0, 6.4)
_GT_WIDTH_IN: float = 7.0

# Deserts and ice in the Blue Marble image are nearly white; dimming the globe
# keeps the pale colours of the plane palette readable.
_GLOBE_BRIGHTNESS_3D: float = 0.82
_GLOBE_BRIGHTNESS_2D: float = 0.55

# Matplotlib leaves a wide margin around the projected cube. A zoom of 1.72
# fills the frame without clipping the MEO orbit rings at the default view.
_CUBE_ZOOM: float = 1.72


@dataclasses.dataclass(frozen=True)
class FigureOptions:
    """Rendering and overlay options; each is also a command-line flag."""

    outdir: Path = Path("figures/constellation")
    prefix: str = "constellation"
    dpi: int = 300
    # overlays
    tracks: bool = True
    assignment: bool = True
    isl: bool = False
    footprints: bool = False
    # 3-D view
    view_elev: float = 24.0
    view_azim: float = -62.0
    globe_lon: int = 541          # longitude samples of the 3-D globe mesh
    texture_width: int = 2048     # texture width used for the 3-D globe colours
    # outputs
    png: bool = True
    pdf: bool = True
    html: bool = True
    html_globe_lon: int = 181  # coarse mesh to keep the HTML file small
    html_cdn: bool = False
    animate: bool = False
    anim_frames: int = 90
    anim_periods: float = 1.0
    anim_fps: int = 15
    anim_format: str = "gif"


# --- Occlusion: what is hidden behind the Earth from the viewpoint ---


def view_direction(elev_deg: float, azim_deg: float) -> np.ndarray:
    """
    Unit vector from the scene centre toward the camera, in Matplotlib's 3-D
    convention (elevation above the x-y plane, azimuth from +x toward +y).

    Parameters
    ----------
    elev_deg : float  elevation of the viewpoint in degrees
    azim_deg : float  azimuth of the viewpoint in degrees

    Returns
    -------
    np.ndarray, (3,)  unit vector
    """
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def occluded(pos: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Orthographic occlusion test against a sphere of Earth radius.

    A point is hidden iff it lies on the far side of the plane through the Earth's
    centre perpendicular to the view axis (component along `view` < 0) and its
    distance from that axis is less than the Earth radius. The Earth is treated as
    a sphere; against the WGS-84 ellipsoid the error is a band of at most 21 km at
    the limb, less than a pixel at the figure size used.

    Parameters
    ----------
    pos  : np.ndarray, (..., 3) km  positions in the plotted (GCRS) frame
    view : np.ndarray, (3,)         unit vector toward the camera

    Returns
    -------
    np.ndarray, (...) bool  True where the point is hidden by the Earth
    """
    p = np.asarray(pos, dtype=float)
    along = p @ view                                      # (...)
    perp_sq = np.sum(p * p, axis=-1) - along**2           # (...)
    return (along < 0.0) & (perp_sq < _R_EARTH_KM**2)


def _runs(pts: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """
    Split a polyline into maximal contiguous runs where `mask` is True.

    Parameters
    ----------
    pts  : np.ndarray, (K, D)  polyline vertices
    mask : np.ndarray, (K,) bool  which vertices to keep

    Returns
    -------
    list[np.ndarray]  each (M, D) with M >= 2
    """
    out: list[np.ndarray] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= 2:
                out.append(pts[i:j])
            i = j
        else:
            i += 1
    return out


def split_front_back(
    pts: np.ndarray, view: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split a 3-D polyline into the segments in front of, and behind, the Earth.

    Parameters
    ----------
    pts  : np.ndarray, (K, 3) km  polyline in the plotted frame
    view : np.ndarray, (3,)       unit vector toward the camera

    Returns
    -------
    front, back : list[np.ndarray]  each element (M, 3), M >= 2
    """
    occ = occluded(pts, view)                             # (K,)
    return _runs(pts, ~occ), _runs(pts, occ)


def split_lon_wrap(lon: np.ndarray, lat: np.ndarray) -> list[np.ndarray]:
    """
    Split a lat/lon polyline wherever it crosses the +-180 deg seam.

    An equirectangular plot would otherwise draw a horizontal line all the way
    across the map at every antimeridian crossing.

    Parameters
    ----------
    lon : np.ndarray, (K,) degrees in [-180, 180]
    lat : np.ndarray, (K,) degrees

    Returns
    -------
    list[np.ndarray]  each (M, 2) of (lon, lat) with M >= 2
    """
    pts = np.column_stack([np.asarray(lon, float), np.asarray(lat, float)])   # (K, 2)
    jump = np.abs(np.diff(pts[:, 0])) > 180.0                                 # (K-1,)
    cuts = np.flatnonzero(jump) + 1
    return [seg for seg in np.split(pts, cuts) if len(seg) >= 2]


# --- 3-D snapshot figure ---

def _add3d(ax, collection) -> None:
    """Add a Line3DCollection without autoscaling the axes from it.

    The autoscale path of add_collection3d stacks the segment list into one array,
    which fails for segment lists of unequal length such as those produced by
    occlusion splitting. The axis limits are set explicitly elsewhere.

    Parameters
    ----------
    ax         : mpl_toolkits.mplot3d.axes3d.Axes3D
    collection : matplotlib.collections.Collection  a 3-D line collection
    """
    ax.add_collection3d(collection, autolim=False)


def draw_3d(snap: ConstellationSnapshot, opts: FigureOptions) -> plt.Figure:
    """3-D snapshot of the whole constellation at the snapshot epoch.

    Earth, LEO shell and MEO tier share one scale. The axis limits follow the MEO
    radius (about 2.57 Earth radii), so the Earth fills only the middle of the
    frame.

    Parameters
    ----------
    snap : ConstellationSnapshot
    opts : FigureOptions

    Returns
    -------
    matplotlib.figure.Figure
    """
    view = view_direction(opts.view_elev, opts.view_azim)
    colors = palettes.plane_colors(snap.n_planes)
    meo_cols = palettes.meo_colors(len(snap.constellation.meos))

    fig = plt.figure(figsize=_FIG3D_IN)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_facecolor("white")

    # --- Earth (zorder 3; everything behind it is 1-2, in front 4-6) -----
    gx, gy, gz, grgb = earth.globe_surface(
        snap.rot_itrf_to_gcrs,
        n_lat=(opts.globe_lon + 1) // 2,
        n_lon=opts.globe_lon,
        texture_width=opts.texture_width,
        brightness=_GLOBE_BRIGHTNESS_3D,
    )
    ax.plot_surface(
        gx, gy, gz,
        facecolors=grgb, rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False,
        zorder=3, rasterized=True,
    )

    # --- orbit tracks: one closed ring per LEO plane, plus each MEO -----
    if opts.tracks:
        for p in range(snap.n_planes):
            seat0 = int(np.flatnonzero((snap.plane_id == p) & (snap.in_plane_idx == 0))[0])
            ring = snap.leo_orbit_pos[:, seat0, :]                 # (K_leo, 3)
            front, back = split_front_back(ring, view)
            for segs, z, alpha in ((back, 1, 0.22), (front, 4, 0.85)):
                if segs:
                    _add3d(
                        ax,
                        Line3DCollection(
                            segs, colors=colors[p], linewidths=0.7, alpha=alpha, zorder=z
                        )
                    )
        for k in range(len(snap.constellation.meos)):
            ring = snap.meo_orbit_pos[:, k, :]                      # (K_meo, 3)
            front, back = split_front_back(ring, view)
            for segs, z, alpha in ((back, 1, 0.22), (front, 4, 0.85)):
                if segs:
                    _add3d(
                        ax,
                        Line3DCollection(
                            segs, colors=meo_cols[k], linewidths=1.1,
                            linestyles="dashed", alpha=alpha, zorder=z,
                        )
                    )

    # --- LEO->MEO assignment links -----
    if opts.assignment:
        for k in range(len(snap.constellation.meos)):
            idx = np.flatnonzero(snap.assignment == k)              # (n_k,)
            if idx.size == 0:
                continue
            segs = np.stack(
                [snap.leo_pos[idx], np.repeat(snap.meo_pos[k][None, :], idx.size, axis=0)],
                axis=1,
            )                                                       # (n_k, 2, 3)
            mid_occ = occluded(segs.mean(axis=1), view)              # (n_k,)
            for sel, z, alpha in ((~mid_occ, 4, 0.45), (mid_occ, 2, 0.12)):
                if sel.any():
                    _add3d(
                        ax,
                        Line3DCollection(
                            list(segs[sel]), colors=meo_cols[k],
                            linewidths=0.5, alpha=alpha, zorder=z,
                        )
                    )

    # --- ISL / line-of-sight edges among LEOs -----
    if opts.isl and snap.isl_edges:
        e = np.asarray(snap.isl_edges)                               # (E, 2)
        segs = np.stack([snap.leo_pos[e[:, 0]], snap.leo_pos[e[:, 1]]], axis=1)  # (E, 2, 3)
        mid_occ = occluded(segs.mean(axis=1), view)                  # (E,)
        for sel, z, alpha in ((~mid_occ, 4, 0.30), (mid_occ, 2, 0.08)):
            if sel.any():
                _add3d(
                    ax,
                    Line3DCollection(
                        list(segs[sel]), colors=palettes.ISL_EDGE,
                        linewidths=0.35, alpha=alpha, zorder=z,
                    )
                )

    # --- footprint rings at the elevation-mask (rivalry) radius -----
    if opts.footprints:
        # Raised 30 km above the surface so the ring does not coincide with the globe mesh.
        for i in range(snap.n_leo):
            rlat, rlon = earth.small_circle_latlon(
                float(snap.leo_lat[i]), float(snap.leo_lon[i]),
                snap.rivalry_radius_km, n_points=73,
            )
            ring = earth.itrf_to_gcrs(
                earth.geodetic_to_itrf(rlat, rlon, 30.0), snap.rot_itrf_to_gcrs
            )                                                        # (73, 3)
            front, _ = split_front_back(ring, view)
            if front:
                _add3d(
                    ax,
                    Line3DCollection(
                        front, colors=palettes.FOOTPRINT_RING,
                        linewidths=0.35, alpha=0.5, zorder=4,
                    )
                )

    # --- satellites -----
    leo_occ = occluded(snap.leo_pos, view)                           # (N_leo,)
    for p in range(snap.n_planes):
        in_plane = snap.plane_id == p
        for sel, z, alpha, size in (
            (in_plane & leo_occ, 2, 0.30, 9.0),
            (in_plane & ~leo_occ, 5, 1.00, 16.0),
        ):
            if sel.any():
                ax.scatter(
                    snap.leo_pos[sel, 0], snap.leo_pos[sel, 1], snap.leo_pos[sel, 2],
                    s=size, c=colors[p], alpha=alpha, zorder=z,
                    edgecolors=palettes.MARKER_EDGE, linewidths=0.25, depthshade=False,
                )

    # Unassigned LEOs (no MEO in line of sight) get a distinct open marker.
    blind = snap.assignment < 0                                      # (N_leo,)
    if blind.any():
        for sel, z, alpha in ((blind & ~leo_occ, 6, 1.0), (blind & leo_occ, 2, 0.3)):
            if sel.any():
                ax.scatter(
                    snap.leo_pos[sel, 0], snap.leo_pos[sel, 1], snap.leo_pos[sel, 2],
                    s=46.0, marker="s", facecolors="none",
                    edgecolors=palettes.UNASSIGNED_COLOR, linewidths=1.2,
                    alpha=alpha, zorder=z, depthshade=False,
                )

    meo_occ = occluded(snap.meo_pos, view)                           # (N_meo,)
    for k in range(len(snap.constellation.meos)):
        z, alpha = (2, 0.35) if meo_occ[k] else (6, 1.0)
        ax.scatter(
            [snap.meo_pos[k, 0]], [snap.meo_pos[k, 1]], [snap.meo_pos[k, 2]],
            s=150.0, marker="*", c=meo_cols[k], alpha=alpha, zorder=z,
            edgecolors=palettes.MARKER_EDGE, linewidths=0.5, depthshade=False,
        )
        ax.text(
            snap.meo_pos[k, 0], snap.meo_pos[k, 1], snap.meo_pos[k, 2] + 900.0,
            f"MEO-{k:02d}", color=meo_cols[k], fontsize=7.5, ha="center",
            zorder=6, fontweight="bold",
        )

    # --- framing -----
    lim = float(np.linalg.norm(snap.meo_pos, axis=-1).max()) * 1.10
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    # zoom > 1 reduces the empty margin around the projected cube; the data limits
    # are unchanged, so the figure stays to scale.
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=_CUBE_ZOOM)
    ax.view_init(elev=opts.view_elev, azim=opts.view_azim)
    ax.set_axis_off()

    fig.suptitle(
        f"orbitfl constellation — {snap.n_leo} LEO (Walker-delta "
        f"{snap.n_planes}/{snap.n_leo}/{snap.phasing_f}, {snap.leo_alt_km:g} km, "
        f"{snap.leo_inc_deg:g}°) + {len(snap.constellation.meos)} MEO "
        f"({snap.meo_period_h:g} h, {snap.meo_inc_deg:g}°)\n{snap.epoch_iso}  ·  "
        "GCRS, WGS-84 ellipsoid, true scale",
        fontsize=9.5, y=0.97,
    )
    fig.legend(
        handles=_legend_handles(snap, colors, meo_cols, opts),
        loc="lower center", ncol=8, fontsize=6.5, frameon=False,
        handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.08)
    return fig


def _legend_handles(
    snap: ConstellationSnapshot,
    colors: list[str],
    meo_cols: list[str],
    opts: FigureOptions,
) -> list[Line2D]:
    """
    Legend proxies: one per LEO plane, one per MEO, plus active overlays.

    Parameters
    ----------
    snap     : ConstellationSnapshot
    colors   : list[str], length n_planes  plane colours
    meo_cols : list[str], length n_meo     MEO colours
    opts     : FigureOptions

    Returns
    -------
    list[matplotlib.lines.Line2D]  proxy artists in legend order
    """
    handles = [
        Line2D([], [], color=c, marker="o", markersize=3.6, linestyle="-",
               linewidth=0.8, markeredgecolor=palettes.MARKER_EDGE,
               markeredgewidth=0.25, label=f"plane {p}")
        for p, c in enumerate(colors)
    ]
    handles += [
        Line2D([], [], color=c, marker="*", markersize=8.0, linestyle="--",
               linewidth=1.0, markeredgecolor=palettes.MARKER_EDGE,
               markeredgewidth=0.4, label=f"MEO-{k:02d}")
        for k, c in enumerate(meo_cols)
    ]
    if (snap.assignment < 0).any():
        handles.append(
            Line2D([], [], color=palettes.UNASSIGNED_COLOR, marker="s", markersize=5.5,
                   markerfacecolor="none", markeredgewidth=1.2, linestyle="none",
                   label="unassigned (-1)")
        )
    if opts.isl:
        handles.append(
            Line2D([], [], color="0.35", linewidth=0.6,
                   label=f"ISL (LOS, ≤{snap.isl_max_range_km:g} km)")
        )
    if opts.footprints:
        handles.append(
            Line2D([], [], color="0.35", linewidth=0.6, linestyle=":",
                   label=f"{snap.elevation_mask_deg:g}° mask "
                         f"({snap.rivalry_radius_km:.0f} km)")
        )
    return handles


# --- Ground-track panel ---


def draw_groundtrack(snap: ConstellationSnapshot, opts: FigureOptions) -> plt.Figure:
    """Equirectangular map of all sub-satellite points with trailing tracks.

    Assignment is shown by the colour of the marker edge. A LEO-to-MEO link is a
    chord in 3-D and cannot be drawn meaningfully on this map.

    Parameters
    ----------
    snap : ConstellationSnapshot
    opts : FigureOptions

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = palettes.plane_colors(snap.n_planes)
    meo_cols = palettes.meo_colors(len(snap.constellation.meos))

    fig, ax = plt.subplots(figsize=(_GT_WIDTH_IN, _GT_WIDTH_IN / 2.0 + 1.15))
    ax.imshow(
        np.clip(earth.load_texture(2048) * _GLOBE_BRIGHTNESS_2D, 0.0, 1.0),
        extent=(-180.0, 180.0, -90.0, 90.0), origin="upper",
        aspect="equal", interpolation="bilinear", zorder=0, rasterized=True,
    )

    # graticule every 30 deg
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color=palettes.GRATICULE, lw=0.3, alpha=0.25, zorder=1)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color=palettes.GRATICULE, lw=0.3, alpha=0.25, zorder=1)

    # LEO coverage limit: peak |sub-satellite latitude| over one orbit
    lat_limit = float(np.abs(snap.leo_orbit_lat).max())
    for sign in (+1.0, -1.0):
        ax.axhline(sign * lat_limit, color="white", lw=0.6, ls=(0, (4, 3)),
                   alpha=0.85, zorder=2)
    ax.text(
        -178.0, lat_limit + 2.5, f"LEO coverage limit  ±{lat_limit:.1f}°",
        color="white", fontsize=6.0, va="bottom", zorder=3,
    )

    # --- trailing tracks -----
    for p in range(snap.n_planes):
        segs: list[np.ndarray] = []
        for i in np.flatnonzero(snap.plane_id == p):
            segs += split_lon_wrap(snap.trail_leo_lon[:, i], snap.trail_leo_lat[:, i])
        if segs:
            ax.add_collection(
                LineCollection(segs, colors=colors[p], linewidths=0.6, alpha=0.8, zorder=4)
            )
    for k in range(len(snap.constellation.meos)):
        segs = split_lon_wrap(snap.trail_meo_lon[:, k], snap.trail_meo_lat[:, k])
        if segs:
            ax.add_collection(
                LineCollection(segs, colors=meo_cols[k], linewidths=1.2,
                               linestyles="dashed", alpha=0.9, zorder=4)
            )

    # --- footprint rings -----
    if opts.footprints:
        rings: list[np.ndarray] = []
        for i in range(snap.n_leo):
            rlat, rlon = earth.small_circle_latlon(
                float(snap.leo_lat[i]), float(snap.leo_lon[i]),
                snap.rivalry_radius_km, n_points=145,
            )
            rings += split_lon_wrap(rlon, rlat)
        ax.add_collection(
            LineCollection(rings, colors=palettes.FOOTPRINT_RING, linewidths=0.35,
                           alpha=0.55, zorder=5)
        )

    # --- markers -----
    edge = (
        np.where(
            snap.assignment < 0,
            palettes.UNASSIGNED_COLOR,
            np.asarray(meo_cols)[np.clip(snap.assignment, 0, None)],
        )
        if opts.assignment
        else np.full(snap.n_leo, palettes.MARKER_EDGE)
    )                                                             # (N_leo,) hex strings
    for p in range(snap.n_planes):
        sel = snap.plane_id == p
        ax.scatter(
            snap.leo_lon[sel], snap.leo_lat[sel], s=22.0, c=colors[p],
            edgecolors=edge[sel], linewidths=0.8, zorder=6,
        )
    blind = snap.assignment < 0
    if blind.any():
        ax.scatter(
            snap.leo_lon[blind], snap.leo_lat[blind], s=60.0, marker="s",
            facecolors="none", edgecolors=palettes.UNASSIGNED_COLOR,
            linewidths=1.1, zorder=7,
        )
    for k in range(len(snap.constellation.meos)):
        ax.scatter(
            [snap.meo_lon[k]], [snap.meo_lat[k]], s=190.0, marker="*",
            c=meo_cols[k], edgecolors=palettes.MARKER_EDGE, linewidths=0.6, zorder=8,
        )
        ax.annotate(
            f"MEO-{k:02d}", (snap.meo_lon[k], snap.meo_lat[k]),
            textcoords="offset points", xytext=(0, 9), ha="center",
            fontsize=7.0, color="white", zorder=8,
            path_effects=None, fontweight="bold",
        )

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude (deg)", fontsize=8)
    ax.set_ylabel("latitude (deg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"Sub-satellite points of all {snap.n_sats} satellites at {snap.epoch_iso}\n"
        f"trailing track = {snap.trail_minutes:g} min; marker edge = assigned MEO"
        if opts.assignment
        else f"Sub-satellite points of all {snap.n_sats} satellites at {snap.epoch_iso}\n"
             f"trailing track = {snap.trail_minutes:g} min",
        fontsize=8.5,
    )
    fig.legend(
        handles=_legend_handles(snap, colors, meo_cols, opts),
        loc="lower center", ncol=8, fontsize=6.0, frameon=False,
        handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.17)
    return fig


# --- Interactive HTML ---


def write_html(snap: ConstellationSnapshot, opts: FigureOptions, path: Path) -> Path:
    """
    Write an interactive Plotly version of the 3-D snapshot.

    WebGL does its own depth testing, so no occlusion splitting is needed here.
    Legend entries toggle each plane, each MEO, the tracks, the assignment links
    and the ISL edges.

    Parameters
    ----------
    snap : ConstellationSnapshot
    opts : FigureOptions
    path : Path  output .html path

    Returns
    -------
    Path  the written path
    """
    import plotly.graph_objects as go

    colors = palettes.plane_colors(snap.n_planes)
    meo_cols = palettes.meo_colors(len(snap.constellation.meos))

    # --- globe: texture reduced to an indexed colourscale ---
    n_lon = opts.html_globe_lon
    n_lat = (n_lon + 1) // 2
    indexed, colorscale = earth.texture_colorscale(n_colors=255, texture_width=512)
    lat = np.linspace(90.0, -90.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")              # (n_lat, n_lon)
    v = earth.itrf_to_gcrs(
        earth.geodetic_to_itrf(lat_g, lon_g, 0.0), snap.rot_itrf_to_gcrs
    )                                                                 # (n_lat, n_lon, 3)
    h, w = indexed.shape
    row = np.clip(((90.0 - lat_g) / 180.0 * h).astype(int), 0, h - 1)
    col = np.clip(((lon_g + 180.0) / 360.0 * w).astype(int), 0, w - 1)

    traces: list = [
        go.Surface(
            x=v[..., 0], y=v[..., 1], z=v[..., 2],
            surfacecolor=indexed[row, col].astype(float),
            colorscale=colorscale, cmin=0.0, cmax=255.0,
            showscale=False, name="Earth", hoverinfo="skip", lighting={"ambient": 1.0},
        )
    ]

    # --- orbit tracks -----
    if opts.tracks:
        for p in range(snap.n_planes):
            seat0 = int(np.flatnonzero((snap.plane_id == p) & (snap.in_plane_idx == 0))[0])
            ring = snap.leo_orbit_pos[:, seat0, :]                    # (K_leo, 3)
            traces.append(
                go.Scatter3d(
                    x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                    line={"color": colors[p], "width": 2},
                    name=f"plane {p} track", legendgroup=f"plane{p}",
                    showlegend=False, hoverinfo="skip",
                )
            )
        for k in range(len(snap.constellation.meos)):
            ring = snap.meo_orbit_pos[:, k, :]                        # (K_meo, 3)
            traces.append(
                go.Scatter3d(
                    x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                    line={"color": meo_cols[k], "width": 3, "dash": "dash"},
                    name=f"MEO-{k:02d} track", legendgroup=f"meo{k}",
                    showlegend=False, hoverinfo="skip",
                )
            )

    # --- assignment links -----
    if opts.assignment:
        for k in range(len(snap.constellation.meos)):
            idx = np.flatnonzero(snap.assignment == k)                # (n_k,)
            if idx.size == 0:
                continue
            xs, ys, zs = [], [], []
            for i in idx:
                xs += [snap.leo_pos[i, 0], snap.meo_pos[k, 0], None]
                ys += [snap.leo_pos[i, 1], snap.meo_pos[k, 1], None]
                zs += [snap.leo_pos[i, 2], snap.meo_pos[k, 2], None]
            traces.append(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line={"color": meo_cols[k], "width": 1},
                    opacity=0.45, name=f"→ MEO-{k:02d} ({idx.size})",
                    hoverinfo="skip",
                )
            )

    # --- ISL edges -----
    if opts.isl and snap.isl_edges:
        xs, ys, zs = [], [], []
        for i, j in snap.isl_edges:
            xs += [snap.leo_pos[i, 0], snap.leo_pos[j, 0], None]
            ys += [snap.leo_pos[i, 1], snap.leo_pos[j, 1], None]
            zs += [snap.leo_pos[i, 2], snap.leo_pos[j, 2], None]
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line={"color": "#888888", "width": 1}, opacity=0.35,
                name=f"ISL ≤{snap.isl_max_range_km:g} km ({len(snap.isl_edges)})",
                visible="legendonly", hoverinfo="skip",
            )
        )

    # --- satellites -----
    for p in range(snap.n_planes):
        sel = snap.plane_id == p                                      # (N_leo,)
        idx = np.flatnonzero(sel)
        traces.append(
            go.Scatter3d(
                x=snap.leo_pos[sel, 0], y=snap.leo_pos[sel, 1], z=snap.leo_pos[sel, 2],
                mode="markers",
                marker={"size": 3.2, "color": colors[p],
                        "line": {"color": "#222222", "width": 0.4}},
                name=f"plane {p}", legendgroup=f"plane{p}",
                customdata=np.column_stack([
                    idx, snap.in_plane_idx[sel], snap.leo_lat[sel],
                    snap.leo_lon[sel], snap.assignment[sel],
                ]),
                hovertemplate=(
                    "LEO %{customdata[0]} (plane " + str(p) + ", seat %{customdata[1]})"
                    "<br>lat %{customdata[2]:.2f}°, lon %{customdata[3]:.2f}°"
                    "<br>assigned MEO %{customdata[4]}<extra></extra>"
                ),
            )
        )
    for k in range(len(snap.constellation.meos)):
        traces.append(
            go.Scatter3d(
                x=[snap.meo_pos[k, 0]], y=[snap.meo_pos[k, 1]], z=[snap.meo_pos[k, 2]],
                mode="markers+text",
                marker={"size": 9, "color": meo_cols[k], "symbol": "diamond",
                        "line": {"color": "#222222", "width": 0.6}},
                text=[f"MEO-{k:02d}"], textposition="top center",
                textfont={"size": 10, "color": meo_cols[k]},
                name=f"MEO-{k:02d}", legendgroup=f"meo{k}",
                hovertemplate=(
                    f"MEO-{k:02d}<br>lat {snap.meo_lat[k]:.2f}°, "
                    f"lon {snap.meo_lon[k]:.2f}°<extra></extra>"
                ),
            )
        )

    lim = float(np.linalg.norm(snap.meo_pos, axis=-1).max()) * 1.10
    axis = {"range": [-lim, lim], "visible": False}
    fig = go.Figure(data=traces)
    fig.update_layout(
        title={
            "text": (
                f"orbitfl constellation — {snap.n_leo} LEO (Walker-delta "
                f"{snap.n_planes}/{snap.n_leo}/{snap.phasing_f}, "
                f"{snap.leo_alt_km:g} km, {snap.leo_inc_deg:g}°) + "
                f"{len(snap.constellation.meos)} MEO ({snap.meo_period_h:g} h, "
                f"{snap.meo_inc_deg:g}°)<br>"
                f"<sub>{snap.epoch_iso} · GCRS · WGS-84 ellipsoid · true scale</sub>"
            ),
            "font": {"size": 14},
        },
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis,
               "aspectmode": "cube", "bgcolor": "white"},
        legend={"itemsizing": "constant", "font": {"size": 10}},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        template="plotly_white",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path, include_plotlyjs=("cdn" if opts.html_cdn else True), full_html=True
    )
    return path


# --- Optional animation ---


def write_animation(snap: ConstellationSnapshot, opts: FigureOptions, path: Path) -> Path:
    """Write a short orbit animation over `anim_periods` LEO orbital periods.

    The frame is GCRS, so the Earth rotates under the satellites. The globe is
    rotated again for every frame, on a coarser mesh to keep rendering time down.

    Parameters
    ----------
    snap : ConstellationSnapshot
    opts : FigureOptions
    path : Path  output .gif or .mp4 path

    Returns
    -------
    Path  the written path
    """
    from matplotlib import animation

    from orbitfl.viz.constellation_data import _times_from  # same time convention

    colors = palettes.plane_colors(snap.n_planes)
    meo_cols = palettes.meo_colors(len(snap.constellation.meos))
    view = view_direction(opts.view_elev, opts.view_azim)

    offsets = np.linspace(
        0.0, opts.anim_periods * snap.leo_period_s, opts.anim_frames
    )                                                                 # (F,)
    leo_f, meo_f = propagate_window(snap, offsets)                    # (F, N, 3) each
    times = _times_from(snap.epoch, offsets)
    rots = [
        earth.itrf_to_gcrs_rotation(times[i]) for i in range(opts.anim_frames)
    ]                                                                 # F x (3, 3)

    lim = float(np.linalg.norm(snap.meo_pos, axis=-1).max()) * 1.10
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    def frame(fi: int) -> None:
        """Render animation frame `fi` in place."""
        ax.clear()
        ax.set_axis_off()
        gx, gy, gz, grgb = earth.globe_surface(
            rots[fi], n_lat=91, n_lon=181, texture_width=512,
            brightness=_GLOBE_BRIGHTNESS_3D,
        )
        ax.plot_surface(
            gx, gy, gz, facecolors=grgb, rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False, zorder=3,
        )
        occ = occluded(leo_f[fi], view)
        for p in range(snap.n_planes):
            sel = snap.plane_id == p
            for m, z, alpha, size in ((sel & occ, 2, 0.3, 6.0), (sel & ~occ, 5, 1.0, 12.0)):
                if m.any():
                    ax.scatter(
                        leo_f[fi][m, 0], leo_f[fi][m, 1], leo_f[fi][m, 2],
                        s=size, c=colors[p], alpha=alpha, zorder=z, depthshade=False,
                    )
        mocc = occluded(meo_f[fi], view)
        for k in range(len(snap.constellation.meos)):
            z, alpha = (2, 0.35) if mocc[k] else (6, 1.0)
            ax.scatter(
                [meo_f[fi][k, 0]], [meo_f[fi][k, 1]], [meo_f[fi][k, 2]],
                s=110.0, marker="*", c=meo_cols[k], alpha=alpha, zorder=z,
                depthshade=False,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=_CUBE_ZOOM)
        ax.view_init(elev=opts.view_elev, azim=opts.view_azim)
        ax.set_title(
            f"{snap.epoch_iso}  +{offsets[fi] / 60.0:6.1f} min", fontsize=9, family="monospace"
        )

    anim = animation.FuncAnimation(
        fig, frame, frames=opts.anim_frames, interval=1000 // opts.anim_fps
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp4":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("--anim-format mp4 needs ffmpeg on PATH; use gif instead")
        anim.save(path, writer=animation.FFMpegWriter(fps=opts.anim_fps, bitrate=2400))
    else:
        anim.save(path, writer=animation.PillowWriter(fps=opts.anim_fps))
    plt.close(fig)
    return path


# --- Caption ---


def caption_text(
    snap: ConstellationSnapshot,
    checks: list[Check],
    opts: FigureOptions,
    outputs: list[Path],
) -> str:
    """Markdown figure caption: epoch, parameters, overlays and check results.

    Parameters
    ----------
    snap    : ConstellationSnapshot
    checks  : list[Check]   rows from run_checks
    opts    : FigureOptions
    outputs : list[Path]    files written by this run

    Returns
    -------
    str  markdown, newline-terminated
    """
    n_pass = sum(c.passed for c in checks)
    overlays = [
        name for name, on in (
            ("orbit tracks", opts.tracks),
            ("LEO→MEO assignment", opts.assignment),
            ("ISL/LOS edges", opts.isl),
            (f"{snap.elevation_mask_deg:g}° footprint rings", opts.footprints),
        ) if on
    ]
    n_blind = int((snap.assignment < 0).sum())
    lat_limit = float(np.abs(snap.leo_orbit_lat).max())
    meo_m = ", ".join(f"{m:g}°" for m in snap.meo_mean_anom_deg)

    body = f"""# Constellation figure: caption and parameters

**Epoch.** {snap.epoch_iso} (UTC). Single fixed snapshot; the trailing ground
track covers the {snap.trail_minutes:g} min before it.

**Caption.**

> The orbitfl constellation at {snap.epoch_iso}: {snap.n_leo} LEO satellites in a
> Walker-delta {snap.n_planes}/{snap.n_leo}/{snap.phasing_f} shell at
> {snap.leo_alt_km:g} km altitude and {snap.leo_inc_deg:g}° inclination (RAANs
> spaced {360.0 / snap.n_planes:g}° over the full 360°, {snap.sats_per_plane} seats
> per plane, coloured by plane), together with {len(snap.constellation.meos)} MEO
> relays in circular {snap.meo_period_h:g} h orbits
> (a = {snap.meo_sma_km:.1f} km, altitude {snap.meo_sma_km - _R_EARTH_KM:.0f} km
> ≈ {(snap.meo_sma_km) / _R_EARTH_KM:.2f} Earth radii from the centre,
> {snap.meo_inc_deg:g}° inclination, RAANs {360.0 / snap.n_meo:g}° apart). Earth,
> LEO shell and MEO tier are drawn to a common scale on the WGS-84 ellipsoid.
> Overlays: {", ".join(overlays) if overlays else "none"}. Positions are
> propagated with SGP4 through the simulator's own
> `geometry.orbits.propagate`; the LEO→MEO links are the simulator's own
> `geometry.visibility.leo_meo_assignment` (nearest MEO with line of sight,
> 80 km grazing altitude), with {n_blind} LEO(s) unassigned at this epoch.

**Parameters, as read from `{snap.config_path}`**
{f"(override `{snap.override_path}`)" if snap.override_path else ""}

| Parameter | Value |
|---|---|
| LEO planes × seats | {snap.n_planes} × {snap.sats_per_plane} = {snap.n_leo} |
| LEO altitude | {snap.leo_alt_km:g} km |
| LEO inclination | {snap.leo_inc_deg:g}° |
| Walker phasing factor *f* | {snap.phasing_f} |
| LEO RAAN spacing | {360.0 / snap.n_planes:g}° over 360° (Walker delta) |
| LEO period (SGP4 mean motion) | {snap.leo_period_s / 60.0:.2f} min |
| Peak \\|sub-satellite latitude\\| | ±{lat_limit:.2f}° |
| MEO count | {len(snap.constellation.meos)} |
| MEO period | {snap.meo_period_h:g} h |
| MEO semi-major axis | {snap.meo_sma_km:.3f} km |
| MEO altitude | {snap.meo_sma_km - _R_EARTH_KM:.3f} km |
| MEO inclination | {snap.meo_inc_deg:g}° |
| MEO RAANs | {", ".join(f"{r:g}°" for r in snap.meo_raan_deg)} |
| MEO initial mean anomaly | {meo_m} (set in `build_meo_tier`) |
| ISL max range | {snap.isl_max_range_km:g} km ({len(snap.isl_edges)} LOS edges at this epoch) |
| Elevation mask | {snap.elevation_mask_deg:g}° → rivalry radius {snap.rivalry_radius_km:.1f} km |
| Beam half-cone | {snap.beam_half_cone_deg:g}° → footprint {snap.footprint_radius_km:.1f} km |
| Earth model | WGS-84: R = {_R_EARTH_KM} km, 1/f = 298.257223563, μ = 3.986004418e5 km³/s² |

**Verification.** {n_pass}/{len(checks)} checks passed
(`python -m orbitfl.viz.check_constellation`). Full table:

```
{format_table(checks, snap)}```

**Files written by this run**

{chr(10).join(f"- `{p}`" for p in outputs)}
"""
    return body


# --- CLI ---


def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser; every figure option is defined here."""
    p = argparse.ArgumentParser(
        description="Render the orbitfl constellation figures from the simulator pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--override", type=Path, default=None,
                   help="optional override YAML, deep-merged over --config")
    p.add_argument("--epoch", type=str, default=None,
                   help="ISO-8601 UTC snapshot epoch; default: sim.epoch from config")
    p.add_argument("--outdir", type=Path, default=Path("figures/constellation"))
    p.add_argument("--prefix", type=str, default="constellation")
    p.add_argument("--dpi", type=int, default=300)

    ov = p.add_argument_group("overlays")
    ov.add_argument("--tracks", action=argparse.BooleanOptionalAction, default=True,
                    help="full-orbit track lines, one per LEO plane and per MEO")
    ov.add_argument("--assignment", action=argparse.BooleanOptionalAction, default=True,
                    help="LEO->MEO links from geometry.visibility.leo_meo_assignment")
    ov.add_argument("--isl", action=argparse.BooleanOptionalAction, default=False,
                    help="ISL/LOS edges from geometry.visibility.isl_pairs")
    ov.add_argument("--footprints", action=argparse.BooleanOptionalAction, default=False,
                    help="sub-satellite rings at the elevation-mask rivalry radius")

    vw = p.add_argument_group("3-D view")
    vw.add_argument("--view-elev", type=float, default=24.0)
    vw.add_argument("--view-azim", type=float, default=-62.0)
    vw.add_argument("--globe-lon", type=int, default=541,
                    help="longitude samples of the globe mesh (quality vs render time)")
    vw.add_argument("--texture-width", type=int, default=2048)

    sn = p.add_argument_group("snapshot sampling")
    sn.add_argument("--orbit-samples", type=int, default=241,
                    help="samples per full orbit for track lines and max-latitude check")
    sn.add_argument("--trail-min", type=float, default=20.0,
                    help="length of the trailing ground track in minutes")
    sn.add_argument("--trail-samples", type=int, default=41)

    ou = p.add_argument_group("outputs")
    ou.add_argument("--png", action=argparse.BooleanOptionalAction, default=True)
    ou.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True)
    ou.add_argument("--html", action=argparse.BooleanOptionalAction, default=True)
    ou.add_argument("--html-globe-lon", type=int, default=181)
    ou.add_argument("--html-cdn", action="store_true",
                    help="load plotly.js from a CDN instead of embedding it "
                         "(smaller file, needs network to view)")
    ou.add_argument("--animate", action="store_true", help="also render an orbit animation")
    ou.add_argument("--anim-frames", type=int, default=90)
    ou.add_argument("--anim-periods", type=float, default=1.0)
    ou.add_argument("--anim-fps", type=int, default=15)
    ou.add_argument("--anim-format", choices=("gif", "mp4"), default="gif")
    return p


def main(argv: list[str] | None = None) -> int:
    """Build the snapshot, check it, then render every requested output.

    The checks run before any figure is written, and a failed check aborts the run.

    Parameters
    ----------
    argv : list[str] | None  argument vector (defaults to sys.argv[1:])

    Returns
    -------
    int  0 on success, 1 if a check failed
    """
    args = _build_parser().parse_args(argv)
    opts = FigureOptions(
        outdir=args.outdir, prefix=args.prefix, dpi=args.dpi,
        tracks=args.tracks, assignment=args.assignment,
        isl=args.isl, footprints=args.footprints,
        view_elev=args.view_elev, view_azim=args.view_azim,
        globe_lon=args.globe_lon, texture_width=args.texture_width,
        png=args.png, pdf=args.pdf, html=args.html,
        html_globe_lon=args.html_globe_lon, html_cdn=args.html_cdn,
        animate=args.animate, anim_frames=args.anim_frames,
        anim_periods=args.anim_periods, anim_fps=args.anim_fps,
        anim_format=args.anim_format,
    )

    print(f"config : {args.config}" + (f" + {args.override}" if args.override else ""))
    snap = build_snapshot(
        args.config,
        override_path=args.override,
        epoch_iso=args.epoch,
        orbit_samples=args.orbit_samples,
        trail_minutes=args.trail_min,
        trail_samples=args.trail_samples,
    )
    print(f"epoch  : {snap.epoch_iso}")
    print(f"sats   : {snap.n_leo} LEO + {len(snap.constellation.meos)} MEO = {snap.n_sats}")

    checks = run_checks(snap)
    table = format_table(checks, snap)
    print()
    print(table, end="")
    if not all(c.passed for c in checks):
        print("\nChecks failed; no figures written. "
              "Fix the discrepancy; do not loosen the tolerances.", file=sys.stderr)
        return 1

    stamp = snap.epoch_iso.replace("-", "").replace(":", "").replace("Z", "Z")
    opts.outdir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    def _save(fig: plt.Figure, name: str) -> None:
        """Save one matplotlib figure in every requested static format."""
        for ext, on in (("png", opts.png), ("pdf", opts.pdf)):
            if not on:
                continue
            path = opts.outdir / f"{opts.prefix}_{name}_{stamp}.{ext}"
            fig.savefig(path, dpi=opts.dpi, bbox_inches="tight", facecolor="white")
            outputs.append(path)
            print(f"wrote {path}")

    print("\nrendering 3-D snapshot ...", flush=True)
    fig3d = draw_3d(snap, opts)
    _save(fig3d, "3d")
    plt.close(fig3d)

    print("rendering ground-track panel ...", flush=True)
    figgt = draw_groundtrack(snap, opts)
    _save(figgt, "groundtrack")
    plt.close(figgt)

    if opts.html:
        print("rendering interactive HTML ...", flush=True)
        path = write_html(snap, opts, opts.outdir / f"{opts.prefix}_3d_{stamp}.html")
        outputs.append(path)
        print(f"wrote {path}")

    if opts.animate:
        print(f"rendering animation ({opts.anim_frames} frames) ...", flush=True)
        path = write_animation(
            snap, opts, opts.outdir / f"{opts.prefix}_orbit_{stamp}.{opts.anim_format}"
        )
        outputs.append(path)
        print(f"wrote {path}")

    ver_path = opts.outdir / f"{opts.prefix}_verification_{stamp}.txt"
    ver_path.write_text(table)
    outputs.append(ver_path)
    print(f"wrote {ver_path}")

    cap_path = opts.outdir / f"{opts.prefix}_caption_{stamp}.md"
    cap_path.write_text(caption_text(snap, checks, opts, outputs))
    print(f"wrote {cap_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
