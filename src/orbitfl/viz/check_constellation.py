"""Checks on the plotted constellation: measured against expected, as a table.

Every check is made on a ConstellationSnapshot, the object the figures draw.
Expected values are derived from the config or from orbital mechanics; only the
tolerances are literals, each explained in TOLERANCES below.

Run on its own:
    python -m orbitfl.viz.check_constellation --config configs/base.yaml
The exit status is 0 if every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

from orbitfl.geometry.constants import R_EARTH_KM as _R_EARTH_KM
from orbitfl.viz.constellation_data import ConstellationSnapshot, build_snapshot
from orbitfl.viz.earth import geodetic_to_itrf, great_circle_km, itrf_to_gcrs

# --- Tolerances ---
TOLERANCES: dict[str, float] = {
    # SGP4 propagates mean elements, so the instantaneous radius oscillates about
    # the nominal value by about 6 km at 550 km (J2 short-period terms). The
    # orbit-mean altitude is within about 0.2 km.
    "leo_alt_km": 2.0,
    # At MEO the oscillation is about 5 km and the orbit-mean error about 0.4 km.
    "meo_alt_km": 1.0,
    # The sub-satellite point is geodetic, so the peak sub-satellite latitude
    # exceeds the geocentric inclination by about f*sin(2i) ~ 0.18 deg. 0.5 deg
    # covers that plus the finite track sampling.
    "max_sublat_deg": 0.5,
    # The period from the SGP4 mean motion is accurate to the rounding of the TLE
    # text format (about 1e-5 h).
    "period_h": 0.01,
    # In-plane ground-track separation, as a fraction.
    "ground_sep_frac": 0.01,
    # RAAN values round-trip through the TLE's 4-decimal field.
    "raan_deg": 1e-3,
    # Angle between a satellite's radial direction and its plotted sub-satellite
    # point. About 0.02 deg because the sub-satellite point is geodetic; a larger
    # value means the Earth texture is rotated incorrectly.
    "globe_orientation_deg": 0.05,
    # The rivalry radius is a closed-form function of altitude and mask.
    "rivalry_km": 0.1,
}


@dataclasses.dataclass(frozen=True)
class Check:
    """One row of the verification table."""

    name: str
    measured: str
    expected: str
    tolerance: str
    passed: bool
    note: str = ""


def _fmt(x: float, nd: int = 3) -> str:
    """Format a float for the table with `nd` decimals."""
    return f"{x:.{nd}f}"


# --- The checks ---


def run_checks(snap: ConstellationSnapshot) -> list[Check]:
    """Measure the checkable properties of the plotted constellation.

    Parameters
    ----------
    snap : ConstellationSnapshot  the object the figures draw

    Returns
    -------
    list[Check]  one row per property, in table order
    """
    checks: list[Check] = []
    n_leo_expected = snap.n_planes * snap.sats_per_plane

    # --- 1-2. satellite counts -----
    checks.append(
        Check(
            "LEO satellite count",
            str(snap.n_leo),
            str(n_leo_expected),
            "exact",
            snap.n_leo == n_leo_expected,
            f"{snap.n_planes} planes x {snap.sats_per_plane} seats",
        )
    )
    n_meo = len(snap.constellation.meos)
    checks.append(
        Check("MEO satellite count", str(n_meo), str(snap.n_meo), "exact", n_meo == snap.n_meo)
    )

    # --- 3-4. plane structure, from the RAANs of the plotted objects ---
    raan_leo = np.asarray(snap.leo_raan_deg)                     # (N_leo,)
    raan_rounded = np.round(raan_leo, 4)
    uniq, counts = np.unique(raan_rounded, return_counts=True)   # (P,), (P,)
    checks.append(
        Check(
            "Distinct LEO orbital planes",
            str(uniq.size),
            str(snap.n_planes),
            "exact",
            uniq.size == snap.n_planes,
            "counted as distinct RAANs of the built satellites",
        )
    )
    checks.append(
        Check(
            "Satellites per plane (min/max)",
            f"{counts.min()}/{counts.max()}",
            f"{snap.sats_per_plane}/{snap.sats_per_plane}",
            "exact",
            bool(counts.min() == counts.max() == snap.sats_per_plane),
        )
    )

    # --- 5. Walker delta: RAANs span the full 360 deg ---
    spacing = np.diff(np.sort(uniq))                             # (P-1,)
    expected_spacing = 360.0 / snap.n_planes
    checks.append(
        Check(
            "LEO RAAN spacing (Walker-delta)",
            _fmt(float(spacing.mean())) + " deg",
            _fmt(expected_spacing) + " deg",
            f"+/-{TOLERANCES['raan_deg']:g} deg",
            bool(np.all(np.abs(spacing - expected_spacing) < TOLERANCES["raan_deg"])),
            "RAANs span 360 deg (delta), not 180 deg (star)",
        )
    )

    # --- 6-7. peak sub-satellite latitude over one full orbit -----
    for label, lat, inc in (
        ("LEO", snap.leo_orbit_lat, snap.leo_inc_deg),
        ("MEO", snap.meo_orbit_lat, snap.meo_inc_deg),
    ):
        measured = float(np.abs(lat).max())
        checks.append(
            Check(
                f"{label} max |sub-satellite lat| (1 orbit)",
                _fmt(measured) + " deg",
                _fmt(inc) + " deg",
                f"+/-{TOLERANCES['max_sublat_deg']:g} deg",
                abs(measured - inc) < TOLERANCES["max_sublat_deg"],
                "geodetic subpoint exceeds inclination by ~f*sin(2i)",
            )
        )

    # --- 8. LEO altitude from the plotted positions -----
    r_leo = np.linalg.norm(snap.leo_orbit_pos, axis=-1)          # (K_leo, N_leo)
    alt_leo = float(r_leo.mean()) - _R_EARTH_KM
    checks.append(
        Check(
            "LEO altitude (orbit-mean radius - R_e)",
            _fmt(alt_leo) + " km",
            _fmt(snap.leo_alt_km) + " km",
            f"+/-{TOLERANCES['leo_alt_km']:g} km",
            abs(alt_leo - snap.leo_alt_km) < TOLERANCES["leo_alt_km"],
            f"instantaneous range {r_leo.min() - _R_EARTH_KM:.1f}"
            f"-{r_leo.max() - _R_EARTH_KM:.1f} km (SGP4 J2 short-period)",
        )
    )

    # --- 9. MEO altitude -----
    r_meo = np.linalg.norm(snap.meo_orbit_pos, axis=-1)          # (K_meo, N_meo)
    alt_meo = float(r_meo.mean()) - _R_EARTH_KM
    alt_meo_expected = snap.meo_sma_km - _R_EARTH_KM             # from the 5.8 h period
    checks.append(
        Check(
            "MEO altitude (orbit-mean radius - R_e)",
            _fmt(alt_meo) + " km",
            _fmt(alt_meo_expected) + " km",
            f"+/-{TOLERANCES['meo_alt_km']:g} km",
            abs(alt_meo - alt_meo_expected) < TOLERANCES["meo_alt_km"],
            f"a = {snap.meo_sma_km:.3f} km from T = {snap.meo_period_h} h",
        )
    )

    # --- 10. MEO period -----
    meo_period_h = snap.meo_period_s / 3600.0
    checks.append(
        Check(
            "MEO orbital period",
            _fmt(meo_period_h, 4) + " h",
            _fmt(snap.meo_period_h, 4) + " h",
            f"+/-{TOLERANCES['period_h']:g} h",
            abs(meo_period_h - snap.meo_period_h) < TOLERANCES["period_h"],
            "from the built satrec's mean motion",
        )
    )

    # --- 11. in-plane ground-track neighbour separation -----
    seps: list[float] = []
    for p in range(snap.n_planes):
        idx = np.flatnonzero(snap.plane_id == p)                 # (sats_per_plane,)
        order = idx[np.argsort(snap.in_plane_idx[idx])]
        nxt = np.roll(order, -1)
        seps.extend(
            great_circle_km(
                snap.leo_lat[order], snap.leo_lon[order],
                snap.leo_lat[nxt], snap.leo_lon[nxt],
            ).tolist()
        )
    sep_arr = np.asarray(seps)                                   # (N_leo,)
    sep_mean = float(sep_arr.mean())
    # Adjacent seats are 360/sats_per_plane of mean anomaly apart, so their
    # sub-satellite points are the same central angle apart on the surface.
    sep_expected = 2.0 * math.pi * _R_EARTH_KM / snap.sats_per_plane
    checks.append(
        Check(
            "In-plane ground-track neighbour separation",
            _fmt(sep_mean, 2) + " km",
            _fmt(sep_expected, 2) + " km",
            f"+/-{TOLERANCES['ground_sep_frac'] * 100:g}%",
            abs(sep_mean - sep_expected) / sep_expected < TOLERANCES["ground_sep_frac"],
            f"per-pair range {sep_arr.min():.1f}-{sep_arr.max():.1f} km",
        )
    )

    # --- 12. MEO RAAN spacing -----
    raan_meo = np.sort(np.asarray(snap.meo_raan_deg))            # (N_meo,)
    meo_spacing = np.diff(raan_meo)                              # (N_meo-1,)
    meo_spacing_expected = 360.0 / snap.n_meo
    checks.append(
        Check(
            "MEO RAAN spacing",
            "/".join(_fmt(s, 1) for s in meo_spacing) + " deg",
            _fmt(meo_spacing_expected, 1) + " deg",
            f"+/-{TOLERANCES['raan_deg']:g} deg",
            bool(np.all(np.abs(meo_spacing - meo_spacing_expected) < TOLERANCES["raan_deg"])),
            "RAANs " + "/".join(_fmt(r, 1) for r in raan_meo) + " deg",
        )
    )

    # --- 13. MEO initial mean anomaly (informational) -----
    checks.append(
        Check(
            "MEO initial mean anomaly",
            "/".join(_fmt(m, 1) for m in snap.meo_mean_anom_deg) + " deg",
            "no config key",
            "n/a",
            True,
            "build_meo_tier hard-codes M=0 for every MEO; configs/base.yaml "
            "has no meo mean-anomaly field",
        )
    )

    # --- 14. globe orientation ---
    sub_gcrs = itrf_to_gcrs(
        geodetic_to_itrf(snap.leo_lat, snap.leo_lon, 0.0), snap.rot_itrf_to_gcrs
    )                                                            # (N_leo, 3)
    u = snap.leo_pos / np.linalg.norm(snap.leo_pos, axis=-1, keepdims=True)
    v = sub_gcrs / np.linalg.norm(sub_gcrs, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))   # (N_leo,)
    ang_max = float(ang.max())
    checks.append(
        Check(
            "Globe orientation vs sub-satellite points",
            _fmt(ang_max, 4) + " deg",
            "0 deg",
            f"<{TOLERANCES['globe_orientation_deg']:g} deg",
            ang_max < TOLERANCES["globe_orientation_deg"],
            "max angle between a satellite's radial direction and its own "
            "plotted subpoint; validates the ITRF->GCRS texture rotation",
        )
    )

    # --- 15. rivalry radius used by the footprint overlay -----
    # Closed form: cos(phi + theta) = R cos(theta)/(R + h), D = R phi.
    theta = math.radians(snap.elevation_mask_deg)
    phi = math.acos(_R_EARTH_KM * math.cos(theta) / (_R_EARTH_KM + snap.leo_alt_km)) - theta
    rivalry_expected = _R_EARTH_KM * phi
    checks.append(
        Check(
            f"Rivalry radius at {snap.elevation_mask_deg:g} deg mask",
            _fmt(snap.rivalry_radius_km, 1) + " km",
            _fmt(rivalry_expected, 1) + " km",
            f"+/-{TOLERANCES['rivalry_km']:g} km",
            abs(snap.rivalry_radius_km - rivalry_expected) < TOLERANCES["rivalry_km"],
            "from geometry.linkbudget.rivalry_radius_km",
        )
    )

    # --- 16. LEO->MEO assignment validity -----
    assign = snap.assignment                                     # (N_leo,)
    valid = bool(np.all((assign >= -1) & (assign < n_meo)))
    n_blind = int((assign == -1).sum())
    checks.append(
        Check(
            "LEO->MEO assignment range",
            f"[{assign.min()}, {assign.max()}]",
            f"within [-1, {n_meo - 1}]",
            "exact",
            valid,
            f"{n_blind} LEO(s) with no MEO in line of sight at this epoch",
        )
    )

    return checks


# --- Reporting ---


def format_table(checks: list[Check], snap: ConstellationSnapshot | None = None) -> str:
    """Render the checks as a fixed-width table.

    Parameters
    ----------
    checks : list[Check]  rows from run_checks
    snap   : ConstellationSnapshot | None  if given, a header with the epoch and
             config is added

    Returns
    -------
    str  the table, newline-terminated
    """
    headers = ("CHECK", "MEASURED", "EXPECTED", "TOL", "RESULT")
    rows = [
        (c.name, c.measured, c.expected, c.tolerance, "PASS" if c.passed else "FAIL")
        for c in checks
    ]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(5)
    ]

    def line(cells: tuple[str, ...], fill: str = " ") -> str:
        return "  ".join(c.ljust(widths[i], fill) for i, c in enumerate(cells)).rstrip()

    out: list[str] = []
    if snap is not None:
        out += [
            "Constellation verification",
            f"  config : {snap.config_path}"
            + (f" + {snap.override_path}" if snap.override_path else ""),
            f"  epoch  : {snap.epoch_iso}",
            f"  sats   : {snap.n_leo} LEO + {len(snap.constellation.meos)} MEO "
            f"= {snap.n_sats}",
            "",
        ]
    out.append(line(headers))
    out.append(line(tuple("-" * w for w in widths)))
    for c, r in zip(checks, rows):
        out.append(line(r))
        if c.note:
            out.append(f"      ({c.note})")

    n_pass = sum(c.passed for c in checks)
    out += ["", f"{n_pass}/{len(checks)} checks passed"]
    if n_pass != len(checks):
        out += [
            "",
            "FAILURES:",
            *(f"  - {c.name}: measured {c.measured}, expected {c.expected} ({c.tolerance})"
              for c in checks if not c.passed),
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point: build the snapshot, run the checks, print the table.

    Parameters
    ----------
    argv : list[str] | None  argument vector (defaults to sys.argv[1:])

    Returns
    -------
    int  0 if all checks passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Verify the constellation that viz/constellation_figure.py draws."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--override", type=Path, default=None)
    parser.add_argument(
        "--epoch", type=str, default=None, help="ISO-8601 UTC; default: sim.epoch from config"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="also write the table to this path"
    )
    args = parser.parse_args(argv)

    snap = build_snapshot(args.config, override_path=args.override, epoch_iso=args.epoch)
    checks = run_checks(snap)
    table = format_table(checks, snap)
    print(table, end="")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
        print(f"wrote {args.out}")

    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
