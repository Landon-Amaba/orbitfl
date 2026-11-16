"""Checkpoint B: is the interference graph of the constellation dense enough?

Builds the LEO interference graph at 20 timesteps spread over one day and
reports, per timestep, the mean degree and the fraction of nodes in a connected
component of at least 3 nodes. The check passes if every sampled timestep has
mean degree >= 1.8 and at least 55% of nodes in such a component. The thesis
constellation (12 planes x 15 satellites, 12 deg elevation mask) gives a mean
degree of 1.88 to 1.91 and a fraction of 0.59 to 0.60.

Plots are written to results/checkpoint_b/.

Usage:
  python scripts/checkpoint_b_interference_graph.py
  python scripts/checkpoint_b_interference_graph.py configs/constellation/leo30.yaml

The optional argument is an override YAML; its constellation is checked in
addition to the one in configs/base.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from skyfield.api import load as skyfield_load

from orbitfl.allocation.igraph import build_interference_graph  # single source of truth
from orbitfl.config import load_config
from orbitfl.geometry.footprint import footprint_radius_km
from orbitfl.geometry.linkbudget import (
    BW_HZ, EIRP_DBW, FREQ_GHZ, G_RX_DBI, NOISE_TEMP_K,
    rivalry_radius_km,
)
from orbitfl.geometry.orbits import _SECONDS_PER_DAY, build_walker_delta, propagate

# --- Pass thresholds, applied to every sampled timestep ---

N_SAMPLES: int = 20
MEAN_DEG_THRESHOLD: float = 1.8
LARGE_COMPONENT_MIN_SIZE: int = 3
LARGE_COMPONENT_FRAC: float = 0.55

_TS = skyfield_load.timescale(builtin=True)


def _build_geometry(eph: object, t_idx: int) -> dict:
    """Pack one timestep into the geometry dict expected by linkbudget."""
    return {
        "ecef": eph.ecef[t_idx],           # (N, 3) km  - satellite ECEF positions
        "lat": eph.subpoint_lat[t_idx],    # (N,)       - sub-satellite latitude  deg
        "lon": eph.subpoint_lon[t_idx],    # (N,)       - sub-satellite longitude deg
    }


def _build_graph(
    geo: dict,
    threshold_db: float,
    elevation_mask_deg: float,
) -> nx.Graph:
    """Build the interference graph with orbitfl.allocation.igraph."""
    return build_interference_graph(geo, threshold_db, elevation_mask_deg)


def _passes_density_threshold(mean_degs: list[float], comp_fracs: list[float]) -> bool:
    """True iff every sampled timestep satisfies the pass condition."""
    return (
        all(d >= MEAN_DEG_THRESHOLD for d in mean_degs)
        and all(f >= LARGE_COMPONENT_FRAC for f in comp_fracs)
    )


def run_density_check(
    label: str,
    leos: list,
    threshold_db: float,
    elevation_mask_deg: float,
    verbose: bool = True,
) -> dict:
    """Propagate the satellites to N_SAMPLES timesteps and collect graph statistics.

    Parameters
    ----------
    label             : str   display label for this run
    leos              : list  EarthSatellite objects from build_walker_delta
    threshold_db      : float INR threshold (allocation.threshold_db)
    elevation_mask_deg: float elevation mask in degrees (allocation.elevation_mask_deg)
    verbose           : bool  if False, do not print per-timestep lines

    Returns
    -------
    dict  with keys: label, n_leo, mean_degs, comp_fracs, all_degrees,
          all_comp_sizes, passed
    """
    N = len(leos)
    if verbose:
        rrk = rivalry_radius_km(550.0, elevation_mask_deg)
        print(f"\n{'='*62}")
        print(f"Checkpoint B: {label} ({N} LEOs)")
        print(f"  Link budget: {FREQ_GHZ} GHz, EIRP={EIRP_DBW} dBW, "
              f"G_rx={G_RX_DBI} dBi, T={NOISE_TEMP_K} K, B={BW_HZ/1e6:.0f} MHz")
        print(f"  INR threshold: {threshold_db} dB   "
              f"elevation mask: {elevation_mask_deg}°  "
              f"rivalry radius: {rrk:.0f} km")
        print(f"  Pass (per timestep): mean_deg ≥ {MEAN_DEG_THRESHOLD} AND "
              f"≥{LARGE_COMPONENT_FRAC*100:.0f}% nodes in component ≥{LARGE_COMPONENT_MIN_SIZE}")
        print(f"{'='*62}")

    epoch_tt = _TS.utc(2026, 9, 1, 0, 0, 0).tt
    offsets_s = np.linspace(1800.0, 23.0 * 3600.0, N_SAMPLES)
    times = _TS.tt_jd(epoch_tt + offsets_s / _SECONDS_PER_DAY)

    if verbose:
        print(f"  Propagating {N} LEOs × {N_SAMPLES} timesteps ...", flush=True)
    eph = propagate(leos, times)

    mean_degs: list[float] = []
    comp_fracs: list[float] = []
    all_degrees: list[int] = []
    all_comp_sizes: list[int] = []

    for t in range(N_SAMPLES):
        geo = _build_geometry(eph, t)
        G = _build_graph(geo, threshold_db, elevation_mask_deg)

        degrees = [d for _, d in G.degree()]
        components = [len(c) for c in nx.connected_components(G)]
        mean_d = float(np.mean(degrees))
        large_nodes = sum(s for s in components if s >= LARGE_COMPONENT_MIN_SIZE)
        frac = large_nodes / N if N > 0 else 0.0

        mean_degs.append(mean_d)
        comp_fracs.append(frac)
        all_degrees.extend(degrees)
        all_comp_sizes.extend(components)

        if verbose:
            ok = mean_d >= MEAN_DEG_THRESHOLD and frac >= LARGE_COMPONENT_FRAC
            print(f"  [{'OK' if ok else '!!'}] t={t+1:02d}  mean_deg={mean_d:5.2f}  "
                  f"large_frac={frac:.2f}  edges={G.number_of_edges():3d}  "
                  f"components={len(components)}")

    passed = _passes_density_threshold(mean_degs, comp_fracs)
    if verbose:
        verdict = "PASS" if passed else "FAIL"
        print(f"\n  -> {verdict}  "
              f"min_mean_deg={min(mean_degs):.2f}  "
              f"min_comp_frac={min(comp_fracs):.2f}")

    return dict(
        label=label,
        n_leo=N,
        mean_degs=mean_degs,
        comp_fracs=comp_fracs,
        all_degrees=all_degrees,
        all_comp_sizes=all_comp_sizes,
        passed=passed,
    )


def make_plots(results: list[dict], out_dir: Path) -> None:
    """Save the three plots to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff"]
    dark, grid_c, text_c = "#0d1117", "#30363d", "white"

    # Plot 1: mean degree per sampled timestep
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(dark)
    ax.set_facecolor(dark)
    for i, r in enumerate(results):
        ax.plot(range(1, N_SAMPLES + 1), r["mean_degs"],
                color=colors[i % len(colors)], lw=1.5, marker="o", ms=3,
                label=r["label"])
    ax.axhline(MEAN_DEG_THRESHOLD, color="#ffa657", lw=1.2, ls="--",
               label=f"threshold ({MEAN_DEG_THRESHOLD})")
    ax.set_xlabel("Sample timestep", color=text_c)
    ax.set_ylabel("Mean degree", color=text_c)
    ax.set_title("Mean interference-graph degree", color=text_c)
    ax.tick_params(colors=text_c)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid_c)
    ax.yaxis.grid(True, color=grid_c, lw=0.5)
    ax.legend(labelcolor=text_c, framealpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_degree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: degree distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(dark)
    ax.set_facecolor(dark)
    max_deg = max((max(r["all_degrees"], default=0) for r in results), default=0)
    bins = list(range(0, max_deg + 3))
    for i, r in enumerate(results):
        ax.hist(r["all_degrees"], bins=bins, alpha=0.7,
                color=colors[i % len(colors)], label=r["label"], edgecolor=dark)
    ax.set_xlabel("Degree", color=text_c)
    ax.set_ylabel("Count (all timesteps)", color=text_c)
    ax.set_title("Degree distribution", color=text_c)
    ax.tick_params(colors=text_c)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid_c)
    ax.legend(labelcolor=text_c, framealpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: component size distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(dark)
    ax.set_facecolor(dark)
    max_comp = max((max(r["all_comp_sizes"], default=0) for r in results), default=0)
    bins_c = list(range(1, max_comp + 2))
    for i, r in enumerate(results):
        ax.hist(r["all_comp_sizes"], bins=bins_c, alpha=0.7,
                color=colors[i % len(colors)], label=r["label"], edgecolor=dark)
    ax.axvline(LARGE_COMPONENT_MIN_SIZE - 0.5, color="#ffa657", lw=1.2, ls="--",
               label=f"size ≥ {LARGE_COMPONENT_MIN_SIZE} threshold")
    ax.set_xlabel("Component size (nodes)", color=text_c)
    ax.set_ylabel("Count (all timesteps)", color=text_c)
    ax.set_title("Connected-component size distribution", color=text_c)
    ax.tick_params(colors=text_c)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid_c)
    ax.legend(labelcolor=text_c, framealpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "component_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved -> {out_dir}/")
    for name in ["mean_degree.png", "degree_distribution.png", "component_sizes.png"]:
        print(f"  {name}")


if __name__ == "__main__":
    base_cfg_path = _REPO / "configs" / "base.yaml"
    out_dir = _REPO / "results" / "checkpoint_b"

    positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    override_path = Path(positional_args[0]) if positional_args else None

    cfg = load_config(base_cfg_path)
    threshold_db = cfg.allocation.threshold_db
    elevation_mask_deg = cfg.allocation.elevation_mask_deg
    epoch_t = _TS.utc(2026, 9, 1, 0, 0, 0)
    lc = cfg.constellation.leo

    rrk = rivalry_radius_km(lc.alt_km, elevation_mask_deg)
    fp_r = footprint_radius_km(lc.alt_km, cfg.world.beam_half_cone_deg)
    print(f"Rivalry radius:   {rrk:.0f} km  "
          f"(alt={lc.alt_km} km, mask={elevation_mask_deg}°)")
    print(f"Footprint radius: {fp_r:.1f} km  "
          f"(alt={lc.alt_km} km, half_cone={cfg.world.beam_half_cone_deg}°)")

    results = []

    # Constellation from configs/base.yaml
    leos_base = build_walker_delta(
        lc.n_planes, lc.sats_per_plane, lc.alt_km, lc.inc_deg, lc.phasing_f, epoch_t,
    )
    r_base = run_density_check(
        f"Walker {lc.n_planes}×{lc.sats_per_plane}={len(leos_base)} LEOs "
        f"(base, mask={elevation_mask_deg}°)",
        leos_base, threshold_db, elevation_mask_deg,
    )
    results.append(r_base)

    if override_path is not None:
        cfg_ov = load_config(base_cfg_path, override_path)
        lc_ov = cfg_ov.constellation.leo
        leos_ov = build_walker_delta(
            lc_ov.n_planes, lc_ov.sats_per_plane, lc_ov.alt_km, lc_ov.inc_deg,
            lc_ov.phasing_f, epoch_t,
        )
        r_ov = run_density_check(
            f"Walker {lc_ov.n_planes}×{lc_ov.sats_per_plane}={len(leos_ov)} LEOs "
            f"({override_path.stem}, mask={elevation_mask_deg}°)",
            leos_ov, threshold_db, elevation_mask_deg,
        )
        results.append(r_ov)

    make_plots(results, out_dir)

    print("\n" + "=" * 62)
    print("Checkpoint B summary")
    print("=" * 62)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['label'][:55]:<55}  {status}")

    if not any(r["passed"] for r in results):
        print("\nAll configurations FAILED.")
        sys.exit(1)
    else:
        print("\nCheckpoint B passed.")
        sys.exit(0)
