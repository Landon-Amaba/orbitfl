r"""Generate a trace (satellite positions, visibility and LEO-to-MEO assignment).

Usage:
  # 180-satellite constellation of configs/base.yaml
  python scripts/make_trace.py --config configs/base.yaml --out data/traces/leo180

  # 30-satellite constellation
  python scripts/make_trace.py --config configs/base.yaml \
      --override configs/constellation/leo30.yaml --out data/traces/leo30

Prints the metadata row of the trace when done, including its config_hash.
"""

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable when run from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbitfl.config import load_config
from orbitfl.geometry.orbits import build_meo_tier, build_walker_delta
from orbitfl.geometry.orbits import Constellation
from orbitfl.trace.generate import TraceGenerator
from orbitfl.trace.reader import Trace
from skyfield.api import load

_TS = load.timescale(builtin=True)


def _parse_epoch(epoch_str: str):
    """Convert an ISO-8601 UTC string to a skyfield Time."""
    from datetime import datetime
    dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
    return _TS.from_datetime(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an orbitfl trace.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to base config YAML (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--override",
        type=Path,
        default=None,
        help="Optional override YAML deep-merged over --config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for trace parquet files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.override)
    sim = cfg.sim
    leo_cfg = cfg.constellation.leo
    meo_cfg = cfg.constellation.meo

    T = sim.duration_h * 3600 // sim.timestep_s
    N_leo = leo_cfg.n_planes * leo_cfg.sats_per_plane
    N_meo = meo_cfg.n

    print(f"Config  : {args.config}" + (f" + {args.override}" if args.override else ""))
    print(f"Out     : {args.out}")
    print(f"Epoch   : {sim.epoch}")
    print(f"Steps   : T={T}  (dt={sim.timestep_s}s, dur={sim.duration_h}h)")
    print(f"LEOs    : {N_leo}  ({leo_cfg.n_planes} planes × {leo_cfg.sats_per_plane})")
    print(f"MEOs    : {N_meo}")
    print()

    epoch_sf = _parse_epoch(sim.epoch)

    print("Building constellation ... ", end="", flush=True)
    t0 = time.perf_counter()

    if cfg.constellation.source == "walker":
        leos = build_walker_delta(
            n_planes=leo_cfg.n_planes,
            sats_per_plane=leo_cfg.sats_per_plane,
            alt_km=leo_cfg.alt_km,
            inc_deg=leo_cfg.inc_deg,
            phasing_f=leo_cfg.phasing_f,
            epoch=epoch_sf,
        )
    else:
        raise NotImplementedError(
            f"constellation.source='{cfg.constellation.source}' not supported in make_trace.py; "
            "only 'walker' is implemented."
        )

    meos = build_meo_tier(
        n=meo_cfg.n,
        period_hours=meo_cfg.period_h,
        inc_deg=meo_cfg.inc_deg,
        epoch=epoch_sf,
    )

    # Satellite k belongs to plane k // sats_per_plane
    import numpy as np
    plane_id = np.repeat(
        np.arange(leo_cfg.n_planes, dtype=np.int32), leo_cfg.sats_per_plane
    )
    in_plane_idx = np.tile(
        np.arange(leo_cfg.sats_per_plane, dtype=np.int32), leo_cfg.n_planes
    )
    constellation = Constellation(
        leos=leos, meos=meos, plane_id=plane_id, in_plane_idx=in_plane_idx
    )
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print("Generating trace ... ", end="", flush=True)
    t0 = time.perf_counter()
    gen = TraceGenerator(cfg, constellation)
    gen.run(args.out)
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s)")

    # Print the metadata row
    trace = Trace.load(args.out)
    print()
    print("Meta:")
    print(trace.meta.to_string(index=False))
    print()

    churn = trace.churn_rate_per_orbit()
    print(f"Churn rate : {churn:.4f} handovers / satellite / orbit")
    print(f"Trace path : {args.out}")


if __name__ == "__main__":
    main()
