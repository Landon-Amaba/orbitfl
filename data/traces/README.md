# Traces

A trace holds the orbital geometry of one constellation: satellite positions,
sub-satellite points, visibility and the LEO-to-MEO assignment at every
timestep (15 s steps over 24 h from 2026-09-01T00:00:00Z). It contains no
spectrum data. The occupancy world, the model and the FL settings are applied
when an experiment runs, so one trace serves many experiments.

| Trace | Satellites | Used by | In the repository |
|---|---|---|---|
| `leo30` | 30 LEO (6 x 5), 4 MEO | 30-satellite concept-shift sweeps | yes (20 MB) |
| `leo180` | 180 LEO (12 x 15), 4 MEO | all other experiments | no (265 MB), regenerate |
| `leo180_meo3` | 180 LEO, 3 MEO | centralized baseline | no (255 MB), regenerate |

## Generate

Run from the repository root. `leo180` takes about 15 seconds.

```bash
python scripts/make_trace.py --config configs/base.yaml --out data/traces/leo180

python scripts/make_trace.py --config configs/base.yaml \
    --override configs/constellation/leo30.yaml --out data/traces/leo30

python scripts/make_trace.py --config configs/base.yaml \
    --override configs/constellation/leo180_meo3.yaml --out data/traces/leo180_meo3
```

Each command writes `positions.parquet`, `visibility.parquet`,
`assignment.parquet` and `meta.parquet`, and prints the metadata row.

## Identifying a trace

`meta.parquet` records a `config_hash` (hash of the full resolved config) and
the git hash of the code that generated the trace. Every run copies these into
its `trace_meta.yaml`.

| Trace | `config_hash` when generated with this repository | `config_hash` in the shipped runs |
|---|---|---|
| `leo30` | `e2bebd453e617549` | `e2bebd453e617549` |
| `leo180` | `bd81dad77be02421` | `c3a976f68be11e35` |
| `leo180_meo3` | `557d4239e7358bc8` | `d492a8d75aec6361` |

The hashes of the two 180-satellite traces differ because the traces behind the
shipped runs were generated before several options were added to the config
schema, and the hash covers every config field. The geometry is the same:
a `leo180` trace generated with this repository matches the one used for the
shipped runs to within 1e-12 km in every position, with identical assignment
and visibility tables.

To tell which trace a run used, look at `trace_meta.yaml` and at
`n_participating` in its `metrics.parquet` (at most 30 or 180).

Regenerating a trace on another machine reproduces positions to floating-point
precision (differences around 1e-12 km), not bit for bit.
