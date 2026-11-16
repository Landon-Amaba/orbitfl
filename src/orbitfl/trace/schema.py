"""Parquet table schemas and the version string for orbitfl traces.

TRACE_VERSION must be bumped on any change to the schema or its meaning;
Trace.load() and the golden-trace regression test both check it.

A trace directory holds four parquet files written by TraceGenerator:
  positions  : t_idx, time_utc, sat_id, kind (leo|meo),
               x, y, z [km ECEF], lat, lon [deg], alt_km, plane_id
  visibility : t_idx, a, b, kind (isl|leo_meo), distance_km, rate_bps
  assignment : t_idx, leo_id, meo_id  (meo_id = -1 means no aggregator in view)
  meta       : single row with trace_version, config_hash, git_hash,
               timestep_s, duration_s, epoch, seed
"""

import pyarrow as pa

TRACE_VERSION: str = "1.0"

SCHEMA_POSITIONS: pa.Schema = pa.schema(
    [
        pa.field("t_idx", pa.int32()),
        pa.field("time_utc", pa.timestamp("s", tz="UTC")),
        pa.field("sat_id", pa.int32()),
        pa.field("kind", pa.string()),       # "leo" | "meo"
        pa.field("x", pa.float64()),         # km ECEF
        pa.field("y", pa.float64()),         # km ECEF
        pa.field("z", pa.float64()),         # km ECEF
        pa.field("lat", pa.float64()),       # sub-satellite latitude, degrees
        pa.field("lon", pa.float64()),       # sub-satellite longitude, degrees
        pa.field("alt_km", pa.float64()),    # geocentric altitude above WGS-84 surface, km
        pa.field("plane_id", pa.int32()),    # orbital plane index; -1 for MEOs
    ]
)

SCHEMA_VISIBILITY: pa.Schema = pa.schema(
    [
        pa.field("t_idx", pa.int32()),
        pa.field("a", pa.int32()),           # sat_id of first satellite
        pa.field("b", pa.int32()),           # sat_id of second satellite (b > a for ISL)
        pa.field("kind", pa.string()),       # "isl" | "leo_meo"
        pa.field("distance_km", pa.float64()),
        pa.field("rate_bps", pa.float64()),  # Shannon capacity at provisional link budget
    ]
)

SCHEMA_ASSIGNMENT: pa.Schema = pa.schema(
    [
        pa.field("t_idx", pa.int32()),
        pa.field("leo_id", pa.int32()),
        pa.field("meo_id", pa.int32()),      # MEO sat_id; -1 = no aggregator in view
    ]
)

SCHEMA_META: pa.Schema = pa.schema(
    [
        pa.field("trace_version", pa.string()),
        pa.field("config_hash", pa.string()),  # SHA-256 prefix of serialised config YAML
        pa.field("git_hash", pa.string()),     # "nogit" if not in a git repo
        pa.field("timestep_s", pa.int32()),
        pa.field("duration_s", pa.int64()),
        pa.field("epoch", pa.string()),        # ISO-8601 UTC start time
        pa.field("seed", pa.int32()),
    ]
)
