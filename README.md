# orbitfl

Simulation code and results for the honors thesis *Ground-Track Region-Clustered
Federated Learning (GTRFL) for Spectrum Forecasting under Data Drift* (Landon
Amaba, University of Florida, 2026).

The simulator models a constellation of low Earth orbit (LEO) satellites that
train a spectrum-occupancy forecaster with hierarchical federated learning (FL),
using medium Earth orbit (MEO) satellites as aggregators. Each satellite observes
whichever ground region it is flying over, so the data a client sees changes as
it moves, in a way that is known in advance from its orbit. The thesis asks
whether clustering clients by the ground region under them, which costs nothing
to compute, can replace clustering by model similarity, which has to be
discovered during training.

Main results, all reproducible from this repository:

1. Efficiency. Region clustering reaches the same accuracy as similarity
   clustering, plane clustering and random clustering (balanced accuracy about
   0.595), and needs no cluster-discovery rounds, against 10 to 210 for
   similarity clustering.
2. Predictability contrast. Making some regions easier to forecast than others
   does not separate the strategies, whether or not the model sees its position.
3. Hidden concept shift. When the forecast target is inverted in half of the
   regions without any change in the observations, region clustering beats the
   best other partition by a margin that grows with the shift (+0.17 balanced
   accuracy at the largest shift, 180 satellites). The margin disappears when
   the model is given its position.

## Contents

- [Terminology](#terminology)
- [How a run works](#how-a-run-works)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [What a run folder contains](#what-a-run-folder-contains)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
  - [Level 1: figures and tables from the shipped runs](#level-1-figures-and-tables-from-the-shipped-runs)
  - [Level 2: run part of an experiment yourself](#level-2-run-part-of-an-experiment-yourself)
  - [Level 3: full replication on a cluster](#level-3-full-replication-on-a-cluster)
- [Checkpoints A, B and C](#checkpoints-a-b-and-c)
- [Seeds and determinism](#seeds-and-determinism)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [Licence and citation](#licence-and-citation)

## Terminology

The thesis and the code use different names for the four clustering strategies.
The code names appear in configs, in result files and in the tables that the
scripts print.

| Thesis | Config key (`fl.strategy.name`) | Key in tables and figures code | Clusters |
|---|---|---|---|
| GTRFL (proposed) | `ephemeris` with `use_meo_proximity: false` | `EphemerisRegion` | ground region under each satellite, known in advance |
| MCFL | `similarity` | `SimilarityReactive_R<R>` | found from model-update similarity; re-clustered every `R` rounds |
| Plane | `plane` | `PlaneClustering` | one per orbital plane, fixed |
| Random churn | `random_churn` | `RandomChurnClustering` | random, changing at the rate satellites change MEO |

Other terms:

| Term | Meaning |
|---|---|
| Trace | Orbital geometry of a constellation over 24 h: positions, visibility, LEO-to-MEO assignment. No spectrum data. |
| World | The simulated spectrum: per region and channel, a two-state (busy/idle) Markov chain. A satellite observes the region under it, with noise. |
| Horizon (`delta_steps`) | How many 15 s timesteps ahead the model forecasts. 4 in all experiments. |
| Round | One FL round, 300 s (20 timesteps). Every run has 300 rounds (`round_budget`); there is no early stopping. |
| Sync interval | Cluster models are averaged into a global model every `sync_interval` rounds. Between syncs each cluster keeps its own model. |
| Global / personalized accuracy | `ba_probe`: the global model on held-out data. `ba_personalized`: each satellite scored with its own cluster's model on the same data. |
| BA, AUC | Balanced accuracy at a 0.5 threshold, and area under the ROC curve. |
| Overhead rounds | Rounds a strategy spends discovering clusters. Zero for every strategy except `similarity`. |
| Predictability contrast | World option that makes some regions more predictable than others. Visible in the observations. |
| Concept shift | World option that inverts the forecast target on a fraction of channels in every second region. Not visible in the observations. |

## How a run works

```
configs/base.yaml + override        scripts/make_trace.py
          |                                  |
          v                                  v
   orbitfl.config.Config            data/traces/<name>/  (geometry)
          |                                  |
          +----------------+-----------------+
                           v
              orbitfl.world.OccupancyWorld      per-region Markov chains
                           |
                           v
              one SatDataset per satellite      (window, position, target)
                           |
                           v
              orbitfl.fl.loop.FederatedRunner   300 rounds
                |  membership from a ClusteringStrategy
                |  local training on each client
                |  FedAvg within clusters, FedAvg across clusters at each sync
                v
          results/<experiment>/<run>/metrics.parquet
```

1. Configuration. `orbitfl.config.load_config` reads `configs/base.yaml`, merges
   an override file on top, and returns a frozen `Config` dataclass. Unknown
   keys raise an error.
2. Trace. `scripts/make_trace.py` builds the satellites
   (`orbitfl.geometry.orbits`), propagates them with SGP4, computes visibility
   and the nearest visible MEO for every LEO (`orbitfl.geometry.visibility`),
   and writes four parquet tables (`orbitfl.trace`). A trace depends only on the
   constellation, so it is generated once and reused.
3. World. `orbitfl.world.OccupancyWorld` simulates the busy/idle state of every
   channel in every ground region. `observation_stream` replays a satellite's
   ground track through that world and returns a `SatDataset` of
   (observation window, position encoding, target) samples. The last 20% of each
   satellite's samples are held out as the probe set.
4. Model. `orbitfl.models.forecaster.OccupancyForecaster` is a small CNN over
   the last 16 observations, with an optional position input
   (`model.use_position`).
5. Clustering. A `ClusteringStrategy` (`orbitfl.fl.clustering`) returns a
   cluster id for every satellite in every round. `ephemeris`, `plane` and
   `random_churn` know their clusters without training; `similarity` spends
   `T_init` rounds collecting model updates before each clustering.
6. Training loop. `orbitfl.fl.loop.FederatedRunner` runs the rounds. Each
   participating client trains from its cluster's model
   (`orbitfl.fl.client`); updates are averaged within each cluster every round
   and across clusters every `sync_interval` rounds (`orbitfl.fl.aggregation`).
7. Output. `orbitfl.experiments.runner.run_experiment` creates a run folder and
   writes the resolved config, the trace metadata and the per-round metrics.

## Repository layout

```
configs/            base.yaml and the overrides merged on top of it
  constellation/    constellation overrides for the other two traces
  experiments/      settings for single scripts and tests
  sweeps/           one file per experiment grid (the thesis experiments)
  diagnostics/      settings for the MCFL threshold diagnostics
data/
  tle/              Starlink TLE snapshot (CelesTrak, 2 Sep 2026); used only by two tests
  traces/           leo30 trace; the 180-satellite traces are generated locally
figures/            the figures as they appear in the thesis
results/            the runs behind every figure and table
scripts/            command-line entry points
src/orbitfl/        the simulator package
tests/              pytest suite
```

`src/orbitfl/`, in the order a run uses it:

| Package | Contents | Start with |
|---|---|---|
| `config.py` | configuration dataclasses and YAML loading | `load_config` |
| `geometry/` | orbits and propagation, line of sight, footprints, link budget | `orbits.py` |
| `trace/` | trace generation, schema and reader | `reader.py` |
| `world/` | region definitions, occupancy chains, per-satellite datasets | `occupancy.py` |
| `models/` | the CNN forecaster and three simple baselines | `forecaster.py` |
| `fl/` | client, FedAvg aggregation, training loop | `loop.py` |
| `fl/clustering/` | the four clustering strategies | `base.py` |
| `metrics/` | forecast, communication and efficiency metrics | `forecast.py` |
| `experiments/` | run one experiment; expand a sweep | `runner.py` |
| `viz/` | constellation figure and checks on the plotted geometry | `constellation_figure.py` |
| `allocation/` | interference graph and channel allocation by graph colouring | `igraph.py` |
| `utils/` | seeding, run folders, device selection, plot style | |

`scripts/`:

| Script | Purpose |
|---|---|
| `make_trace.py` | generate a trace |
| `run_experiment.py` | run one experiment or one cell of a sweep |
| `slurm_submit.py` | turn a sweep file into a SLURM array job, or run it locally |
| `compare_arms.py` | compare strategies across the runs of an experiment; summary table and figures |
| `plot_contrast_pair.py` | the two-panel predictability-contrast figure |
| `train_centralized.py` | centralized (non-federated) baseline, Checkpoint C |
| `checkpoint_a_fedavg_equivalence.py` | Checkpoint A |
| `checkpoint_b_interference_graph.py` | Checkpoint B |
| `diag_mcfl_threshold.py`, `.sbatch` | diagnostics on the MCFL threshold sweep |

## Configuration

`configs/base.yaml` describes the 180-satellite constellation and the
four-region world of the thesis. Everything else is an override merged on top
of it. A sweep file is an override with an extra `sweep:` section that lists
the values to vary; the sweep is the Cartesian product of those lists, one run
per combination.

Always pass `configs/base.yaml` as the base config. The dataclass defaults in
`orbitfl/config.py` describe a different, smaller setup (for example a 20 degree
elevation mask where the thesis uses 12 degrees).

The settings that differ between experiments:

| Key | Meaning | Values used |
|---|---|---|
| `constellation.leo.n_planes`, `sats_per_plane` | constellation size | 12 x 15 (180), 6 x 5 (30) |
| `experiment.delta_steps` | forecast horizon in timesteps | 4 |
| `experiment.seeds` | seeds of the runs | 0-9 or 0-4, one per task |
| `fl.round_budget` | rounds per run | 300 |
| `fl.sync_interval` | rounds between global merges | 20; 300 for concept shift |
| `fl.strategy.name`, `fl.strategy.params` | clustering strategy and its parameters | see [Terminology](#terminology) |
| `fl.cluster_local_train`, `fl.per_sample_eval` | train and score by the cluster each sample was collected in | on for concept shift only |
| `world.predictability_contrast` | spread of predictability across regions | 0, 0.3, 0.6, 0.9 |
| `world.concept_shift` | fraction of channels with inverted targets in every second region | 0, 0.3, 0.6, 0.9 |
| `model.use_position` | whether the model sees its position | true or false |

The concept-shift experiments also replace the world with four equal longitude
bands that have the same occupancy and dynamics, so that the shift is the only
difference between regions (see the `world:` section of those sweep files).

## What a run folder contains

Each task of a sweep writes one folder,
`results/<experiment>/<UTC time>_<git hash>_<token>/`:

| File | Content |
|---|---|
| `config.yaml` | the fully resolved configuration of the run |
| `trace_meta.yaml` | metadata of the trace it used (`config_hash`, timestep, epoch) |
| `metrics.parquet` | one row per round and seed |
| `seed_XX/metrics.parquet` | the same, per seed |
| `summary.csv` | mean and standard deviation per round across the folder's seeds |
| `fl_aggregate.png`, `.pdf` | accuracy against round for this run |

Columns of `metrics.parquet`:

| Column | Meaning |
|---|---|
| `round` | round index, 0 to 299 |
| `n_participating`, `n_eligible` | clients that trained, and clients that could have |
| `n_cluster_changes` | satellites whose cluster changed since the previous round |
| `auc_probe`, `ba_probe` | global model on the probe set (NaN on rounds without evaluation) |
| `auc_personalized`, `ba_personalized` | per-cluster models on the same probe set |
| `comm_bytes_up`, `comm_bytes_down` | model bytes sent in the round |
| `overhead_rounds` | cumulative cluster-discovery rounds |
| `ari_vs_regions` | adjusted Rand index between the clusters and the regions under the satellites at t = 0 (see [Known limitations](#known-limitations)) |
| `n_clusters` | number of clusters in the round (only in runs made after the column was added) |
| `seed`, `strategy` | run seed and strategy class |

The shipped run folders are stored exactly as the cluster jobs wrote them, so
some labels inside them predate the folder names used here:

| In this repository | Label inside the run files |
|---|---|
| `results/efficiency` | `e1_drift_v3` |
| `results/contrast_position_hidden`, `_visible` | `e1_ablation`, `e1_contrast_v2` |
| `results/concept_shift_hidden_30`, `_visible_30` | `e1_concept_shift`, `e1_concept_shift_pos` |
| `results/concept_shift_hidden_180`, `_visible_180` | `e1_concept_shift_fullv2`, `e1_concept_shift_pos_fullv2` |
| `results/mcfl_threshold` | `e1_mcfl_dth` |
| traces `leo30`, `leo180`, `leo180_meo3` | `small_v1`, `full_v2`, `full_v1` |

Most folder names contain `nogit` because the cluster copy of the code was not a
git checkout. The short hashes that do appear refer to the private development
history and do not exist in this repository.

## Installation

Python 3.11 or later. With conda:

```bash
conda env create -f environment.yml
conda activate orbitfl
pip install -e .
```

Or with pip in any environment:

```bash
pip install -e ".[dev,full]"
```

A GPU is optional. Reproducing the figures from the shipped runs needs only
pandas and matplotlib and takes seconds. Training uses CUDA or Apple MPS when
available and otherwise the CPU.

## Reproducing the results

### Level 1: figures and tables from the shipped runs

`results/` holds every run behind the thesis figures and tables. The commands
below read those runs, print a summary table and write figures to
`figures/regenerated/`. Each takes a few seconds on a laptop. Run them from the
repository root.

| Thesis item | Command | Reads |
|---|---|---|
| Efficiency figure and table | `python scripts/compare_arms.py results/efficiency --out figures/regenerated/efficiency --sync 20 --target 0.58` | 76 runs: 10 seeds x 7 strategy variants, plus 6 repeated tasks |
| Predictability-contrast figure | `python scripts/plot_contrast_pair.py --out figures/regenerated` | `contrast_position_hidden` and `contrast_position_visible`, 60 runs each |
| Concept shift, position hidden | `python scripts/compare_arms.py results/concept_shift_hidden_180 --out figures/regenerated/concept_shift_hidden` | 60 runs |
| Concept shift, position visible | `python scripts/compare_arms.py results/concept_shift_visible_180 --out figures/regenerated/concept_shift_visible` | 59 runs |
| Constellation figure | `python -m orbitfl.viz.constellation_figure --config configs/base.yaml` | nothing (geometry only) |

`compare_arms.py` writes `efficiency.png`, `comparison.png` and, for the
concept-shift folders, `concept_shift.png` (plus PDF versions) into the `--out`
folder. The thesis uses `efficiency.png` from the first command and
`concept_shift.png` from the third and fourth. The copies in `figures/` are the
ones in the thesis. Regenerated figures show the same data; they can differ
from those copies in text rendering, depending on the Matplotlib version, and
the current code draws the key of the efficiency figure under its first panel
where the thesis copy has it under the whole figure.

The efficiency command should print this table (columns abbreviated here):

| Strategy | seeds | BA (global) | BA (personalized) | rounds to BA 0.58 | GB to BA 0.58 | overhead rounds |
|---|---|---|---|---|---|---|
| `EphemerisRegion` | 10 | 0.5951 | 0.5941 | 108 | 6.64 | 0 |
| `PlaneClustering` | 10 | 0.5969 | 0.5929 | 104 | 6.39 | 0 |
| `RandomChurnClustering` | 10 | 0.5951 | 0.5947 | 108 | 6.64 | 0 |
| `SimilarityReactive_Rinf` | 10 | 0.5947 | 0.5947 | 108 | 6.64 | 10 |
| `SimilarityReactive_R20` | 10 | 0.5947 | 0.5947 | 108 | 6.64 | 110 |
| `SimilarityReactive_R10` | 10 | 0.5947 | 0.5947 | 108 | 6.64 | 160 |
| `SimilarityReactive_R5` | 10 | 0.5947 | 0.5947 | 108 | 6.64 | 210 |

`compare_arms.py` prints one row per strategy averaged over everything in the
folder. For the concept-shift table, which is broken down by shift level, run:

```bash
python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import compare_arms as ca

for name in ["concept_shift_hidden_180", "concept_shift_visible_180",
             "concept_shift_hidden_30", "concept_shift_visible_30"]:
    runs = ca._discover_run_dirs(Path("results") / name)
    table = ca._per_run_table(ca._load_all(runs), ca.DEFAULT_TARGETS)
    means = table.groupby(["arm", "concept_shift"])["final_pers"].mean().unstack()
    gap = means.loc["EphemerisRegion"] - means.drop("EphemerisRegion").max()
    print(name)
    print(means.round(3).to_string())
    print("gap to best baseline:", gap.round(3).to_dict(), "\n")
EOF
```

Expected personalized balanced accuracy at round 299 (mean over 5 seeds):

| 180 satellites, position hidden | shift 0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| `EphemerisRegion` | 0.838 | 0.806 | 0.756 | 0.714 |
| `PlaneClustering` | 0.864 | 0.772 | 0.631 | 0.538 |
| `RandomChurnClustering` | 0.865 | 0.774 | 0.632 | 0.541 |
| gap to best baseline | -0.026 | +0.032 | +0.124 | +0.173 |

| Gap to best baseline | shift 0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| 180 satellites, position visible | -0.028 | -0.029 | -0.033 | -0.027 |
| 30 satellites, position hidden | -0.032 | +0.028 | +0.118 | +0.180 |
| 30 satellites, position visible | -0.035 | -0.037 | -0.042 | -0.036 |

Gaps are computed from unrounded means, so they can differ by 0.001 from the
difference of the rounded entries. One cell of the 180-satellite,
position-visible grid (shift 0.9, plane) has 4 seeds because one job produced no
output.

`plot_contrast_pair.py` prints the predictability-contrast numbers: in both
panels all three strategies are within 0.003 of each other at every contrast
level, falling together from about 0.594 at contrast 0 to about 0.583 at 0.9.

### Level 2: run part of an experiment yourself

This runs two cells of the 30-satellite concept-shift experiment (shift 0.9,
position hidden, seed 0): once with region clusters and once with plane
clusters. The 30-satellite trace is in the repository.

```bash
for strategy in ephemeris plane; do
  python scripts/run_experiment.py configs/base.yaml \
      --override configs/sweeps/concept_shift_hidden_30_sweep.yaml \
      --override-json "{\"world\": {\"concept_shift\": 0.9}, \"fl\": {\"strategy\": {\"name\": \"$strategy\"}}, \"experiment\": {\"seeds\": [0]}}" \
      --trace data/traces/leo30 --out runs/
done

python scripts/compare_arms.py runs/concept_shift_hidden_30 --out runs/figures
```

Each run takes roughly 20 to 40 minutes on a laptop; the rounds get slower as
they go, because clients train on all the data collected so far. Progress is
printed every 10 rounds. The summary table printed by `compare_arms.py` should
show a personalized balanced accuracy (`pers`) of about 0.70 to 0.73 for
`EphemerisRegion` and about 0.54 for `PlaneClustering`, with the global model
(`BA`) near 0.5 for both. The exact values depend on the machine (see
[Seeds and determinism](#seeds-and-determinism)); the shipped run of the same
two cells gives 0.726 and 0.540.

`run_experiment.py` refuses to run from a git checkout with uncommitted changes,
so that every result can be tied to a commit; add `--dirty-ok` to override. The
`runs/` folder is ignored by git.

To run a whole 30-satellite sweep locally, one task after another:

```bash
python scripts/slurm_submit.py configs/sweeps/concept_shift_hidden_30_sweep.yaml \
    --serial --trace data/traces/leo30 --out runs/
```

### Level 3: full replication on a cluster

1. Generate the 180-satellite trace (about 15 seconds) and check its hash:

   ```bash
   python scripts/make_trace.py --config configs/base.yaml --out data/traces/leo180
   ```

   The printed `config_hash` should be `bd81dad77be02421`. The shipped runs are
   stamped `c3a976f68be11e35`; `data/traces/README.md` explains why the hashes
   differ while the geometry is the same.

2. Submit the sweeps. Each sweep file expands to an array job with one seed per
   task.

   | Experiment | Sweep file(s) in `configs/sweeps/` | Tasks | Trace | Results folder | Per task (approx.) |
   |---|---|---|---|---|---|
   | Efficiency | `efficiency_sweep.yaml`, `efficiency_mcfl_sweep.yaml` | 30 + 40 | `leo180` | `efficiency` | 0.5-2 h, 1 GPU |
   | Predictability contrast, hidden | `contrast_position_hidden_sweep.yaml` | 60 | `leo180` | `contrast_position_hidden` | 2 h, 2 CPUs |
   | Predictability contrast, visible | `contrast_position_visible_sweep.yaml` | 60 | `leo180` | `contrast_position_visible` | 2 h, 2 CPUs |
   | Concept shift, hidden | `concept_shift_hidden_180_sweep.yaml` | 60 | `leo180` | `concept_shift_hidden_180` | several hours, 2 CPUs |
   | Concept shift, visible | `concept_shift_visible_180_sweep.yaml` | 60 | `leo180` | `concept_shift_visible_180` | several hours, 2 CPUs |
   | Concept shift at 30 satellites | `concept_shift_hidden_30_sweep.yaml`, `concept_shift_visible_30_sweep.yaml` | 60 + 60 | `leo30` | `concept_shift_hidden_30`, `_visible_30` | see Level 2 |
   | MCFL threshold (not in the thesis) | `mcfl_threshold_sweep.yaml` | 50 | `leo180` | `mcfl_threshold` | several hours, 2 CPUs |

   Example submission:

   ```bash
   python scripts/slurm_submit.py configs/sweeps/efficiency_sweep.yaml \
       --trace data/traces/leo180 --out results/ \
       --account <group> --qos <group> \
       --conda-env-path /path/to/conda/envs/orbitfl \
       --partition <partition> --gpus 1 --mem 32G --time 12:00:00 | sbatch
   ```

   `slurm_submit.py` writes `<sweep>_runs.json` next to the sweep file and
   prints a batch script to standard output. Run it on the cluster, from the
   repository root, so that the paths in the script exist on the compute nodes.
   Its default partition names are those of the University of Florida
   HiPerGator cluster; pass your own with `--partition`. The header of each
   sweep file gives the resources that were used. `--dry-run` lists the tasks
   without submitting anything.

3. Check that a grid is complete before plotting. A task that hits its time
   limit leaves a folder without `metrics.parquet`, or one that stops before
   round 299. `compare_arms.py` skips such runs and says so; the `n` column of
   its summary table shows how many seeds each strategy has. Re-submit missing
   tasks with `sbatch --array=<indices>`.

4. Plot with the Level 1 commands.

## Checkpoints A, B and C

Chapter 3 of the thesis describes three checks made before the experiments.

Checkpoint A: hierarchical FedAvg equals flat FedAvg when every client
participates, there is one local training phase per merge, and averages are
weighted by data size.

```bash
python scripts/checkpoint_a_fedavg_equivalence.py
```

Prints four checks and `4/4 checks passed`: the two rules agree to about 1e-16
when all three conditions hold and differ when any one is broken. The same
property is tested on the simulator's own aggregation code in
`tests/test_aggregation.py` and on the training loop in `tests/test_fl_loop.py`.

Checkpoint B: the interference graph of the constellation is dense enough.

```bash
python scripts/checkpoint_b_interference_graph.py
```

For the 180-satellite constellation with a 12 degree elevation mask it prints a
mean degree between 1.88 and 1.91 and a large-component fraction of 0.59 to
0.60 at each of 20 timesteps, and `Checkpoint B passed`. Plots go to
`results/checkpoint_b/`.

Checkpoint C: how the CNN forecaster compares with simple baselines when trained
centrally.

```bash
python scripts/make_trace.py --config configs/base.yaml \
    --override configs/constellation/leo180_meo3.yaml --out data/traces/leo180_meo3
python scripts/train_centralized.py --config configs/base.yaml \
    --override configs/experiments/centralized_baseline.yaml --seeds 0 1 2 \
    --trace data/traces/leo180_meo3 --out runs/
```

This trains the CNN 30 times (5 horizons, 3 seeds, with and without position)
for 30 epochs each on the data of all 180 satellites, which takes several hours. Add
`--delta-steps 4 --seeds 0 --epochs 2` for a quick trial. The run behind the
thesis numbers is in `results/centralized_baseline/`; to read it:

```bash
python - <<'EOF'
import glob
import pandas as pd

df = pd.read_parquet(glob.glob("results/centralized_baseline/*/metrics.parquet")[0])
cols = ["ba_cnn", "ba_pers", "ba_eph", "auc_cnn", "auc_pers", "auc_eph"]
print(df[df.use_position].groupby("delta_steps")[cols].mean().round(3))
EOF
```

At a horizon of 4 steps this gives a balanced accuracy of 0.592 for the CNN
against 0.658 for persistence, and an AUC of 0.670 for the CNN against 0.658
for persistence and 0.635 for the position-only model (`eph`).

## Seeds and determinism

- `experiment.seeds` lists the seeds of a run. A seed sets the world (the
  occupancy chains and observation noise), the model initialisation and
  training order, and any randomness in the clustering strategy. Sweeps give
  each task one seed.
- The top-level `seed` (5 in `base.yaml`) is recorded in the trace metadata and
  seeds the sampling of clients when `fl.participation_rate` is below 1. It
  does not change the world or the model, so sweeping it produces identical
  runs.
- The same code, trace, config, seed and machine give the same metrics.
  Across machines, PyTorch and BLAS builds differ in the last bits, and a
  regenerated trace matches to about 1e-12 km, so re-run metrics agree closely
  but not bit for bit. The summary tables above are computed from the shipped
  runs and reproduce exactly.
- The figure scripts are deterministic for a given Matplotlib version.

## Known limitations

- Similarity clustering formed a single cluster in the efficiency experiment.
  With the distance threshold used there (`d_th = 1.0`) all 180 satellites were
  placed in one cluster at every clustering, so its training was identical to
  unclustered FedAvg, and the four values of `R` give identical accuracy. Its
  overhead rounds were still spent. The efficiency runs predate the
  `n_clusters` column, so this is not visible in their metrics; it is shown by
  the threshold sweep in `results/mcfl_threshold` and its diagnostics
  (`python scripts/compare_arms.py results/mcfl_threshold --out figures/regenerated/mcfl_threshold`
  and `python scripts/diag_mcfl_threshold.py summarize --out results/`). There,
  thresholds of 0.5 and above give one cluster at the first clustering. Every
  setting that ends up with more than one cluster has lower personalized
  accuracy (0.579 to 0.594) than the single cluster (0.595). This sweep is not
  part of the thesis text.
- Under concept shift the region strategy is given the true partition. The
  shift is applied per region, and the regions are that strategy's clusters.
  The experiment shows that such a partition can be exploited when it is
  known, not that it would be found. `similarity` is not part of this
  experiment: the options it needs (`cluster_local_train`, `per_sample_eval`)
  require a membership schedule that is known before training.
- The global model in the concept-shift runs is a single final merge
  (`sync_interval` equals the round budget), so `ba_probe` there reflects how
  well very different cluster models average, not how well a shared model could
  be trained.
- `ari_vs_regions` compares each round's clusters with the regions under the
  satellites at t = 0. It is 1.0 for region clustering at round 0 by
  construction and then falls as the satellites move. It does not measure how
  well clusters follow geography over a run.
- The link-budget constants in `orbitfl/geometry/linkbudget.py` are placeholder
  values that were not taken from a published link budget. In the experiments
  they only enter through the interference graph, whose edges are decided by
  the elevation mask; the interference threshold never rejects a pair that
  passes the mask.
- `orbitfl/allocation/` (channel allocation by graph colouring, boundary
  reconciliation) and `orbitfl/metrics/allocation.py` are unit-tested but not
  used by any experiment; only `build_interference_graph` is used, by
  Checkpoint B. The two `allocation_*.yaml` files in `configs/experiments/` are
  placeholders for experiments that were not run.
- `Ephemeris.ecef` holds inertial (GCRS) coordinates despite its name. The
  simulator uses only distances and angles between satellites, which are
  unaffected; sub-satellite latitude and longitude are Earth-fixed.
- The dataclass defaults in `orbitfl/config.py` are not the thesis settings.
  Use `configs/base.yaml` as the base config.
- The centralized baseline was trained on the `leo180_meo3` trace (3 MEOs),
  while the federated experiments use `leo180` (4 MEOs). The LEO geometry and
  therefore the data are the same; only the aggregator assignment differs,
  which centralized training does not use.

## Tests

```bash
pytest
ruff check .
```

The suite has 270 tests and runs in about a minute on a laptop. One test is
skipped when the measured cluster-change rate of the tiny test trace is too low
to check.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). The Earth texture
used by the constellation figure is NASA Blue Marble imagery, in the public
domain (see `src/orbitfl/viz/assets/LICENSE.md`).

If you use this code, please cite the thesis:

> Landon Amaba. *Ground-Track Region-Clustered Federated Learning (GTRFL) for
> Spectrum Forecasting under Data Drift.* Honors thesis, Department of
> Electrical and Computer Engineering, University of Florida, 2026.

Citation metadata is in `CITATION.cff`.
