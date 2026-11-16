#!/bin/bash
# Operating-point diagnostics for the MCFL threshold sweep: full replays of stored
# results/mcfl_threshold runs. The array indices are that sweep's task indices, so
# each task replays the stored run with the same index:
#   0-4   = d_th 1.0, R 20, seeds 0-4
#   30-34 = d_th 0.3, R 20, seeds 0-4
# Set the account, QOS, partition and environment path for your cluster, set
# SWEEP_JOB to the SLURM job id of the sweep (it is part of the stored run folder
# names), then submit from the repo root after `mkdir -p logs`.
#SBATCH --job-name=mcfl_threshold_diag
#SBATCH --array=0-4,30-34
#SBATCH --output=logs/mcfl_threshold_diag_%A_%a.out
#SBATCH --error=logs/mcfl_threshold_diag_%A_%a.err
#SBATCH --account=<group>
#SBATCH --qos=<group>
#SBATCH --partition=<partition>
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2

set -euo pipefail

module load conda
export PATH=/path/to/conda/envs/orbitfl/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SWEEP_JOB=j49621586
matches=(results/mcfl_threshold/*_"${SWEEP_JOB}-${SLURM_ARRAY_TASK_ID}")
if [ "${#matches[@]}" -ne 1 ] || [ ! -d "${matches[0]}" ]; then
    echo "Expected one source run for task ${SLURM_ARRAY_TASK_ID}, got: ${matches[*]}" >&2
    exit 1
fi

python scripts/diag_mcfl_threshold.py run --mode threshold \
    --source-run "${matches[0]}" \
    --trace data/traces/leo180 \
    --out results/
