#!/usr/bin/env bash
#SBATCH --job-name=anchorblend_primaries
#SBATCH --partition=cluster
#SBATCH --qos=normal
#SBATCH --account=ls-gruen
#SBATCH --constraint=x86-64-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_measured_cuts_v1
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_primaries_v1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI=/home/z/Zekang.Zhang/SBSI

# The production run's own source snapshot, so the emulator call is the same
# code that produced the published pooled response, not today's checkout.
export PYTHONPATH="$RUN/source_snapshot"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mode=${1:?mode is required: smoke or run}
case "$mode" in
  smoke) start=400; stop=402; out="$OUT/smoke" ;;
  run)
    task=${SLURM_ARRAY_TASK_ID:?array task is required}
    start=$((400 + 25 * task)); stop=$((start + 25)); out="$OUT/parts"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

exec "$PY" "$SBSI/scripts/dump_anchorblend_scene_primaries.py" \
  --manifest "$RUN/manifest.json" \
  --case-start "$start" --case-stop "$stop" \
  --output "$out"
