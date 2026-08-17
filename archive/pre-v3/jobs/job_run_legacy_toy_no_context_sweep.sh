#!/bin/bash
#SBATCH --job-name=toy_nc_sw
#SBATCH --array=0-11%12
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CONFIG_ID=${SLURM_ARRAY_TASK_ID:?array task required}
NREAL=${TOY_NREAL:-300}
OUTDIR=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/legacy_toy_no_context_exact_v1/sweep}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUTDIR"
DRAW=$OUTDIR/draws_config$(printf '%02d' "$CONFIG_ID").feather
AUDIT=$OUTDIR/audit_config$(printf '%02d' "$CONFIG_ID").json
for output in "$DRAW" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
"$PY" -u scripts/run_legacy_toy_no_context.py \
  --suite sweep --config-id "$CONFIG_ID" --nreal "$NREAL" \
  --g 0.05 --sky 10 --stamp 96 \
  --output-draws "$DRAW" --output-json "$AUDIT"
test -s "$DRAW"
test -s "$AUDIT"
echo LEGACY_TOY_NO_CONTEXT_SWEEP_JOB_DONE
date
