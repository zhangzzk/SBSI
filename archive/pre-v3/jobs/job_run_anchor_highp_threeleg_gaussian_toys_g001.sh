#!/bin/bash
#SBATCH --job-name=ab3g_g001
#SBATCH --array=0-11%12
#SBATCH --time=02:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CHECK_IDS=(0 1 2 3 16 17 18 19 32 33 34 35)
ARRAY_ID=${SLURM_ARRAY_TASK_ID:?array task required}
TEMPLATE_ID=${CHECK_IDS[$ARRAY_ID]}
PREFIX=$ROOT/results/anchor_highp_threeleg_gaussian_manifest_v22_c700-899
OUTDIR=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_highp_threeleg_gaussian_v22_g001_r200}
NREAL=${TOY_NREAL:-200}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUTDIR"
DRAW=$OUTDIR/draws_template$(printf '%02d' "$TEMPLATE_ID").feather
PAIR=$OUTDIR/pairs_template$(printf '%02d' "$TEMPLATE_ID").feather
AUDIT=$OUTDIR/audit_template$(printf '%02d' "$TEMPLATE_ID").json
for output in "$DRAW" "$PAIR" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
"$PY" -u scripts/run_anchor_highp_threeleg_gaussian_toy.py \
  --templates "$PREFIX.feather" --sources "$PREFIX.sources.feather" \
  --template-id "$TEMPLATE_ID" --nominal-snr 12 20 35 \
  --nreal "$NREAL" --g 0.01 --stamp 96 --pixel-rms 10 \
  --output-draws "$DRAW" --output-pairs "$PAIR" --output-json "$AUDIT"
test -s "$DRAW"
test -s "$PAIR"
test -s "$AUDIT"
echo ANCHOR_HIGHP_THREELEG_GAUSSIAN_G001_JOB_DONE
date
