#!/bin/bash
#SBATCH --job-name=v30tailtgt
#SBATCH --array=0-1%2
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30tailtgt_%A_%a.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
NAMES=(tailq10 tailq12)
QUANTILES=(
  '0,.05,.125,.25,.375,.5,.625,.75,.875,.95,1'
  '0,.035,.07,.125,.25,.375,.5,.625,.75,.875,.93,.965,1'
)
NAME=${NAMES[$TASK]:?unknown task}
Q=${QUANTILES[$TASK]:?unknown task}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
OUT=results/v30_rblend_${NAME}_6x4_target.npz
for file in "$D/det_meas_crowd_g0.05_val_full.feather" "$MAIN_RESULTS/g0_lookup_c0-99.feather"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V3.0 6x4 CONDITIONAL R_BLEND TARGET name=$NAME quantiles=$Q — NO TRAINING ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend --crowd-conditional --crowd-quantiles "$Q" \
  --n-flux 6 --n-size 4 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --output "$OUT"
echo "V30_TAIL_TARGET_DONE name=$NAME"; date
