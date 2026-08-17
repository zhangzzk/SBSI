#!/bin/bash
#SBATCH --job-name=v30nstgt
#SBATCH --array=0-1%2
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30nstgt_%A_%a.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
MAGS=(10 12)
NMAG=${MAGS[$TASK]:?unknown task}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
OUT=results/v30_rblend_mag${NMAG}_nosize_c16_target.npz
for file in "$D/det_meas_crowd_g0.05_val_full.feather" "$MAIN_RESULTS/g0_lookup_c0-99.feather"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V3.0 ${NMAG}x1x16 CONDITIONAL R_BLEND TARGET — NO TRAINING ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend --crowd-conditional --n-crowd 16 \
  --n-flux "$NMAG" --n-size 1 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --output "$OUT"
echo "V30_NOSIZE_TARGET_DONE nmag=$NMAG"; date
