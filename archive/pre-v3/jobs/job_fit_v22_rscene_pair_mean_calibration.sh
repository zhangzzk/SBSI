#!/bin/bash
#SBATCH --job-name=v22_rscene_cal
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
FIT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_validation_tail_scenes_c40-199.feather
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
ANCHOR=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUT=$ROOT/results/v22_rscene_pair_mean_calibration_v2_c0-39_anchor_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for input in "$FIT" "$CACHE/metadata.json" "$ANCHOR"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in json md png pdf fit_bins.csv external_bins.csv anchor_bins.csv strengths.csv model.json; do
  test ! -e "$OUT.$suffix" || { echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

cd "$ROOT"
python -u scripts/fit_v22_rscene_pair_mean_calibration.py \
  --fit-scenes "$FIT" \
  --cache "$CACHE" \
  --anchor-features "$ANCHOR" \
  --n-quantile-bins 20 \
  --output-prefix "$OUT"
echo V22_RSCENE_CALIBRATION_JOB_DONE
