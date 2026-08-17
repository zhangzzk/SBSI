#!/bin/bash
#SBATCH --job-name=posw_trv_x21
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/posw_trv_x21_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/posw_trv_x21_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
MASK_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_official_exact_v21_domain_v1
MASK=$MASK_DIR/fit_v21_primary_domain.npy
OLD_ALL=$ROOT/results/v22_halfshear_rblend_emulator_label_calibration_training_c40-199.json
TRAIN=$ROOT/results/v22_pair_residual_vs_prediction_weight_scan_official_training_exactv21_c40-199_overlay
VALIDATION=$ROOT/results/v22_pair_residual_vs_prediction_weight_scan_official_validation_exactv21_c40-199_overlay
COMPARISON=$ROOT/results/v22_pair_residual_vs_prediction_weight_scan_official_train_vs_validation_v21domain_c40-199
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "$ROOT"
if [[ ! -e "$MASK_DIR/metadata.json" ]]; then
  "$PY" -u scripts/build_v22_exact_v21_split_mask.py \
    --cache "$CACHE" \
    --old-all-reference "$OLD_ALL" \
    --output-dir "$MASK_DIR"
fi
"$PY" -u scripts/plot_positive_weighted_pair_residual_official_validation_overlay.py \
  --cache "$CACHE" \
  --case-min 40 --case-max 199 --n-bins 20 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --official-split training \
  --v21-primary-domain --v21-mask "$MASK" \
  --output-prefix "$TRAIN"
"$PY" -u scripts/plot_positive_weighted_pair_residual_official_validation_overlay.py \
  --cache "$CACHE" \
  --case-min 40 --case-max 199 --n-bins 20 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --official-split validation \
  --v21-primary-domain --v21-mask "$MASK" \
  --output-prefix "$VALIDATION"
"$PY" -u scripts/plot_official_train_validation_v21_comparison.py \
  --training-json "$TRAIN.json" \
  --validation-json "$VALIDATION.json" \
  --all-rows-reference "$OLD_ALL" \
  --output-prefix "$COMPARISON"
