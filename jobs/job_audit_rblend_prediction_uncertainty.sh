#!/usr/bin/env bash
# Case-bootstrap uncertainty of frozen V3 R_blend means on training and ConstGold.

#SBATCH --job-name=sbsi_rb_unc
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
source_cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
model=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
constgold=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold/constgold_response_E_s501_c40-139.json
output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/rblend_prediction_uncertainty_v3.json

[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cpu
cd "$repo"
"$python" -u scripts/audit_rblend_prediction_uncertainty.py \
  --source-cache "$source_cache" --model "$model" \
  --constgold-result "$constgold" --output "$output" \
  --n-boot 100000 --bootstrap-seed 20260824
