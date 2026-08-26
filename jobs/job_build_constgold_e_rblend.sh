#!/usr/bin/env bash
# Build ten immutable 10-case V3 R_blend lookup shards for ConstGold c40--139.

#SBATCH --job-name=sbsi_e_cgrb
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold
model=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
metadata=$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json
input_pattern=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather

first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
output=$root/rblend_v3_c${first}-${last}.feather
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
mapfile -t cases < <(seq "$first" "$last")
export PYTHONPATH=$repo:/home/z/Zekang.Zhang/blendemu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cpu
"$python" -u "$repo/scripts/build_constgold_blend_lookup.py" \
  --cases "${cases[@]}" --input-pattern "$input_pattern" \
  --measurement-model "$model" --emulator-metadata "$metadata" \
  --emulator-model "$emulator" --output "$output"
echo "CONSTGOLD_RBLEND_DONE cases=$first-$last output=$output"
