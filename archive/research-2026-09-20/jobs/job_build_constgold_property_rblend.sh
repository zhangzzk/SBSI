#!/usr/bin/env bash
# Extend the certified v2 fixed-g0 R_blend lookup from cases90 through 139.

#SBATCH --job-name=sbsi_cgprop_rb
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_property_bias_c40_139_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_property_bias_c40_139_v1/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
root=$cache/fixed_g0_m258_r060_v2/constgold_property_bias_c40_139_v1
anchor=$cache/fixed_g0_m258_r060_v2/domain/flow/g0/case{case:03d}.npz
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather
loader=$cache/fixed_g0_m258_r060_v2/flow/nll/selected.pt
model_root=$blend/fixed_g0_m258_r060_v2/models
model=$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2.json
metadata=$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json

first=$((90 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
output=$root/rblend/blocks/cases${first}_${last}.feather
mkdir -p "$root/logs" "$root/rblend/blocks"
mapfile -t cases < <(seq "$first" "$last")

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern "$truth" \
  --anchor-pattern "$anchor" \
  --measurement-model "$loader" \
  --emulator-model "$model" \
  --emulator-metadata "$metadata" \
  --output "$output" \
  --device cpu

echo "CONSTGOLD_PROPERTY_RBLEND_BLOCK_DONE cases=$first-$last output=$output"
