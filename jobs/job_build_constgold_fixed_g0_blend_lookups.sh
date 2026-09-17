#!/usr/bin/env bash
# Build old-v1 and corrected-v2 fixed-g0 R_blend blocks on cases40--89.

#SBATCH --job-name=sbsi_cgfg0_rb
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
root=$cache/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
anchor=$cache/fixed_g0_m258_r060_v2/domain/flow/g0/case{case:03d}.npz
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather
loader=$cache/fixed_g0_m258_r060_v2/flow/nll/selected.pt

first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
mkdir -p "$root/logs" "$root/rblend_v1/blocks" "$root/rblend_v2/blocks"
mapfile -t cases < <(seq "$first" "$last")

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

for version in v1 v2; do
  model_root=$blend/fixed_g0_m258_r060_${version}/models
  model=$model_root/regression_model_lsst_r_fixed_g0_m258_r060_${version}.json
  metadata=$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_${version}.json
  output=$root/rblend_${version}/blocks/cases${first}_${last}.feather
  "$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
    --cases "${cases[@]}" \
    --input-pattern "$truth" \
    --anchor-pattern "$anchor" \
    --measurement-model "$loader" \
    --emulator-model "$model" \
    --emulator-metadata "$metadata" \
    --output "$output" \
    --device cpu
done

echo "CONSTGOLD_FIXED_G0_RBLEND_BLOCK_DONE cases=$first-$last"
