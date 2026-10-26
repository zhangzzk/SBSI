#!/usr/bin/env bash
#SBATCH --job-name=sbsi_cgparent_rb
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/constgold_truth_parent_20260917_v1
models=/project/ls-gruen/users/zekang.zhang/blendemu_runs/fixed_g0_m258_r060_v2/models
first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
mapfile -t cases < <(seq "$first" "$last")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern '/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather' \
  --anchor-pattern "$root/parent_v2/case{case:03d}.npz" \
  --measurement-model "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --emulator-model "$models/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  --output "$root/rblend/cases${first}_${last}.feather" --device cpu \
  --population-description 'complete target-role truth parent, strict r_input_p<26; no g0 measurement requirement; fixed-g0-trained emulator extrapolated beyond its fit cohort'
