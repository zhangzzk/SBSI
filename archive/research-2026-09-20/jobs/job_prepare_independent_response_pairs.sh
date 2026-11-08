#!/usr/bin/env bash
# Truth-only pair preparation; no independent measured response is evaluated.
#SBATCH --job-name=sbsi_independent_pairs
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --array=0-3%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/harmonized_secondary_20260917_v1
models=/project/ls-gruen/users/zekang.zhang/blendemu_runs/fixed_g0_m258_r060_v2/models
first=$((200 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
mapfile -t cases < <(seq "$first" "$last")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd /home/z/Zekang.Zhang/SBSI
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern '/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239/case{case}_0.05/real0/catalogues/input/gals_info_tile180.0_-0.5.feather' \
  --anchor-pattern "$root/independent_parent/case{case:03d}.npz" \
  --measurement-model "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --emulator-model "$models/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  --output "$root/independent_pair_lookups/cases${first}_${last}.feather" \
  --pair-output-root "$root/independent_pairs" --device cpu \
  --population-description 'complete generated secondary-role truth parent; strict true r<26; no observed-g0 or measured selection'
