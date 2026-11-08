#!/usr/bin/env bash
# Where does the inference-time R_blend sum actually come from?
#
# The aperture audit says the sum runs over ~16.2 neighbour pairs per primary
# while only ~8.5 carry a label.  This bands the summed response by neighbour
# magnitude and by magnitude difference, on the ConstGold scenes and on the
# labelled training catalogue, to show how much of the global prediction sits
# outside the labelled support.

#SBATCH --job-name=sbsi_nbr_support
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/blend_neighbour_support_v3/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/blend_neighbour_support_v3/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/blend_neighbour_support_v3
model_root=$blend/fixed_g0_m258_r060_v2_baseline/models

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in 40 41 42 43 44; do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_constgold_blend_neighbour_support \
  "${cases[@]}" \
  --input-pattern "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --measurement-model "$v2/flow/nll/selected.pt" \
  --emulator-model "$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2_baseline.json" \
  --emulator-metadata "$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2_baseline.json" \
  --response-catalogue "$blend/rejection_relaxed/baseline_v1/response_catalogue_train.feather" \
  --catalogue-cases 20 --response-shear 0.2 \
  --output "$root/result.json" --device cpu

echo "NEIGHBOUR_SUPPORT_DONE output=$root/result.json"
