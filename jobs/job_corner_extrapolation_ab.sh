#!/usr/bin/env bash
# Corner accuracy for one arm of the bright-neighbour rejection A/B.
#
# Three arms answer three different questions:
#   base_on_base   reproduces the published 20-case result (sanity)
#   base_on_relax  the CERTIFIED emulator against labels it never had:
#                  is it wrong where it could not be checked?
#   relax_on_relax does retraining on the recovered labels fix it?
#
# Pass with --export=ALL,ARM=...,MODEL=baseline|relaxed,CAT=baseline|relaxed

#SBATCH --job-name=sbsi_corner_ab
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/corner_extrapolation_ab_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/corner_extrapolation_ab_v2/logs/%x_%j.err

set -euo pipefail
: "${ARM:?set ARM}"; : "${MODEL:?set MODEL}"; : "${CAT:?set CAT}"
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/corner_extrapolation_ab_v2

case "$CAT" in
  baseline) catalogue=$blend/rejection_relaxed/baseline_v1/response_catalogue_train.feather;;
  relaxed)  catalogue=$blend/rejection_relaxed/v1/response_catalogue_train.feather;;
  *) echo "bad CAT=$CAT" >&2; exit 2;;
esac
model_root=$blend/fixed_g0_m258_r060_v2_${MODEL}/models
model=$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2_${MODEL}.json
metadata=$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2_${MODEL}.json

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

for f in "$catalogue" "$model" "$metadata"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 3; }
done
echo "ARM=$ARM MODEL=$MODEL CAT=$CAT"
echo "  catalogue=$catalogue"
echo "  model=$model"

cases=()
for case in $(seq 0 19); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_corner_extrapolation \
  --domain-root "$v2/domain" \
  --response-catalogue "$catalogue" \
  --measurement-model "$v2/flow/paired/selected.pt" \
  --emulator-model "$model" \
  --emulator-metadata "$metadata" \
  "${cases[@]}" --response-shear 0.2 \
  --output "$root/result_${ARM}.json" \
  --n-boot 10000 --bootstrap-seed 20260915 --device cpu

echo "CORNER_AB_JOB_DONE arm=$ARM output=$root/result_${ARM}.json"
