#!/usr/bin/env bash
# Measure the R_blend carried by pairs the label pipeline's rejection removes.

#SBATCH --job-name=sbsi_label_reject
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/label_rejection_c0_19_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/label_rejection_c0_19_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
sims=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/label_rejection_c0_19_v1

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 0 19); do
  cases+=(--case "$case")
done

model_root=$blend/fixed_g0_m258_r060_v2/models
"$python" -u -m scripts.diagnose_label_rejection \
  --domain-root "$v2/domain" \
  --input-pattern "$sims/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --response-catalogue "$sims/response_catalogue_train.feather" \
  --measurement-model "$v2/flow/paired/selected.pt" \
  --emulator-model "$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  "${cases[@]}" \
  --output "$root/result.json" \
  --n-boot 10000 --bootstrap-seed 20260915 --device cpu

echo "LABEL_REJECTION_JOB_DONE output=$root/result.json"
