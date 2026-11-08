#!/usr/bin/env bash
# Decompose the exact R_blend response contribution without flow evaluation.

#SBATCH --job-name=sbsi_cgfg0_rblend
#SBATCH --partition=cip,inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_constgold_rblend_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --rblend "historical_v35=$root/historical_v35_rblend_c40_89.feather" \
  --rblend "old_fixed_v1_nll=$root/../rblend_v1_cases40_89.feather" \
  --rblend "new_v2=$root/../rblend_v2_cases40_89.feather" \
  "${cases[@]}" --output "$root/rblend_components.json" \
  --h 0.02 --n-boot 10000 --bootstrap-seed 20260915

echo "CONSTGOLD_RBLEND_COMPONENT_JOB_DONE output=$root/rblend_components.json"
