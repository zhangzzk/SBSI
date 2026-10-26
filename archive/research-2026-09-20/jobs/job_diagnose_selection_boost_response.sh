#!/usr/bin/env bash
# Bin the flow self-response residual on true magnitude by measurement boost.

#SBATCH --job-name=sbsi_sel_boost
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/selection_boost_c0_19_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/selection_boost_c0_19_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
sims=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/selection_boost_c0_19_v1

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

"$python" -u -m scripts.diagnose_selection_boost_response \
  --domain-root "$v2/domain" \
  --input-pattern "$sims/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --self-model "$v2/response_profile_predictions_c0_19_v2/R_self_model.feather" \
  "${cases[@]}" \
  --output "$root/result.json" \
  --n-boot 10000 --bootstrap-seed 20260915

echo "SELECTION_BOOST_JOB_DONE output=$root/result.json"
