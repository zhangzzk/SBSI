#!/usr/bin/env bash
# Split the ConstGold R_blend by whether the primary can be labelled at all.

#SBATCH --job-name=sbsi_unlab_blend
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/unlabelled_blend_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/unlabelled_blend_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/unlabelled_blend_c40_89_v1

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" -u -m scripts.diagnose_unlabelled_blend_primaries \
  --rblend-lookup "$v2/constgold_response_c40_89_v1/rblend_v2_cases40_89.feather" \
  --response-catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --output "$root/result.json" \
  --n-boot 10000 --bootstrap-seed 20260915

echo "UNLABELLED_BLEND_JOB_DONE output=$root/result.json"
