#!/usr/bin/env bash
# Resolve the fixed-g0 FLUX_RADIUS boundary into 32 response/component bins.

#SBATCH --job-name=sbsi_cgrad
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_small_radius_c40_139_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_small_radius_c40_139_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
root=$v2/constgold_small_radius_c40_139_v1
lookup=$root/rblend_cases40_139.feather
output=$root/result.json

mkdir -p "$root/logs"
export NO_EXPANDABLE_SEGMENTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

test -f "$lookup"
cases=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_constgold_response_split \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$v2/flow/paired/selected.pt" \
  --rblend "baseline=$lookup" \
  "${cases[@]}" --bins 32 \
  --anchor-bin-column g0_flux_radius_arcsec \
  --magnitude-edge 24 --magnitude-edge 25 --magnitude-edge 26 \
  --primary-catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260916 --device cuda

echo "CONSTGOLD_SMALL_RADIUS_DONE output=$output"
