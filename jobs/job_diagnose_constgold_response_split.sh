#!/usr/bin/env bash
# Split the ConstGold model response into its flow and blending terms.
#
# ConstGold measures only the sum, so the +0.55 pp that the relaxed rejection
# bought could be removing an R_blend error or cancelling an R_flow one.
# Binning on the model's own R_blend separates them: the fitted coefficients
# are the fractional error of each term.

#SBATCH --job-name=sbsi_cgfg0_split
#  gpu:a40 matches only the WHOLE cards and queues for hours while cip's idle
#  vGPU slices (a40-8gb/16gb/24gb) sit free; those nodes carry ~25G of RAM, so
#  a 96G request cannot land on one either.  This job peaks near 1.2G.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_split_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_split_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v2=$cache/fixed_g0_m258_r060_v2
ab=$v2/constgold_response_rejection_ab_v1
root=$v2/constgold_response_split_v1

mkdir -p "$root/logs"
#  the A40-16Q slice lacks the CUDA VMM APIs torch's expandable segments need
export NO_EXPANDABLE_SEGMENTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
nvidia-smi -L

for f in "$ab/rblend_v2_baseline_cases40_89.feather" \
         "$ab/rblend_v2_relaxed_cases40_89.feather" \
         "$v2/flow/paired/selected.pt"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 3; }
done

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_constgold_response_split \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$v2/flow/paired/selected.pt" \
  --rblend "reject_baseline=$ab/rblend_v2_baseline_cases40_89.feather" \
  --rblend "reject_relaxed=$ab/rblend_v2_relaxed_cases40_89.feather" \
  "${cases[@]}" --bins 12 \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260915 --device cuda

echo "CONSTGOLD_SPLIT_DONE output=$root/result.json"
