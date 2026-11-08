#!/usr/bin/env bash
# Measure m for both arms of the rejection A/B.
#
# The SAME certified flow is used for both entries, so the only thing that
# differs between them is which response emulator supplied R_blend.  Any change
# in m is attributable to the bright-neighbour rejection and nothing else.

#SBATCH --job-name=sbsi_cgfg0_eval_ab
#  gpu:a40 matches only the WHOLE cards and queues for hours while cip's idle
#  vGPU slices (a40-8gb/16gb/24gb) sit free; those nodes carry ~25G of RAM, so
#  a 96G request cannot land on one either.  This job peaks near 1.2G.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2

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

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
classifiers=()
for seed in 20260913 20260914 20260915; do
  classifiers+=(--classifier "$v1/classifier/seed${seed}/selected.pt")
done

"$python" -u -m scripts.evaluate_constgold_fixed_g0_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --model "reject_baseline=$v2/flow/paired/selected.pt" \
  --model "reject_relaxed=$v2/flow/paired/selected.pt" \
  --rblend "reject_baseline=$root/rblend_v2_baseline_cases40_89.feather" \
  --rblend "reject_relaxed=$root/rblend_v2_relaxed_cases40_89.feather" \
  "${classifiers[@]}" "${cases[@]}" \
  --subset cases40_59=40:60 \
  --subset cases60_79=60:80 \
  --subset cases80_89=80:90 \
  --subset cases40_89=40:90 \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260914 --device cuda

echo "CONSTGOLD_EVAL_AB_DONE output=$root/result.json"
