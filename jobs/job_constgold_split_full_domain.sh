#!/usr/bin/env bash
# Split the ConstGold model response into its flow and blending terms.
#
# Usage: sbatch jobs/job_constgold_split_full_domain.sh <flow.pt> <tag>
#
# The goal is |m| < 0.3% on ConstGold.  ConstGold measures only the sum
# R = R_flow + R_blend, so its m cannot say which term is spending the budget,
# and a small m can be two large errors cancelling -- which is exactly what the
# radius bands turned out to be doing inside the flow alone.  Binning on the
# model's own R_blend separates them: the fitted coefficients are the
# fractional error of each term, with a case bootstrap on each.
#
# The blending axis is crossed with a primary-magnitude axis because a quiet
# blending bin is strongly enriched in bright primaries, and reading the flow
# only where blending is quiet quotes a bright-primary number as a global one.
#
# The flow is the swappable argument.  R_blend and the bin definition are fixed
# files, so running this before and after a flow change isolates the flow's
# contribution with everything else held identical.

#SBATCH --job-name=sbsi_cg_split
#  gpu:a40 matches only the WHOLE cards and queues for hours while cip's idle
#  vGPU slices sit free; this job peaks near 1.2G so a slice is the right size
#  and it does not compete with the full-card training jobs.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_split_full_domain_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_split_full_domain_v1/logs/%x_%j.err

set -euo pipefail
flow=${1:?flow checkpoint required}
tag=${2:?output tag required}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
refine=$cache/full_domain_true_mag26_guard_refine_v1
blocks=$refine/constgold_property_bias_uncut_true_mag26_c40_139_v1/rblend/blocks
root=$refine/constgold_split_full_domain_v1
merged=$root/rblend_full_domain_cases40_139.feather
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather

[ -f "$flow" ] || { echo "missing flow: $flow" >&2; exit 3; }
mkdir -p "$root/logs"
export NO_EXPANDABLE_SEGMENTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

cases=()
inputs=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done
for first in 40 50 60 70 80 90 100 110 120 130; do
  last=$((first + 9))
  block=$blocks/cases${first}_${last}.feather
  [ -f "$block" ] || { echo "missing R_blend block: $block" >&2; exit 3; }
  inputs+=(--input "$block")
done

# The lookup does not depend on the flow being measured, so it is built once
# and reused by every arm of every comparison.
if [ -f "$merged" ]; then
  echo "SKIP merged lookup already present"
else
  "$python" -u -m scripts.merge_constgold_fixed_g0_blend_lookups \
    "${inputs[@]}" "${cases[@]}" --output "$merged"
fi

output=$root/split_$tag.json
if [ -f "$output" ]; then
  echo "refusing to overwrite existing split result: $output" >&2
  exit 1
fi

"$python" -u -m scripts.diagnose_constgold_response_split \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$cache/full_domain_true_mag26_guard_v1/domain/flow/g0/case{case:03d}.npz" \
  --flow "$flow" \
  --rblend "full_domain=$merged" \
  "${cases[@]}" --bins 12 --bin-column R_blend_abs \
  --magnitude-edge 23 --magnitude-edge 24 \
  --magnitude-edge 25 --magnitude-edge 26 \
  --primary-catalogue "$truth" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260917 --device cuda

echo "CONSTGOLD_SPLIT_DONE tag=$tag output=$output"
