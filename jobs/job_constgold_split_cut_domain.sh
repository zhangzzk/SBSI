#!/usr/bin/env bash
# The same flow/blending split, run inside the emulator's own training domain.
#
# Usage: sbatch jobs/job_constgold_split_cut_domain.sh <flow.pt> <tag>
#
# The full-domain split put the blending term's fractional error at
# -15.41% +- 0.53, against +2.071% +- 0.275 for the flow.  One obvious
# explanation is that the emulator is simply being used outside the population
# it was fitted to: its metadata records an anchor predicate of
# "MAG_AUTO < 25.8; FLUX_RADIUS > 3.0 pixels", while the full-domain anchors
# apply no measured cuts at all.
#
# This run holds the flow, the truth catalogue, the bin definition, the bin
# count and the estimator fixed, and changes only the anchor population, to the
# cut domain `fixed_g0_m258_r060_v2` -- the very predicate the emulator was
# trained under.  Its R_blend lookup was also built on that domain, so the two
# model terms are evaluated on the same objects as before.
#
# If the blending coefficient comes back near zero here, the -15.4% is an
# out-of-domain application and the fix is the emulator's training support.  If
# it comes back near -15%, the emulator is wrong inside its own domain, or the
# coefficient is absorbing something that is not the emulator at all -- most
# likely detection and selection response, which the matched-usable branch this
# script fits on does not model.

#SBATCH --job-name=sbsi_cg_split_cut
#  As in the full-domain job: a vGPU slice, not a whole card.  This peaks near
#  1.2G and must not compete with the full-card training jobs for the owner's
#  two-GPU budget.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_split_cut_domain_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_split_cut_domain_v1/logs/%x_%j.err

set -euo pipefail
flow=${1:?flow checkpoint required}
tag=${2:?output tag required}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
cut=$cache/fixed_g0_m258_r060_v2
root=$cut/constgold_split_cut_domain_v1
merged=$root/rblend_cut_domain_cases40_139.feather
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
for case in $(seq 40 139); do
  cases+=(--case "$case")
done

#  The cut domain's R_blend is stored as one 50-case file for 40--89 and as
#  ten-case blocks for 90--139.  Both were built with the same emulator model
#  and metadata as the full-domain lookup.
inputs=(--input "$cut/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather")
for first in 90 100 110 120 130; do
  last=$((first + 9))
  block=$cut/constgold_property_bias_c40_139_v1/rblend/blocks/cases${first}_${last}.feather
  [ -f "$block" ] || { echo "missing R_blend block: $block" >&2; exit 3; }
  inputs+=(--input "$block")
done

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
  --anchor-pattern "$cut/domain/flow/g0/case{case:03d}.npz" \
  --flow "$flow" \
  --rblend "cut_domain=$merged" \
  "${cases[@]}" --bins 12 --bin-column R_blend_abs \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260917 --device cuda

echo "CONSTGOLD_SPLIT_CUT_DONE tag=$tag output=$output"
