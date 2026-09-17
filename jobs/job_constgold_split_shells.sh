#!/usr/bin/env bash
# Give each neighbour-separation shell its own fractional error.
#
# Usage: sbatch jobs/job_constgold_split_shells.sh <flow.pt> <tag>
#
# The two-term split says the blending term is -15.41% +- 0.53 too large, but it
# reaches that by forcing every neighbour -- a touching pair and one nine
# arcseconds away alike -- to share a single multiplicative error.  On this
# domain that is a strong assumption: R_blend sums the 20 nearest neighbours
# inside 10 arcsec, the cap binds for nearly every primary, and 62% of the total
# comes from beyond 3 arcsec where two galaxies barely overlap.
#
# This fits R_measured - R_flow against the four shell sums jointly, so
# ConstGold reports how much of each separation range it actually supports
# rather than being told in advance that they are all wrong by the same factor.
#
# The bins are cut on the share of absolute blending amplitude beyond 4 arcsec,
# not on the total amplitude.  That choice was made on conditioning alone before
# any m was read from it: across 12 bins the scaled condition number of the
# design is 29 on the share axis against 63 on the total-amplitude axis, and
# every shell moves by at least a factor of three.  `joint_condition_number` in
# the output reports it again on the real fit, and has to be read with the
# coefficients -- a large value means the split between shells is unconstrained
# no matter how small the bootstrap errors look.

#SBATCH --job-name=sbsi_cg_shells
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_split_shells_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_split_shells_v1/logs/%x_%j.err

set -euo pipefail
flow=${1:?flow checkpoint required}
tag=${2:?output tag required}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
refine=$cache/full_domain_true_mag26_guard_refine_v1
blocks=$refine/constgold_rblend_shells_v1/rblend/blocks
root=$refine/constgold_split_shells_v1
merged=$root/rblend_shells_cases40_139.feather

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
  [ -f "$block" ] || { echo "missing shell R_blend block: $block" >&2; exit 3; }
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
  --anchor-pattern "$cache/full_domain_true_mag26_guard_v1/domain/flow/g0/case{case:03d}.npz" \
  --flow "$flow" \
  --rblend "shells=$merged" \
  --blend-column "r0_2=R_blend_r0_2" \
  --blend-column "r2_4=R_blend_r2_4" \
  --blend-column "r4_6=R_blend_r4_6" \
  --blend-column "r6_10=R_blend_r6_10" \
  "${cases[@]}" --bins 12 --bin-column blend_abs_share_beyond_4 \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260917 --device cuda

echo "CONSTGOLD_SPLIT_SHELLS_DONE tag=$tag output=$output"
