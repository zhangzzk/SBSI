#!/usr/bin/env bash
# Read the flow at fixed PRIMARY MAGNITUDE, not just where blending is quiet.
#
# The quiet-blend read gave the flow 2.4% low, the transfer estimate gave it
# about 2% high, and the reason is that a blending bin is not a random slice of
# the cohort: the quietest twelfth is 30% primaries brighter than r=23 against
# 1.7% in the cohort as a whole.  Crossing the blending axis with a primary
# magnitude axis reads the flow error at fixed magnitude, so it can be
# re-weighted to the cohort instead of quoted on whichever primaries happened
# to be isolated.  Same cases, same flow, same lookup, same bin edges as
# constgold_flow_quiet_v1 -- only the reported cells are new.

#SBATCH --job-name=sbsi_cgfg0_quietmag
#  gpu:a40 matches only the whole cards, which queue for hours; the idle cip
#  capacity is vGPU slices on ~25G nodes.  The identical job peaked at 1.2G.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_flow_quiet_mag_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_flow_quiet_mag_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v2=$cache/fixed_g0_m258_r060_v2
ab=$v2/constgold_response_rejection_ab_v1
lookup=$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather
root=$v2/constgold_flow_quiet_mag_v1

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

for f in "$lookup" "$ab/rblend_v2_relaxed_cases40_89.feather" \
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
  --rblend "reject_baseline=$lookup" \
  --rblend "reject_relaxed=$ab/rblend_v2_relaxed_cases40_89.feather" \
  "${cases[@]}" --bins 12 --bin-column R_blend_abs \
  --magnitude-edge 23 --magnitude-edge 24 \
  --magnitude-edge 25 --magnitude-edge 26 \
  --primary-catalogue "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260915 --device cuda

echo "CONSTGOLD_FLOW_QUIETMAG_DONE output=$root/result.json"
