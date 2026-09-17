#!/usr/bin/env bash
# Band-guard refinement with the cumulative cuts re-weighted, at one guard weight.
#
# Usage: sbatch jobs/job_band_guard_refine_v2.sh <guard_weight> [epochs]
#
# This is deliberately not the same output tree as job_band_guard_refine.sh.
# That job's runs were trained with the previous guard loss, in which every cut
# in the bank carried the same response scale.  Since then
# ``refine_band_guard_flow.py`` divides the cumulative cuts' scale by
# ``CUMULATIVE_SCALE_FACTOR``, which multiplies their share of the averaged
# quadratic by a hundred, and averages the weights over the back half of the
# run.  Resuming the old directories would continue a run under a loss it was
# not trained with, so the tree is separate and ``--resume`` here only ever
# picks up a v2 run of its own.
#
# What this run is testing: the 2026-09-17 entry showed the band structure
# improved while the overall level got worse, and that the selected epoch was
# the luckiest of twenty draws from a stationary band centred near +1.1%.  The
# re-weighting makes a falsifiable prediction -- the epoch *mean* must move
# toward zero, not just its spread.  ``averaged.pt`` is the reportable
# checkpoint because no validation case takes part in choosing it.

#SBATCH --job-name=sbsi_band_refine_v2
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.err

set -euo pipefail
weight=${1:?guard weight required}
epochs=${2:-20}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
refine_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1
output=$root/rebalanced_weight_$weight

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

"$python" -u -m scripts.refine_band_guard_flow \
  --domain-root "$domain_root" \
  --initial-flow "$refine_root/flow/selected.pt" \
  --output-root "$output" \
  --epochs "$epochs" \
  --guard-weight "$weight" \
  --resume

test -f "$output/run/averaged.pt" \
  || { echo "missing averaged.pt in $output/run" >&2; exit 1; }
echo "BAND_GUARD_REFINE_V2_COMPLETE weight=$weight output=$output"
