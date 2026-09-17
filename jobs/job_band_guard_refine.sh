#!/usr/bin/env bash
# Combined-gradient band-guard refinement at one guard weight.
#
# Usage: sbatch jobs/job_band_guard_refine.sh <guard_weight> [epochs]
#
# The weight is the only free knob introduced by summing the likelihood and
# guard gradients before the clip.  Smoke job 16546597 measured the two
# gradient norms at 24.2 (likelihood alone) and 2323 (combined, guard weight
# one), so the guard gradient is about ninety-six times the likelihood's and a
# weight near 0.01 makes the two comparable inside the clipped step.  The
# bracket run here is 0.01 and 0.1.

#SBATCH --job-name=sbsi_band_refine
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
output=$root/weight_$weight

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

echo "BAND_GUARD_REFINE_COMPLETE weight=$weight output=$output"
