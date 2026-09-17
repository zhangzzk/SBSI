#!/usr/bin/env bash
# Per-leg hard-cut response for one checkpoint on all 200 domain cases.
#
# Usage: sbatch jobs/job_per_leg_all_domain_cases.sh <flow.pt> <tag>
#
# Why all 200 and not the 40 validation cases.  The 2026-09-17 entry measured
# the case-to-case scatter of ``m`` at 2.7 percentage points, which caps a
# 40-case measurement at 0.42 pp and cannot on its own support a 0.3% claim.
# Decomposing that scatter shows 96-98% of it sits on the simulation side and
# the two sides are uncorrelated (r = +0.008 global, -0.017 at the deployment
# cut), and every case renders the same input truth catalogue, so the scatter
# is measurement noise on the simulation's response rather than real
# case-to-case structure the flow is failing to track.  Measurement noise
# averages down: 200 cases give 0.19 pp, which does resolve 0.3%.
#
# The 160 training cases enter the training gradient, so this cannot be called
# held-out.  It does not have to be: the script stores the full per-case
# matrices, so the train-only and validation-only numbers are both recoverable
# from one output file at no extra cost.  If they agree within their errors,
# the 200-case number is the honest one to quote; if they do not, the gap is
# itself the measurement of how much the flow has memorised.
#
# Note also that this noise cancels exactly when two checkpoints are compared:
# the simulation side does not depend on the flow and is bit-identical across
# checkpoints.  The 0.42 pp cap applies to absolute claims only.

#SBATCH --job-name=sbsi_perleg_all
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.err

set -euo pipefail
flow=${1:?flow checkpoint required}
tag=${2:?tag required}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1
output=$root/all_domain_cases/$tag

if [ ! -f "$flow" ]; then
  echo "missing checkpoint: $flow" >&2
  exit 1
fi

cases=()
for case in $(seq 0 199); do
  cases+=(--case "$case")
done

mkdir -p "$output" "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

if [ -f "$output/cumulative.json" ]; then
  echo "SKIP cumulative.json already present"
else
  "$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
    --domain-root "$domain_root" \
    --flow "$flow" \
    "${cases[@]}" \
    --output "$output/cumulative.json" \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
fi

echo "PER_LEG_ALL_DOMAIN_COMPLETE tag=$tag flow=$flow output=$output"
