#!/usr/bin/env bash
# Measure a band-guard refined checkpoint on the held-out cases.
#
# Usage: sbatch jobs/job_band_guard_evaluate.sh <guard_weight>
#
# Two measurements, both per-leg and under the exact catalogue indicator, so
# they are the estimand the guard loss minimises rather than the zero-leg
# anchored extraction.  The cumulative bank gives the deployment number.  The
# band profile gives the acceptance test: band residuals cannot cancel, so a
# global number that improves while the bands do not has only rearranged the
# cancellation that produced the previous -0.73%.

#SBATCH --job-name=sbsi_band_eval
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.err

set -euo pipefail
weight=${1:?guard weight required}
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1
flow=$root/weight_$weight/selected.pt
output=$root/evaluation_weight_$weight
edges="2.2 2.6 2.8 3.0 3.2 3.5 4.0 5.0 7.0"

if [ ! -f "$flow" ]; then
  echo "missing checkpoint: $flow" >&2
  exit 1
fi

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
    --output "$output/cumulative.json" \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
fi

if [ -f "$output/bands_magnitude_lt_25.8.json" ]; then
  echo "SKIP bands_magnitude_lt_25.8.json already present"
else
  "$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
    --domain-root "$domain_root" \
    --flow "$flow" \
    --output "$output/bands_magnitude_lt_25.8.json" \
    --radius-band-edges $edges --band-magnitude-max 25.8 \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
fi

echo "BAND_GUARD_EVALUATE_COMPLETE weight=$weight output=$output"
