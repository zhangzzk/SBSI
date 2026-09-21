#!/usr/bin/env bash
# Evaluate soft-guard sharpening on validation pairs not used for selection.

#SBATCH --job-name=sbsi_guard_widths
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/guard_width_sweep_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/guard_width_sweep_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1
output=$root/guard_width_sweep_v1

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

"$python" -u -m scripts.evaluate_full_domain_guard_widths \
  --domain-root "$root/domain" \
  --flow "$root/flow/guard/selected.pt" \
  --reference-indices "$root/flow/validation_guard_pair_indices.npy" \
  --output "$output/result.json" \
  --pairs 400000 --draws 16 --groups 4 --batch-size 2048
