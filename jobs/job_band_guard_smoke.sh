#!/usr/bin/env bash
# Two short epochs of the combined-gradient band-guard refinement.
#
# The point of this run is measurement, not training: it records the
# likelihood-only gradient norm and the combined norm at guard weight one, so
# the production weight is chosen from the balance the clip actually applies
# rather than guessed.  It also fixes the per-epoch cost.

#SBATCH --job-name=sbsi_band_smoke
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/smoke/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/smoke/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
refine_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1
output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/smoke

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

"$python" -u -m scripts.refine_band_guard_flow \
  --domain-root "$domain_root" \
  --initial-flow "$refine_root/flow/selected.pt" \
  --output-root "$output/run" \
  --epochs 2 \
  --rows-per-epoch 200000 \
  --guard-weight 1.0

echo "BAND_GUARD_SMOKE_COMPLETE output=$output"
