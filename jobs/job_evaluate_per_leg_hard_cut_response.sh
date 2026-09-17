#!/usr/bin/env bash
# Measure the per-leg hard-cut response of the sharp-guard refinement on all
# held-out cases.  This is the copula-free estimand that the guard training
# loss minimises, and it is the number missing from the existing hard-cut
# validation, which anchors both legs on the g=0 draw.

#SBATCH --job-name=sbsi_perleg_hardcut
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/per_leg_hard_cut_c40_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/per_leg_hard_cut_c40_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
refine_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1
output=$refine_root/per_leg_hard_cut_c40_v1

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

# This launcher deliberately runs from a worktree.  Record which sbsi package
# actually resolves so the result cannot be misattributed to the main checkout.
"$python" -c "import sbsi, scripts; print('sbsi from', sbsi.__file__); print('scripts from', scripts.__file__)"

"$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
  --domain-root "$domain_root" \
  --flow "$refine_root/flow/selected.pt" \
  --output "$output/refined_heldout_per_leg.json" \
  --draws 16 --groups 4 --batch-size 2048 --device cuda

echo "PER_LEG_HARD_CUT_VALIDATION_COMPLETE output=$output"
