#!/usr/bin/env bash
# Feasibility frontier for the requested 20% proposal ESS goal.

#SBATCH --job-name=sbsi_k_eps_frontier
#SBATCH --partition=inter
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_learned_k_epsilon_ladder_n16_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_learned_k_epsilon_ladder_n16_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_learned_k_epsilon_ladder_n16_v1
reference=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/combined/result.json
current=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_contrastive_proposal_b2048_s20000_v1/pilot/model.pt
soft=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_soft_posterior_b2048_q64_s3000_v1/tau1/train/model.pt

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$root/logs"

"$python" "$repo/scripts/test_infer_v1_learned_k_epsilon_ladder.py" \
  --reference-result "$reference" --output "$root/diagnostic" --device cuda \
  --checkpoints "$current" "$soft" --labels current_learned soft_tau1 \
  --object-ids \
    596125 834510 275582 271297 210899 300813 391187 729450 \
    626519 978866 145412 130164 263700 226165 486566 172457 \
  --k-values 65536 131072 262144 524288 \
  --epsilon-values 0.1 0.2 0.3 0.4 0.5 0.6 \
  --atom-chunk 65536
