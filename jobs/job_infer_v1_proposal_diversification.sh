#!/usr/bin/env bash
# Screen score-tail diversification and small-query unions on observation 514716.

#SBATCH --job-name=sbsi_infer_v1_qdiverse
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_diversification_obs514716_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_diversification_obs514716_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
closure=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_diversification_obs514716_v1

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/test_infer_v1_proposal_diversification.py" \
  --reference-result "$closure/combined/result.json" \
  --output "$root/screen" --object-id 514716 --device cuda \
  --score-pool 4194304 --score-prefilter 8388608 \
  --k 32768 65536 --core-fractions 0.75 0.5 \
  --temperatures 1.5 2 4 --epsilons 0.1 0.2 0.3 \
  --candidate-seeds 7301 7302 7303 7304 7305 \
  --query-counts 8 16 --log-strata 16 \
  --mc-replicates 256 --mc-draws 16384 \
  --mc-seed 9917 --holdout-mc-seed 9918
