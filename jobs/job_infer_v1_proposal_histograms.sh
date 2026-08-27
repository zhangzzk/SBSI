#!/usr/bin/env bash
# Plot proposal and exact-target log-likelihood histograms for observation 514716.

#SBATCH --job-name=sbsi_infer_v1_qhist
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=00:30:00
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

"$python" "$repo/scripts/plot_infer_v1_proposal_histograms.py" \
  --reference-result "$closure/combined/result.json" \
  --screen-result "$root/screen/result.json" \
  --output "$root/likelihood_histograms_v1" \
  --object-id 514716 --device cuda --bins 90
