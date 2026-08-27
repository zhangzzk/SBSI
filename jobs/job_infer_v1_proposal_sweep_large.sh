#!/usr/bin/env bash
# Extend the exact proposal search to million-atom candidate supports.

#SBATCH --job-name=sbsi_infer_v1_qsweep_large
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:45:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_sweep_obs514716_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_sweep_obs514716_v2/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
closure=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_proposal_sweep_obs514716_v2

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/plot_infer_v1_sampling.py" \
  --reference-result "$closure/combined/result.json" \
  --output "$root/sweep" \
  --exact-object-id 514716 --proposal-sweep --device cuda \
  --proposal-k-ladder 524288 1048576 2097152 \
  --proposal-prefilter 8388608 \
  --proposal-epsilon-ladder 0.4 0.6 0.75 0.85 0.95 1.0 \
  --proposal-m-ladder 4096 8192 16384 \
  --proposal-replicates 256 --proposal-mc-seed 9917
