#!/usr/bin/env bash
# Compare flatter score-selection families at epsilon zero for observation 514716.

#SBATCH --job-name=sbsi_infer_v1_flatq
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_flatter_selection_obs514716_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_flatter_selection_obs514716_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
closure=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_flatter_selection_obs514716_v1

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/test_infer_v1_flatter_selection.py" \
  --reference-result "$closure/combined/result.json" \
  --output "$root/screen" --object-id 514716 --device cuda \
  --score-pool 4194304 --score-prefilter 8388608 --k 16384 \
  --weight-forms exponential gaussian logistic cauchy \
  --temperatures 0.5 1.0 1.5 \
  --candidate-seeds 7301 7302 7303 7304 7305 --bins 70
