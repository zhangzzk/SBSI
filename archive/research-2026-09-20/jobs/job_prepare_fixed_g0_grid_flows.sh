#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fg0gridprep
#SBATCH --partition=inter,cip
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
"$python" -u -m scripts.train_fixed_g0_grid_flows prepare \
  --domain-root "$root/domain" \
  --output-root "$root/grid_flows_v1" \
  --parent "$root/flow/nll/selected.pt" \
  --parent-optimizer "$root/flow/nll/selected_optimizer.pt"
