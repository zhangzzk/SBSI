#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fg0_truth_predict
#SBATCH --partition=inter,cip
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/root_cause_radius_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/root_cause_radius_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
out=$v2/root_cause_radius_v1/truth_only_predictability

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
"$python" -u -m scripts.diagnose_truth_only_boundary_predictability \
  --domain-root "$v2/domain" \
  --output-root "$out" \
  --fit-rows-per-case 15000 \
  --train-evaluation-rows-per-case 5000 \
  --validation-rows-per-case 20000 \
  --seed 20260916 \
  --threads "${SLURM_CPUS_PER_TASK:-8}"
