#!/usr/bin/env bash
#SBATCH --job-name=sbsi_popguard_sum
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.summarize_population_guard_ab \
  --root /project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1
