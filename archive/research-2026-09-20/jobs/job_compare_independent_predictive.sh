#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_density
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests/test_joint_predictive_comparison.py -q
exec "$python" -u -m scripts.compare_joint_predictive_results \
  --baseline "$root/independent_baseline_full.json" \
  --candidate "$root/independent_candidate_full.json" \
  --output "$root/independent_predictive_comparison.json"
