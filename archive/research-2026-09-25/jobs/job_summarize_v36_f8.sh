#!/bin/bash
#SBATCH --job-name=sbsi-f8-summary
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v36_f8_summary_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1
"$PY" -m scripts.summarize_v36_f8 --root "$ROOT" --output "$ROOT/development_comparison.json"
