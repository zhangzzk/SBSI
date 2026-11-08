#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=guard_compare_prepare
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106/logs/prepare_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106
cd "$root"
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py -o "$root/resources.json"
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_regularized_20261106/prepare.py
