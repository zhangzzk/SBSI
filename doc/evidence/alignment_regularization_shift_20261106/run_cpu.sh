#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=guard_shift_report
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py -o resources.json
script=${1:?script name}
shift
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u "/home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_regularization_shift_20261106/$script" "$@"
