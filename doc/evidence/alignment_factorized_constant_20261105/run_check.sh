#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=factorized_const_check
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_factorized_constant_20261105/logs/check_%j.log
set -euo pipefail
unset PYTHONPATH
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_factorized_constant_20261105/check.py
