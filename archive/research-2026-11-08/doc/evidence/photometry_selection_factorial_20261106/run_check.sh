#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=photo_cross_check
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_selection_factorial_20261106/logs/check_%j.log
set -euo pipefail
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/photometry_selection_factorial_20261106/check.py
