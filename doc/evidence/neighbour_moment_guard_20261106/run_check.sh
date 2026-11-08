#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=moment_guard_check
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_moment_guard_20261106/logs/check_%j.log
set -euo pipefail
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_moment_guard_20261106/check.py
