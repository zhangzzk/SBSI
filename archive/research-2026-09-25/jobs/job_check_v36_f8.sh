#!/bin/bash
#SBATCH --job-name=sbsi-f8-check
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=logs/v36_f8_check_%j.out
set -euo pipefail
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python \
  /home/z/Zekang.Zhang/SBSI/scripts/check_v36_f8_status.py \
  --root /project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1
