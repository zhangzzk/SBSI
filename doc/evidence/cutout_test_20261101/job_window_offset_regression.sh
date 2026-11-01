#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 02:00:00
#SBATCH -J cutreg
#SBATCH -o /project/ls-gruen/users/zekang.zhang/sbsi/cutout_test_20261101/logs/regression_%j.log
set -euo pipefail
D=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/cutout_test_20261101
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u $D/window_offset_regression.py \
  --out /project/ls-gruen/users/zekang.zhang/sbsi/cutout_test_20261101/window_offset_regression.json --workers 16
