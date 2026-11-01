#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 32
#SBATCH --mem 32G
#SBATCH -t 03:00:00
#SBATCH -J cutremeas
#SBATCH -o /project/ls-gruen/users/zekang.zhang/sbsi/cutout_test_20261101/logs/remeasure_%A_%a.log
# Array index = ConstGold case number, plus leg. LIMIT>0 runs a smoke test on the first LIMIT detections.
set -euo pipefail
D=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/cutout_test_20261101
O=/project/ls-gruen/users/zekang.zhang/sbsi/cutout_test_20261101
C=${SLURM_ARRAY_TASK_ID}
LIMIT=${LIMIT:-0}
TAG=case${C}_0.02
[ "$LIMIT" -gt 0 ] && TAG=smoke_${TAG}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u $D/remeasure_centred_cutout.py \
  --case-dir /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case${C}_0.02 \
  --out $O/remeasure/${TAG}.feather --seed-base $((C * 10000000 + 20261101)) --workers 32 --limit $LIMIT
