#!/bin/bash
#SBATCH -J galsbi-guard
#SBATCH -o /project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_guard_strength_20261103/logs/%x_%j.log
# Stronger-guard test for the GalSBI-dev flow; rule: $DATA_DIR/sbsi_caches/galsbi_guard_strength_20261103/rule.md.
#   sbatch -p cip --gres=gpu:a40-24gb:1 -c 4 --mem=40G -t 12:00:00 jobs/job_galsbi_guard_strength.sh train --steps 40
#   sbatch -p cip --gres=gpu:a40-24gb:1 -c 4 --mem=40G -t 12:00:00 jobs/job_galsbi_guard_strength.sh train --steps 160
#   sbatch -p cip --gres=gpu:a40-24gb:1 -c 4 --mem=40G -t 4:00:00 jobs/job_galsbi_guard_strength.sh check [--flows ...]
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
SBSI=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe
FROZEN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python
echo "host=$(hostname) commit=$(git -C "$SBSI" rev-parse HEAD) args=$*"
cd /tmp && $FROZEN -I -u "$SBSI/scripts/run_galsbi_guard_strength.py" "$@"
