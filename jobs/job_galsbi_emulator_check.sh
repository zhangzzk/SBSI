#!/bin/bash
# GalSBI-dev emulator vs half-shear simulation labels (doc/evidence/galsbi_emulator_check_20261104/rule.md).
#   sbatch -p cluster --constraint=x86-64-v3 -c 8 --mem=16G jobs/job_galsbi_emulator_check.sh
#SBATCH --job-name=galsbi_emu_check
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_emulator_check_20261104/logs/%x_%j.log
set -euo pipefail
S=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/galsbi_emulator_check_20261104/emulator_check.py
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
echo "host=$(hostname) commit=$(git -C "$(dirname "$S")" rev-parse HEAD)"
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u "$S" sums
