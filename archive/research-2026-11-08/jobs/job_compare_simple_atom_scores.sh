#!/bin/bash
#SBATCH --job-name=sbsi-simple-scores
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/simple_atom_scores_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-simple-scores-${SLURM_JOB_ID}"
SBSI_SCORE_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$SBSI_SCORE_PYTHON" scripts/compare_simple_atom_scores.py
