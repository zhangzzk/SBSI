#!/bin/bash
#SBATCH --job-name=v22_caseboot
#SBATCH --array=0-16%17
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_caseboot_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_caseboot_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_v22_training_bootstrap
REP=${SLURM_ARRAY_TASK_ID:?array task required}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -u scripts/bootstrap_v22_emulator_cases.py \
  --config configs/fs2_lsst_r_extnbr_v22.yaml \
  --replicate "$REP" --output-dir "$OUT"
echo V22_CASE_BOOTSTRAP_JOB_DONE; date
