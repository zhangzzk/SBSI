#!/bin/bash
#SBATCH --job-name=rbl_oof_export
#SBATCH --time=00:30:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1%2
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
OBJECTIVES=(pair pair_scene)
OBJECTIVE=${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

python -u scripts/export_selected_crossfit_oof.py \
  --summary "$RUN_ROOT/$OBJECTIVE/summary.json" \
  --output "$RUN_ROOT/$OBJECTIVE/selected_oof.npy"
echo "SELECTED_CROSSFIT_OOF_EXPORT_JOB_DONE objective=$OBJECTIVE"
date
