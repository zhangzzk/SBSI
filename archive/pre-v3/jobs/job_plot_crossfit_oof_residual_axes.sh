#!/bin/bash
#SBATCH --job-name=rbl_oof_axes
#SBATCH --time=01:30:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
OUT=$ROOT/results/v22_crossfit_oof_residual_axes_three_models
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

python -u scripts/plot_crossfit_oof_residual_axes.py \
  --source-cache "$SOURCE" \
  --pair-oof "$RUN/pair/selected_oof.npy" \
  --pair-scene-oof "$RUN/pair_scene/selected_oof.npy" \
  --output-prefix "$OUT" --n-bins 20
echo CROSSFIT_OOF_RESIDUAL_AXES_PLOT_JOB_DONE
date
