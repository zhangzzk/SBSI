#!/bin/bash
#SBATCH --job-name=v2rwvfix
#SBATCH --time=02:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2rwvfix_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v2rwvfix_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1
RECIPE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/summary.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
export CONFIG_PATH=$ROOT/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=40
unset WEIGHT_CLOSE
cd "$ROOT"

test -s "$RECIPE"
mkdir -p "$OUTPUT"
nvidia-smi -L
"$PY" -u scripts/train_v2_reweighted_vector_fixed.py \
  --config "$CONFIG_PATH" \
  --recipe-summary "$RECIPE" \
  --output-dir "$OUTPUT"
test -s "$OUTPUT/summary.json"
echo V2_REWEIGHTED_VECTOR_FIXED_JOB_DONE
date
