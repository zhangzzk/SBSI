#!/bin/bash
#SBATCH --job-name=v22_posweak
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --array=0-3%4
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_posweak_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_posweak_%A_%a.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH="$ROOT/configs/fs2_lsst_r_extnbr_v22.yaml"
export HELDOUT_MIN_CASE=40
export RESPONSE_WEIGHT_CAP=50
export RESPONSE_WEIGHT_MODE=positive_square
case "$SLURM_ARRAY_TASK_ID" in
  0) export RESPONSE_WEIGHT_ALPHA=0.010 RESPONSE_WEIGHT_TOKEN=posw0010 ;;
  1) export RESPONSE_WEIGHT_ALPHA=0.020 RESPONSE_WEIGHT_TOKEN=posw0020 ;;
  2) export RESPONSE_WEIGHT_ALPHA=0.035 RESPONSE_WEIGHT_TOKEN=posw0035 ;;
  3) export RESPONSE_WEIGHT_ALPHA=0.050 RESPONSE_WEIGHT_TOKEN=posw0050 ;;
  *) echo "unexpected array task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac
cd "$ROOT"
"$PY" -u scripts/retrain_emulator_v22_response_weighted.py

