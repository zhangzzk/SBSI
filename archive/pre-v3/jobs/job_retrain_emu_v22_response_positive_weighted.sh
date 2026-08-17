#!/bin/bash
#SBATCH --job-name=emu_v22_pos3
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --array=0-2%3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_pos3_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_pos3_%A_%a.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22.yaml
export HELDOUT_MIN_CASE=40
export RESPONSE_WEIGHT_CAP=50
export RESPONSE_WEIGHT_MODE=positive_square
case "$SLURM_ARRAY_TASK_ID" in
  0) export RESPONSE_WEIGHT_ALPHA=0.065 RESPONSE_WEIGHT_TOKEN=posa0065 ;;
  1) export RESPONSE_WEIGHT_ALPHA=0.10 RESPONSE_WEIGHT_TOKEN=posa010 ;;
  2) export RESPONSE_WEIGHT_ALPHA=0.15 RESPONSE_WEIGHT_TOKEN=posa015 ;;
  *) echo "unexpected array task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/retrain_emulator_v22_response_weighted.py
