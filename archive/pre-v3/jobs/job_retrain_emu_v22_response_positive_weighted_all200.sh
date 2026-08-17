#!/bin/bash
#SBATCH --job-name=emu_v22_pos_a200
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_pos_a200_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_pos_a200_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22.yaml

# One fixed reweighting pass.  Only alpha remains a future tuning parameter.
export HELDOUT_MIN_CASE=0
export RESPONSE_WEIGHT_ALPHA=0.065
export RESPONSE_WEIGHT_TOKEN=posa0065_all200
export RESPONSE_WEIGHT_CAP=50
export RESPONSE_WEIGHT_MODE=positive_square
export RESPONSE_WEIGHT_N_TREES=271

cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/retrain_emulator_v22_response_weighted.py
