#!/bin/bash
#SBATCH --job-name=emu_v22_groupg0
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_groupg0_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_groupg0_%j.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22.yaml
export HELDOUT_MIN_CASE=40
export GROUP_VECTOR_N_TREES=271
export GROUP_VECTOR_GAMMA=0
export GROUP_VECTOR_TAG=lsst_r_extnbr_v22_groupvecg0
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/retrain_emulator_v22_group_vector.py
