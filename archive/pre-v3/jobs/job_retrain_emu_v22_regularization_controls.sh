#!/bin/bash
#SBATCH --job-name=emu_v22_regctl
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_regctl_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_regctl_%A_%a.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22.yaml
export HELDOUT_MIN_CASE=40
export RESPONSE_WEIGHT_ALPHA=0
export RESPONSE_WEIGHT_CAP=50
export RESPONSE_WEIGHT_MODE=square
export RESPONSE_WEIGHT_N_TREES=271
export RESPONSE_WEIGHT_GAMMA=0
if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  export RESPONSE_WEIGHT_TOKEN=unwtg0
else
  export RESPONSE_WEIGHT_TOKEN=unwtg0mc20
  export RESPONSE_WEIGHT_MIN_CHILD_WEIGHT=20
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/retrain_emulator_v22_response_weighted.py
