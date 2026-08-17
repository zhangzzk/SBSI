#!/bin/bash
#SBATCH --job-name=emu_mean_pos160
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_mean_pos160_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_mean_pos160_%j.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/score_emulator_global_mean.py \
  --config configs/fs2_lsst_r_extnbr_v22.yaml \
  --source-tag lsst_r_extnbr_v22 \
  --candidate-tag lsst_r_extnbr_v22_rpowposa0065 \
  --minimum-case 40 \
  --output results/emulator_global_mean_v22_rpowposa0065_c40-199.json
