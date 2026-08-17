#!/bin/bash
#SBATCH --job-name=emu_v22_g005
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_g005_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_g005_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH=$ROOT/configs/fs2_lsst_r_extnbr_v22_g005.yaml
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_g005.json

[ ! -e "$META" ] || { echo "REFUSING to overwrite $META"; exit 1; }
cd "$ROOT"
echo "### TRAIN V2.2 g=0.05 PAIR-RESPONSE EMULATOR job=$SLURM_JOB_ID ###"
date
"$PY" -u scripts/retrain_emulator_v22_g005.py
echo EMU_V22_G005_JOB_DONE
date
