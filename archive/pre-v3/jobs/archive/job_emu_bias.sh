#!/bin/bash
#SBATCH --job-name=emu_bias
#SBATCH --time=00:25:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_bias_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/blendemu
date; python -u /home/z/Zekang.Zhang/SBSI/scripts/confirm_emulator_bias.py || { echo EMU_BIAS_FAILED; exit 1; }; date
