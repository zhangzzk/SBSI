#!/bin/bash
#SBATCH --job-name=emu_shift
#SBATCH --time=00:50:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_shift_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; python -u scripts/emu_domain_shift.py || { echo EMU_SHIFT_FAILED; exit 1; }; date
echo EMU_SHIFT_JOB_DONE
