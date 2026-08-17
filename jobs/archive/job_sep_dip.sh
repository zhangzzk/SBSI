#!/bin/bash
#SBATCH --job-name=sepdip
#SBATCH --time=01:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sepdip_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; stdbuf -oL -eL python -u scripts/audit_sep_dip.py 2>&1 | grep -v "module command"; date; echo SEPDIP_JOBDONE
