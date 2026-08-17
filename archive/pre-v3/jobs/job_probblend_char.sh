#!/bin/bash
#SBATCH --job-name=pbchar
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/pbchar_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date
python -u scripts/probblend_characterize.py --cases 0 1 2 3 4 5 6 7 \
  --output results/probblend_char.feather 2>&1 | grep -v "module command"
date
echo PBCHAR_JOB_DONE
