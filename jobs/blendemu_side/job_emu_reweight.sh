#!/bin/bash
#SBATCH --job-name=emu_rw
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_rw_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd /home/z/Zekang.Zhang/blendemu
echo "### MECHANISM A: emulator distance-bias under-count on GOLD (cases 0-19, rd from cases 0-19) ###"
python -u scripts/emulator_mean_residual.py --max-case 19 2>&1 | grep -v "module command not found"
echo "======================================================================"
python -u scripts/emulator_gold_reweight.py --gold-cases 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --rd-max-case 19 2>&1 | grep -v "module command not found"
echo EMU_RW_DONE
