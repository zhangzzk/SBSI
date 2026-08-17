#!/bin/bash
#SBATCH --job-name=ext_pdist
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ext_pdist_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
cd /home/z/Zekang.Zhang/blendemu
echo "### EXTENDED emulator per-pair pred/true by DISTANCE (does it still under-predict <1\" close pairs?) ###"
python -u scripts/emulator_mean_residual.py --tag lsst_r_extnbr --max-case 9 2>&1 | grep -v "module command"
echo "======= for reference, PRODUCTION emulator ======="
python -u scripts/emulator_mean_residual.py --tag lsst_r --max-case 9 2>&1 | grep -v "module command"
echo EXT_PDIST_DONE
