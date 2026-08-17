#!/bin/bash
#SBATCH --job-name=ood_rsim
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --output=/home/z/Zekang.Zhang/logs/ood_rsim_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### FLOW-FREE: <R_sim>, <R_blend_emu>, deficit per bright/faint-OOD bin ###"
python -u scripts/ood_rsim_check.py 2>&1 | grep -v "module command"
echo OOD_RSIM_DONE
