#!/bin/bash
#SBATCH --job-name=blk28_pilot
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk28_pilot_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/build_blend_lookup.py --cases 0 --output results/blend_lookup_const28_c0.feather 2>&1 | grep -v module
echo BLK28_PILOT_DONE
