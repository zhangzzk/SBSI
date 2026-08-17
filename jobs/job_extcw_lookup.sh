#!/bin/bash
#SBATCH --job-name=extcw_lk
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=small
#SBATCH --output=/home/z/Zekang.Zhang/logs/extcw_lk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### BUILD extended-domain gold blend lookup (cases 0-39); per-case R_blend mean should EXCEED production 0.26 ###"
python -u scripts/build_blend_lookup.py --cases $(seq 0 39) --tag lsst_r_extnbr_cw \
  --output results/blend_lookup_extnbrcw_c0-39.feather 2>&1 | grep -v "module command"
if [ -f results/blend_lookup_extnbrcw_c0-39.feather ]; then echo EXTCW_LK_DONE; else echo "EXTCW_LK_FAILED (no output file)"; exit 1; fi
