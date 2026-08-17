#!/bin/bash
#SBATCH --job-name=hsext_lk
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=small
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsext_lk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### BUILD half-shear blend lookup with EXTENDED emulator (same tag as gold, for matched-axis binning) ###"
python -u scripts/build_blend_lookup.py --cases $(seq 0 39) \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 --sign 0.05 --tag lsst_r_extnbr \
  --output results/blend_lookup_hs_extnbr_c0-39.feather 2>&1 | grep -v "module command"
if [ -f results/blend_lookup_hs_extnbr_c0-39.feather ]; then echo HSEXT_LK_DONE; else echo "HSEXT_LK_FAILED"; exit 1; fi
