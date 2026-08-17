#!/bin/bash
#SBATCH --job-name=blmult
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blmult_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date
python -u scripts/build_blend_multiplicity.py --cases $(seq 0 39) \
  --tag lsst_r_extnbr_ho --output results/blend_multiplicity_extnbrho_c0-39.feather 2>&1 | grep -v "module command"
date; echo BLMULT_DONE
