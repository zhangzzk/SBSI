#!/bin/bash
#SBATCH --job-name=const_coh_tgt
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/const_coh_tgt_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
# COHERENT response target from constant cases 0-19 (train split); validate later on 20-39
python -u scripts/compute_response_target_constant.py \
  --blend-lookup results/blend_lookup_const28_c0-39.feather --max-case 19 \
  --n-flux 6 --n-size 3 --n-crowd 5 --min-count 200 \
  --output results/response_target_const_coh_6x3x5.npz 2>&1 | grep -v module
echo CONST_COH_TGT_DONE
