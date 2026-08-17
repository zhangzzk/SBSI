#!/bin/bash
#SBATCH --job-name=size_lookup
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/size_lookup_%j.out

# DIAGNOSTIC helper (Goal 1): dump (case,input_index)->Re_input_p true-size lookup for cases>=40,
# to enable true-SIZE realistic cuts in eval_selection_robustness.py. No model / certified artifact.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### SIZE_LOOKUP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/build_true_size_lookup.py || { echo "SIZE_LOOKUP FAILED"; exit 1; }
echo "### SIZE_LOOKUP_JOB_DONE job=$SLURM_JOB_ID ###"; date
