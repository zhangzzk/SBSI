#!/bin/bash
#SBATCH --job-name=diag_cg
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_cg_%j.out

# DIAGNOSTIC (read-only, tunes nothing): metric-consistent constgold ±g central-diff response by size
# vs the flow's variable-shear training target -> go/no-go for a metric-consistent target rebuild.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### DIAG_CG job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_constgold_vs_varshear_target.py || { echo "DIAG_CG FAILED"; exit 1; }
echo "### DIAG_CG_DONE job=$SLURM_JOB_ID ###"; date
