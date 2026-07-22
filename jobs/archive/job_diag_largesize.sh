#!/bin/bash
#SBATCH --job-name=diag_lgsz
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_lgsz_%j.out

# Goal-1 (task #32): decisive diagnostic to split the stable size_gt1.0 -1.4..-2.0% bias into
# phi-COVERAGE (fixable by a feature) vs SIM/R_flow floor (conclude not feasible for the extreme
# large-size tail). Localizes the residual by isolation / neighbour geometry / primary brightness
# using the iteration-2 (winner) R_blend. Diagnostic only; certified untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/z/Zekang.Zhang/SBSI
echo "### DIAG_LGSZ job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_largesize_residual.py || { echo "DIAG_LGSZ FAILED"; exit 1; }
echo "### DIAG_LGSZ_DONE job=$SLURM_JOB_ID ###"; date
