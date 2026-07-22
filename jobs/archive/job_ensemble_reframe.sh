#!/bin/bash
#SBATCH --job-name=ens_reframe
#SBATCH --time=00:30:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens_reframe_%j.out
# cont.102: certified(16-seed) vs sqrt-wt reframed-loss ENSEMBLE per-cut closure, contiguous windows.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### ENS_REFRAME job=$SLURM_JOB_ID ###"; date
stdbuf -oL -eL python -B -u scripts/ensemble_closure_reframe.py || { echo "ENS_REFRAME FAILED"; exit 1; }
echo "### ENS_REFRAME_DONE ###"; date
