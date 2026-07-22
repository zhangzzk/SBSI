#!/bin/bash
#SBATCH --job-name=g3_collider
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/g3_collider_%j.out

# GOAL 3 collider de-risk (DIAGNOSTIC): infer true mag/size from measured observables (OOS), cut on
# the inferred truth, compare certified m vs true-cut (oracle) vs naive measured cut. No model/certified
# artifact touched; regressions are proxies for P(theta|x). Firewall-safe feasibility test.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### G3_COLLIDER job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/goal3_collider_derisk.py --tag scene || { echo "G3_COLLIDER FAILED"; exit 1; }
echo "### G3_COLLIDER_JOB_DONE job=$SLURM_JOB_ID ###"; date
