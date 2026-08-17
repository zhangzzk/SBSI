#!/bin/bash
#SBATCH --job-name=selrobust
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/selrobust_%j.out

# Acceptance harness (DIAGNOSTIC / EVALUATION only): worst-case |m| of the parameter-free
# estimator over a FIXED a-priori selection family, on the certification per-object dump.
# Baselines the CURRENT certified estimator; --rflow-override / --rblend-override swap in the
# joint-flow harvest for the head-to-head. Never wired into m; tunes nothing.
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

echo "### SELROBUST job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py "$@" \
  || { echo "SELROBUST FAILED"; exit 1; }
echo "### SELROBUST_DONE job=$SLURM_JOB_ID ###"; date
