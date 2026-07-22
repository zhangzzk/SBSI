#!/bin/bash
#SBATCH --job-name=ensbias
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ensbias_%j.out

# CLOSING TEST (owner points 1+2): certified pipeline's REAL-DATA (ensemble-metric) bias on the gentle
# acceptance sample. m_ens = R_total / <R_flow+R_blend> - 1 on the g=0.05 leg. R_flow = certified self-
# response (CPU inference, small flow); R_blend transferred per (mag x blend) cell from constgold
# (r_sim NEVER read). Firewall-safe; no training. CPU to dodge the GPU queue.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI
echo "### ENSBIAS(cpu) job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/eval_ensemble_bias_leg.py --max-rows 2000000 --n-samples 32 --batch-size 16384 \
  || { echo "ENSBIAS FAILED"; exit 1; }
echo "### ENSBIAS_DONE job=$SLURM_JOB_ID ###"; date
