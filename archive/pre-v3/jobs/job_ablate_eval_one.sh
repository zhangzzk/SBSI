#!/bin/bash
#SBATCH --job-name=abl_evalone
#SBATCH --time=00:25:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/abl_evalone_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI-ablation
TAG=$1
echo "### ABL_EVALONE tag=$TAG job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/eval_selfresp_gap.py --ckpt-glob "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_${TAG}_s*_swaavg.pt"
echo "### ABL_EVALONE_DONE tag=$TAG ###"; date
