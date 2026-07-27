#!/bin/bash
#SBATCH --job-name=hs_flowfig
#SBATCH --time=00:40:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_flowfig_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
echo "### HS_FLOWFIG job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
python -B -u scripts/eval_halfshear_flowfig.py --max-case 19 --gtag g0.02 \
  --ckpt /project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_sw_th400_seed421_joint.pt \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_flowfig_th400.npz
echo "### HS_FLOWFIG_DONE job=$SLURM_JOB_ID ###"; date
