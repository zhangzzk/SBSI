#!/bin/bash
#SBATCH --job-name=stage1_7as
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/stage1_7as_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
NN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather
echo "### STAGE1_7AS job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
# 7"-isolation diagnostic (R_blend=0 for isolated => no blend-lookup needed for the m_iso headline).
# ckpt-glob = default 4-seed truecut ensemble; bridge=1.0 (no empirical factor).
python -B -u scripts/eval_constgold_closure.py \
  --bridge 1.0 \
  --nn-lookup "$NN" --iso-radius 7.0 \
  --max-case 39 --true-re-min 0.3 --true-mag-max 26 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/stage1_cg_7as_iso.npz
echo "### STAGE1_7AS_DONE job=$SLURM_JOB_ID ###"; date
