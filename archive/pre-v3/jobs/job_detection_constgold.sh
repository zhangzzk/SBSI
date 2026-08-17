#!/bin/bash
#SBATCH --job-name=cg_detbias
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_detbias_%j.out

# Truth-only constgold DETECTION-bias evaluation (Stage-3): full per-leg detection catalogues vs the
# both-detected intersection. det-bias = R_full/R_both - 1. CPU-only. See eval_detection_constgold.py.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG DETBIAS job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_constgold.py \
  --min-case 40 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_constgold_v1.npz
echo "CG_DETBIAS_JOB_DONE"; date
