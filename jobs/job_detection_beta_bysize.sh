#!/bin/bash
#SBATCH --job-name=cg_betasz
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_betasz_%j.out

# Detection bias vs beta binned by SIZE (does size-binning rescue beta?), mag band 25.5-26.5.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG BETASZ job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_beta_bysize.py --min-case 40 --mag-lo 25.5 --mag-hi 26.5 \
  --n-size 3 --n-beta 6 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_beta_bysize_v1.npz
echo "CG_BETASZ_JOB_DONE"; date
