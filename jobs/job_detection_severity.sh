#!/bin/bash
#SBATCH --job-name=cg_detsev
#SBATCH --time=00:40:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_detsev_%j.out

# Truth-only constgold DETECTION-bias RESOLVED BY BLENDING SEVERITY (Stage-3). Bins the blended
# population by blendedness / distance_scaled / flux ratio (x true primary mag) and measures
# det-bias = R_full/R_both - 1 per bin. CPU-only. See scripts/eval_detection_severity.py.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG DETSEV job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_severity.py \
  --min-case 40 --nbins 6 --kind shint \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_severity_v1.npz
echo "CG_DETSEV_JOB_DONE"; date
