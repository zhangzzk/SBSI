#!/bin/bash
#SBATCH --job-name=aug_crowd_train_full
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/aug_crowd_train_full_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
FL=results/crowd_flux_c0-199.feather
BL=results/blend_lookup_c0-199.feather
OUT=$D/det_meas_crowd_g0.0_train_full.feather

echo "### augment TRAIN g0.0 (full cases 0-199) ###"
date
python -u scripts/augment_crowding.py \
  --catalogue $D/det_meas_ngmix_np7_g0.0_train.feather \
  --flux-lookup $FL \
  --blend-lookup $BL \
  --output $OUT 2>&1 | grep -v module
date
echo "AUG_CROWD_TRAIN_FULL_DONE $OUT"
