#!/bin/bash
#SBATCH --job-name=aug_crowd
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/aug_crowd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
FL=results/crowd_flux_c0-39.feather
BL=results/blend_lookup_hs_c0-39.feather   # r_blend (input field shared across sim sets)

echo "### augment TRAIN g0.0 (cases 0-39) ###"
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.0_train.feather \
  --flux-lookup $FL --blend-lookup $BL --max-case 39 \
  --output $D/det_meas_crowd_g0.0_train_c0-39.feather 2>&1 | grep -v module

echo "### augment VAL g0.05 (cases 0-39) ###"
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.05_val.feather \
  --flux-lookup $FL --blend-lookup $BL --max-case 39 \
  --output $D/det_meas_crowd_g0.05_val_c0-39.feather 2>&1 | grep -v module
echo AUG_CROWD_DONE
