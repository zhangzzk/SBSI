#!/bin/bash
#SBATCH --job-name=aug_crowd_t
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/aug_crowd_t_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "### augment TEST g0.02 (cases 0-39) ###"
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.02_test.feather \
  --flux-lookup results/crowd_flux_c0-39.feather --blend-lookup results/blend_lookup_hs_c0-39.feather \
  --max-case 39 --output $D/det_meas_crowd_g0.02_test_c0-39.feather 2>&1 | grep -v module
echo AUG_CROWD_T_DONE
