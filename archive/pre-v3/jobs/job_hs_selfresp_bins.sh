#!/bin/bash
#SBATCH --job-name=hs_selfr
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_selfr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
echo "### FLOW SELF-RESPONSE on INCOHERENT half-shear (no coherent blend) per R_blend bin ###"
echo "### isolates the crowd_flux flow's self-response error -> its contribution to gold q3 (+10.9%) ###"
python -u scripts/validate_allpairs_response.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_c0-39.feather \
  --blend-lookup results/blend_lookup_hs_c0-39.feather \
  --rb-edges 0.02 0.1 0.25 0.5 \
  --max-case 39 --n-samples 128 2>&1 | grep -v "module command"
echo HS_SELFR_DONE
