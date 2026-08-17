#!/bin/bash
#SBATCH --job-name=hsext_va
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsext_va_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### FLOW SELF-RESPONSE (incoherent half-shear) binned by EXTNBR R_blend (SAME axis+edges as gold) ###"
echo "### gold per-bin m was ISO -2.2 / q1 -2.8 / q2 +2.6 / q3 +10.9 / q4 +1.4 %; this = flow error ALONE ###"
python -u scripts/validate_allpairs_response.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_c0-39.feather \
  --blend-lookup results/blend_lookup_hs_extnbr_c0-39.feather \
  --rb-edges 0.02 0.0745 0.2327 0.7045 \
  --max-case 39 --n-samples 128 2>&1 | grep -v "module command"
echo HSEXT_VA_DONE
