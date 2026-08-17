#!/bin/bash
#SBATCH --job-name=hs_rbdiag_ap7
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_rbdiag_ap7_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### HALF-SHEAR g=0.05 ALL-PAIRS (AP7) flow, NO SNC, binned by emulator R_blend (cases 0-39) ###"
python -u scripts/validate_allpairs_response.py \
  --measurement-model models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather \
  --nominal-g 0.05 --max-rows 0 --max-case 39 --n-samples 64 \
  --blend-lookup results/blend_lookup_hs_c0-39.feather 2>&1 \
  | grep -iE "BLEND-BIN|GLOBAL|Rblend-bin|ISO\(|\[0|>=|INCOHERENT binned|N_eff|BLENDED-ONLY" | grep -v module
echo HS_RBDIAG_AP7_DONE
