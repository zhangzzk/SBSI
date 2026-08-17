#!/bin/bash
#SBATCH --job-name=val_crowd
#SBATCH --time=03:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_crowd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
TAG=${1:?tag}
M=models/measurement_flow_g0_ngmix_${TAG}_lam300_v1.pt
echo "### VALIDATE $TAG : half-shear g=0.05 (cases 0-39), binned by R_blend ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_c0-39.feather \
  --nominal-g 0.05 --max-rows 0 --n-samples 64 \
  --blend-lookup results/blend_lookup_hs_c0-39.feather 2>&1 \
  | grep -iE "BLEND-BIN|GLOBAL|INCOHERENT binned|Rblend-bin|ISO\(|\[0\.|>=0|BLENDED-ONLY" | grep -v module
echo "VAL_CROWD_DONE $TAG"
