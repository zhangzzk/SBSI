#!/bin/bash
#SBATCH --job-name=etilde_rung1_61
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/etilde_rung1_61_%j.out
# Rung-1 verification at 61x61 (WORKLOG cont.33/34): stage E of 15064545 found the
# DEPLOYABLE conditioning (detected-only neighbour fluxes) lands at she -0.20+/-0.13,
# selfcal -0.62+/-0.62 GLOBAL at 41x41 -- but the 41-grid quadrature term (-0.43% at
# w->0) is still inside its faint bins.  This rerun removes it; if rung 1 stays
# subpercent per-bin-flat(ter), the deployment estimator is genuinely close.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
FL_DET=results/crowd_flux_det_c40-139.feather
BL=results/blend_lookup_extnbrho_c40-139.feather
ROWS=${ROWS:-1000000}
CACHE_DIR=${CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}

echo "### RUNG 1 at 61x61, ${ROWS} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold \
  --measurement-model $MODEL --chunk 1024 --max-rows $ROWS --grid-n 61 \
  --crowd-flux-lookup $FL_DET --blend-lookup $BL \
  --priors intrinsic sheared selfcal --prior-conditioning global rmag --n-boot 300 \
  --loglike-cache "$CACHE_DIR/etilde_ll_det61_c40-139_r${ROWS}" \
  --dump "$CACHE_DIR/etilde_gold_rung1_61.feather" 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
