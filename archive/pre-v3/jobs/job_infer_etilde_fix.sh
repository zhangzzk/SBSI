#!/bin/bash
#SBATCH --job-name=infer_etilde_fix
#SBATCH --time=06:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/infer_etilde_fix_%j.out
# A/B/C for the two cont.33 pipeline fixes, SAME 1M rows (seed 7):
#   A: 41x41 baseline (reference at these rows)
#   B: 61x61            -> isolates the grid-quadrature fix (delta_grid -0.43% -> +0.04%)
#   C: 61x61 + mu-corr  -> adds the g0-measured location patch U(e; r-mag cell)
# Verdict criterion: how much of the per-mag m_she pattern (esp. ISO mid-bright
# +3.4/+3.95%) survives after B and C.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
FL=results/crowd_flux_conc_c0-199.feather
BL=results/blend_lookup_extnbrho_c40-139.feather
ROWS=${ROWS:-1000000}
CACHE_DIR=${CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
MUCORR=${MUCORR:-$CACHE_DIR/mu_correction_conc_v1.npz}
COMMON="--measurement-model $MODEL --chunk 1024 --max-rows $ROWS \
  --crowd-flux-lookup $FL --blend-lookup $BL \
  --priors intrinsic sheared selfcal --prior-conditioning global --n-boot 300"

echo "### STAGE A: 41x41 baseline, ${ROWS} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --grid-n 41 --loglike-cache "$CACHE_DIR/etilde_ll_fixA41_r${ROWS}" \
  --dump "$CACHE_DIR/etilde_gold_fixA41.feather" 2>&1 | grep --line-buffered -vE "module command"
date

echo "### STAGE B: 61x61 (grid fix), ${ROWS} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --grid-n 61 --loglike-cache "$CACHE_DIR/etilde_ll_fixB61_r${ROWS}" \
  --dump "$CACHE_DIR/etilde_gold_fixB61.feather" 2>&1 | grep --line-buffered -vE "module command"
date

echo "### STAGE C: 61x61 + mu-correction, ${ROWS} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --grid-n 61 --mu-correction "$MUCORR" \
  --loglike-cache "$CACHE_DIR/etilde_ll_fixC61mu_r${ROWS}" \
  --dump "$CACHE_DIR/etilde_gold_fixC61mu.feather" 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
