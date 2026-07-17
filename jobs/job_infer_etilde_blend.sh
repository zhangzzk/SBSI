#!/bin/bash
#SBATCH --job-name=infer_etilde_blend
#SBATCH --time=12:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/infer_etilde_blend_%j.out
# Faster on a newer GPU: sbatch --gpus-per-node=a100:1 jobs/job_infer_etilde_blend.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# COMPLETES the cont.22 posterior + starts the probblend ladder (WORKLOG cont.31/32).
# Big artifacts (caches, dumps) go to $DATA_DIR (sbsi_caches), NOT $HOME -- first
# submission 15062408 was killed by a home-quota EDQUOT from a results/ write.
# Stages: A g0 calibration (RUN_G0=0 to skip; done in 15062408) -> B gold conditional,
# both prior conditionings, Aitken selfcal, cache build -> C closure with the MARGINAL
# likelihood (tower unit test) -> D gold blind-MARGINAL (ladder lower bracket) ->
# E ladder RUNG 1: detected-only neighbour fluxes (drop undetected; the etilde analogue
# of PROB_BLENDING's +3.29% census step).
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
FL=results/crowd_flux_conc_c0-199.feather
FL_DET=results/crowd_flux_det_c40-139.feather   # symlink -> $DATA_DIR/sbsi_caches
BL=results/blend_lookup_extnbrho_c40-139.feather
ROWS_G0=${ROWS_G0:-1000000}
ROWS_GOLD=${ROWS_GOLD:-4000000}   # anchors are differential vs 15047211's 8M -> 4M suffices
ROWS_CLOM=${ROWS_CLOM:-300000}
ROWS_MARG=${ROWS_MARG:-1000000}   # blind marginal = bracket only -> 1M suffices
ROWS_RUNG=${ROWS_RUNG:-1000000}
MARG_M=${MARG_M:-8}
RUN_G0=${RUN_G0:-1}
CACHE_DIR=${CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
mkdir -p "$CACHE_DIR"
COMMON="--measurement-model $MODEL --chunk 1024"

if [ "$RUN_G0" = 1 ]; then
  echo "### STAGE A: g0 calibration, global vs rmag prior, ${ROWS_G0} rows ###"; date
  stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode g0 $COMMON \
    --max-rows $ROWS_G0 --grid-n 41 --prior-conditioning global rmag \
    2>&1 | grep --line-buffered -vE "module command"
  date
fi

echo "### STAGE B: GOLD conditional, ${ROWS_GOLD} rows, 3 priors x {global,rmag} (cache build) ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --max-rows $ROWS_GOLD --grid-n 41 \
  --crowd-flux-lookup $FL --blend-lookup $BL \
  --priors intrinsic sheared selfcal --prior-conditioning global rmag --n-boot 300 \
  --loglike-cache "$CACHE_DIR/etilde_ll_c40-139_r${ROWS_GOLD}_n41" \
  --dump "$CACHE_DIR/etilde_gold_cond_c40-139.feather" 2>&1 | grep --line-buffered -vE "module command"
date

echo "### STAGE C: closure with dtheta_b MARGINAL likelihood, M=${MARG_M}, ${ROWS_CLOM} draws ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode closure $COMMON \
  --max-rows $ROWS_CLOM --grid-n 41 --marginal-m $MARG_M \
  2>&1 | grep --line-buffered -vE "module command"
date

echo "### STAGE D: GOLD blind-MARGINAL (bracket), M=${MARG_M}, ${ROWS_MARG} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --max-rows $ROWS_MARG --grid-n 41 --marginal-m $MARG_M \
  --crowd-flux-lookup $FL --blend-lookup $BL \
  --priors intrinsic sheared selfcal --prior-conditioning global rmag --n-boot 300 \
  --loglike-cache "$CACHE_DIR/etilde_ll_marg${MARG_M}_c40-139_r${ROWS_MARG}_n41" \
  --dump "$CACHE_DIR/etilde_gold_marg_c40-139.feather" 2>&1 | grep --line-buffered -vE "module command"
date

echo "### STAGE E: RUNG 1 -- detected-only neighbour fluxes, ${ROWS_RUNG} rows ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --max-rows $ROWS_RUNG --grid-n 41 \
  --crowd-flux-lookup $FL_DET --blend-lookup $BL \
  --priors intrinsic sheared selfcal --prior-conditioning global rmag --n-boot 300 \
  --loglike-cache "$CACHE_DIR/etilde_ll_det_c40-139_r${ROWS_RUNG}_n41" \
  --dump "$CACHE_DIR/etilde_gold_rung1_c40-139.feather" 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
