#!/bin/bash
#SBATCH --job-name=infer_etilde
#SBATCH --time=12:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/infer_etilde_%j.out
# Tip: the flow-eval stage is 3-8x faster on a newer GPU than v100 (and TF32 kicks in);
# pick one at submit time, e.g.:  sbatch --gpus-per-node=a100:1 jobs/job_infer_etilde.sh
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# Posterior-mean shape estimator etilde = E[e|ehat,theta_hat] (WORKLOG cont.26):
# the cont.22 deployment object, prototyped for the true-neighbour conditional case.
# Stages: closure (formula unit test [+ optional grid convergence]) -> g0
# (null+calibration) -> gold (antithetic constant render, m_etilde under
# intrinsic/sheared/selfcal priors).
# First full run = job 15047211 (v100): closure 41^2 m=-0.21+/-0.31%, 61^2 m=+0.18+/-0.31%,
# g0 null ok / MSE ratio 0.34, gold sheared m=+0.6976+/-0.0398%, K=0.201.
# This version: GPU reweights + Aitken selfcal (4 sweeps instead of 30 slow iterations)
# + fp16 loglike disk cache -- with a warm cache the gold prior blocks rerun in minutes
# with NO flow evaluations (delete the cache files after changing model/grid/rows/seed).
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
FL=results/crowd_flux_conc_c0-199.feather          # near/far/MAX (matches conc feature set)
BL=results/blend_lookup_extnbrho_c40-139.feather   # R_blend for BINNING only (no additive term)
ROWS_CLO=${ROWS_CLO:-500000}
ROWS_G0=${ROWS_G0:-1000000}
ROWS_GOLD=${ROWS_GOLD:-8000000}   # 2M (+/-0.08% on m) is plenty for prior experiments
RUN_CONV=${RUN_CONV:-0}           # 41^2-vs-61^2 closure convergence: done in job 15047211
RUN_VALID=${RUN_VALID:-1}         # closure + g0 stages (skip for gold-only reruns)
CACHE_DIR=${CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
mkdir -p "$CACHE_DIR"
LL_CACHE=${LL_CACHE:-$CACHE_DIR/etilde_ll_c40-139_r${ROWS_GOLD}_n41}
COMMON="--measurement-model $MODEL --chunk 1024"

if [ "$RUN_VALID" = 1 ]; then
  echo "### STAGE 1a: closure, 41x41 grid, ${ROWS_CLO} draws ###"; date
  stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode closure $COMMON \
    --max-rows $ROWS_CLO --grid-n 41 2>&1 | grep --line-buffered -vE "module command"
  date

  if [ "$RUN_CONV" = 1 ]; then
    echo "### STAGE 1b: closure, 61x61 grid (convergence check, same seed/draws) ###"; date
    stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode closure $COMMON \
      --max-rows $ROWS_CLO --grid-n 61 2>&1 | grep --line-buffered -vE "module command"
    date
  fi

  echo "### STAGE 2: g0 null + calibration, ${ROWS_G0} rows ###"; date
  stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode g0 $COMMON \
    --max-rows $ROWS_G0 --grid-n 41 2>&1 | grep --line-buffered -vE "module command"
  date
fi

echo "### STAGE 3: GOLD c40-139, ${ROWS_GOLD} rows, 3 priors (loglike cache: $LL_CACHE) ###"; date
stdbuf -oL -eL python -u scripts/infer_posterior_shape.py --mode gold $COMMON \
  --max-rows $ROWS_GOLD --grid-n 41 \
  --crowd-flux-lookup $FL --blend-lookup $BL \
  --priors intrinsic sheared selfcal --n-boot 300 \
  --loglike-cache "$LL_CACHE" \
  --dump results/etilde_gold_c40-139.feather 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
