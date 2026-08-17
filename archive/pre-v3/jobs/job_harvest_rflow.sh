#!/bin/bash
#SBATCH --job-name=harvest_rflow
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harvest_rflow_%j.out

# HARVEST R_flow from saved ensemble models (WORKLOG cont.29). ROOT CAUSE of the missing GLOBAL
# lines: the in-job --global-only validation OOM-killed at 48G on the cip GPU nodes (which have only
# 26-41G RAM). The 6.6GB constant feather -> ~20G DataFrame plus the 45M x 64 reshear arrays blew the
# 48G cap; SIGKILL leaves no traceback, so the log showed lookups then silence then ENS_SEED_DONE.
# FIX: validate on a cip null-GRES node (252G RAM) on CPU (--mem=200G naturally routes off the small
# GPU nodes; no GPU contention with the 3 running training jobs that hold this user's 3-GPU cap).
# Subsample rows (--max-rows) to keep CPU reshear fast; a global mean over 12M rows is amply precise.
# Pass MODELS (space-separated seed numbers). Prints one GLOBAL line per seed -> average offline.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEEDS=${SEEDS:?set SEEDS (space-separated seed numbers, e.g. "501 504 507")}
TAGBASE=${TAG:-meas_szfl}
MAXROWS=${MAXROWS:-3000000}
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
MP=results/meas_prim_lookup_c0-139.feather
FL=results/crowd_flux_conc_c0-199.feather

echo "### HARVEST_RFLOW job=$SLURM_JOB_ID seeds='$SEEDS' maxrows=$MAXROWS device=cpu ###"; date; free -g | head -2

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAGBASE}_ens_s${SEED}_lam300_v1.pt
  echo "===== HARVEST seed=$SEED model=$OUT ====="; date
  if [ ! -f "$OUT" ]; then echo "MISSING MODEL seed=$SEED"; continue; fi
  stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
    --measurement-model $OUT --device cpu \
    --catalogue $CD/constant_response_catalogue_train.feather --min-case 40 \
    --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
    --crowd-flux-lookup $FL --meas-prim-lookup $MP \
    --ood-lookup results/ood_split_c40-139.feather \
    --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
    --global-only \
    --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows "$MAXROWS" --n-boot 300 2>&1 \
    | grep --line-buffered -vE "module command" \
    | stdbuf -oL sed "s/^/[s${SEED}] /"
  echo "HARVEST_DONE seed=$SEED"; date
done
echo "HARVEST_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS'"; date
