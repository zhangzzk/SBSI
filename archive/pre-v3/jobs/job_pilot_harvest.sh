#!/bin/bash
#SBATCH --job-name=pilot_harvest
#SBATCH --time=03:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/pilot_harvest_%j.out

# GPU HARVEST of GLOBAL R_flow for pilot models (WORKLOG cont.35). Matches job_pilot_train naming:
# models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt. Prints one GLOBAL line per seed -> average offline.
# Row-batched GPU reshear (--batch-size 16384) so 45M rows fit; expandable_segments avoids fragmentation.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEEDS=${SEEDS:?set SEEDS (space-separated, e.g. "501 502 503")}
TAG=${TAG:?set TAG (e.g. meas_szfl_noz_lam300)}
MAXROWS=${MAXROWS:-45000000}
FLOWSEED=${FLOWSEED:-12345}   # torch seed for CRN R_flow legs (same seed both legs -> noise cancels)
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
# R_sim leg MUST be the FIXED-centroid constant catalogue. constant_response_catalogue_c40-139.feather
# is rebuilt by build100's concat from the OLD 07-09/07-10 half-catalogues (raw c40-79 resp 0.4549 = OLD
# convention -> R_sim=0.4655), silently reverting the centroid fix. The genuinely-fixed constant catalogue
# is constant_response_catalogue_train.feather (const_s4_c2fix 07-15; raw c40-79 resp 0.4422 = FIXED ->
# R_sim=0.4534 after min-case 40). Default to it + held-out split; override with CAT/MINCASE if needed.
CAT=${CAT:-$CD/constant_response_catalogue_train.feather}
MINCASE=${MINCASE:-40}
MP=results/meas_prim_lookup_c0-139.feather
FL=results/crowd_flux_conc_c0-199.feather

echo "### PILOT_HARVEST job=$SLURM_JOB_ID seeds='$SEEDS' TAG=$TAG maxrows=$MAXROWS ###"; nvidia-smi -L; date

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
  echo "===== HARVEST seed=$SEED model=$OUT ====="; date
  if [ ! -f "$OUT" ]; then echo "MISSING MODEL seed=$SEED"; continue; fi
  stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
    --measurement-model $OUT \
    --catalogue $CAT --min-case $MINCASE \
    --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
    --crowd-flux-lookup $FL --meas-prim-lookup $MP \
    --ood-lookup results/ood_split_c40-139.feather \
    --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
    --global-only --flow-seed "$FLOWSEED" \
    --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows "$MAXROWS" --batch-size 16384 --n-boot 300 2>&1 \
    | grep --line-buffered -vE "module command" \
    | stdbuf -oL sed "s/^/[s${SEED}] /"
  echo "HARVEST_DONE seed=$SEED"; date
done
echo "PILOT_HARVEST_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS' TAG=$TAG"; date
