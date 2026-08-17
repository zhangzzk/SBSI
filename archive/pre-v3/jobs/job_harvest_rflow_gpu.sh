#!/bin/bash
#SBATCH --job-name=harvest_gpu
#SBATCH --time=03:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/harvest_gpu_%j.out

# GPU HARVEST of GLOBAL R_flow for the whole seed ensemble (WORKLOG cont.29/30). The in-job --global-only
# OOM-killed at 48G because it loaded the FULL 45M-row constant feather (~20G DataFrame) + 45M x 64 reshear.
# FIX: subsample --max-rows 3000000 -> DataFrame ~1.3G + 3M x 64 reshear fits easily in 90G, and a global
# mean over 3M rows has bootstrap error ~0.0006 (negligible vs the +/-0.01 per-seed scatter we resolve).
# GPU reshear of 3M x 64 is seconds/seed, so all 9 seeds finish in minutes. Prints one GLOBAL line per seed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI

SEEDS=${SEEDS:-501 502 503 504 505 506 507 508 509}
TAGBASE=${TAG:-meas_szfl}
MAXROWS=${MAXROWS:-3000000}
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
MP=results/meas_prim_lookup_c0-139.feather
FL=results/crowd_flux_conc_c0-199.feather

echo "### HARVEST_GPU job=$SLURM_JOB_ID seeds='$SEEDS' maxrows=$MAXROWS ###"; nvidia-smi -L; date

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAGBASE}_ens_s${SEED}_lam300_v1.pt
  echo "===== HARVEST seed=$SEED model=$OUT ====="; date
  if [ ! -f "$OUT" ]; then echo "MISSING MODEL seed=$SEED"; continue; fi
  stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
    --measurement-model $OUT \
    --catalogue $CD/constant_response_catalogue_c40-139.feather \
    --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
    --crowd-flux-lookup $FL --meas-prim-lookup $MP \
    --ood-lookup results/ood_split_c40-139.feather \
    --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
    --global-only \
    --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows "$MAXROWS" --batch-size 16384 --n-boot 300 2>&1 \
    | grep --line-buffered -vE "module command" \
    | stdbuf -oL sed "s/^/[s${SEED}] /"
  echo "HARVEST_DONE seed=$SEED"; date
done
echo "HARVEST_GPU_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS'"; date
