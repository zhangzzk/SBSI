#!/bin/bash
#SBATCH --job-name=cgv29cg
#SBATCH --array=0-99%20
#SBATCH --time=01:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgv29cg_%A_%a.out
set -euo pipefail

# The V2.2 control already has these exact constgold shards, so score only the five new arms.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEEDS=(501 502)
ARMS=(rb5_all rbc8_s6_cap4m rbc8_s6_all rbc8_s4_cap4m rbc8_s4_all)
NCHUNK=10
MODEL=$((TASK / NCHUNK))
CHUNK=$((TASK % NCHUNK))
(( MODEL < ${#ARMS[@]} * ${#SEEDS[@]} )) || { echo "NO MODEL for task $TASK"; exit 1; }
ARM_INDEX=$((MODEL / ${#SEEDS[@]}))
SEED_INDEX=$((MODEL % ${#SEEDS[@]}))
ARM=${ARMS[$ARM_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
MIN_CASE=$((40 + 10 * CHUNK))
MAX_CASE=$((MIN_CASE + 10))

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
CK=$C/measurement_flow_g0_ngmix_${ARM}_s${SEED}_swaavg.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
LOOKUP=results/blend_lookup_v22_c40-139.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
OUTDIR=$C/constgold_scores
OUT=$OUTDIR/${ARM}_s${SEED}_c${MIN_CASE}-${MAX_CASE}.feather
mkdir -p "$OUTDIR"
for f in "$CK" "$CAT" "$LOOKUP" "$CROWD"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V2.9 CONSTGOLD arm=$ARM seed=$SEED cases=$MIN_CASE-$MAX_CASE ###"; date
"$PY" -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case "$MIN_CASE" --max-case "$MAX_CASE" \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --blend-lookup "$LOOKUP" --crowd-flux-lookup "$CROWD" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 0 --batch-size 16384 \
  --dump "$OUT"
echo "V29_CONDITIONAL_CONSTGOLD_DONE arm=$ARM seed=$SEED cases=$MIN_CASE-$MAX_CASE"; date
