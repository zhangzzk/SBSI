#!/bin/bash
#SBATCH --job-name=cgv27
#SBATCH --array=0-39
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgv27_%A_%a.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}; NCHUNK=10
MODEL=$((TASK / NCHUNK)); CHUNK=$((TASK % NCHUNK))
ARMS=(sixshell sixshell purity purity); SEEDS=(501 502 501 502)
ARM=${ARMS[$MODEL]}; SEED=${SEEDS[$MODEL]}
MIN_CASE=$((40 + 10 * CHUNK)); MAX_CASE=$((MIN_CASE + 10))
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v27_${ARM}_s${SEED}_swaavg.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
LOOKUP=results/blend_lookup_v22_c40-139.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
OUTDIR=$C/constgold_scores; mkdir -p "$OUTDIR"
OUT=$OUTDIR/${ARM}_s${SEED}_c${MIN_CASE}-${MAX_CASE}.feather
SHELLS=(logflux_abs_shell_0_0p5 logflux_abs_shell_0p5_1 logflux_abs_shell_1_2 \
        logflux_abs_shell_2_3 logflux_abs_shell_3_5 logflux_abs_shell_5_10)
if [ "$ARM" = sixshell ]; then COLS=("${SHELLS[@]}"); else COLS=(true_blendedness_eq17); fi
for f in "$CK" "$CAT" "$LOOKUP" "$CROWD" "$C/scene_features_const_c40-139.feather"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case "$MIN_CASE" --max-case "$MAX_CASE" \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --blend-lookup "$LOOKUP" --crowd-flux-lookup "$CROWD" \
  --extra-feature-lookup "$C/scene_features_const_c40-139.feather" \
  --extra-feature-columns "${COLS[@]}" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 0 --batch-size 16384 \
  --dump "$OUT"
echo V27_CONSTGOLD_DONE arm=$ARM seed=$SEED cases=$MIN_CASE-$MAX_CASE; date
