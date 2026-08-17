#!/bin/bash
#SBATCH --job-name=flow_v29cg
#SBATCH --array=0-9%5
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v29cg_%A_%a.out
set -euo pipefail

# Controlled two-seed screen of response-grid resolution and training-population size.
# The existing V2.2 6x6x5/4M checkpoints are the sixth (control) cell and are not retrained.
# For a fixed seed, capped arms receive the same priority sample and all-data arms receive the
# same rows/split, so grid comparisons are paired at the catalogue level.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEEDS=(501 502)
ARMS=(rb5_all rbc8_s6_cap4m rbc8_s6_all rbc8_s4_cap4m rbc8_s4_all)
ARM_INDEX=$((TASK / ${#SEEDS[@]}))
SEED_INDEX=$((TASK % ${#SEEDS[@]}))
(( ARM_INDEX < ${#ARMS[@]} )) || { echo "NO ARM for task $TASK"; exit 1; }
ARM=${ARMS[$ARM_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

BASE_TARGET=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
COND6_TARGET=results/v22_rblend_bin_design_conditional8_target.npz
COND4_TARGET=results/v22_rblend_bin_design_conditional8_size4_target.npz
case "$ARM" in
  rb5_all)         TARGET=$BASE_TARGET;  MAXROWS=0       ;;
  rbc8_s6_cap4m)   TARGET=$COND6_TARGET; MAXROWS=4000000 ;;
  rbc8_s6_all)     TARGET=$COND6_TARGET; MAXROWS=0       ;;
  rbc8_s4_cap4m)   TARGET=$COND4_TARGET; MAXROWS=4000000 ;;
  rbc8_s4_all)     TARGET=$COND4_TARGET; MAXROWS=0       ;;
  *) echo "UNKNOWN ARM $ARM"; exit 1 ;;
esac

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
OUT=$CACHE/measurement_flow_g0_ngmix_${ARM}_s${SEED}.pt
mkdir -p "$CACHE"
for f in "$CAT" "$TARGET" "$COUP"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] && [ ! -e "${OUT%.pt}_swaavg.pt" ] || {
  echo "REFUSING overwrite $OUT"; exit 1;
}

echo "### V2.9 CONDITIONAL GRID SCREEN arm=$ARM seed=$SEED maxrows=$MAXROWS ###"
echo "target=$TARGET output=$OUT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi -L
date
"$PY" -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue "$CAT" --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows "$MAXROWS" --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
  --weight-decay 1e-5 --swa-last-k 8 --seed "$SEED" --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz "$TARGET" \
  --response-error absolute --response-rel-floor 0.05 --response-bin-ema 0.0 \
  --coupling-weight 500 --coupling-target-npz "$COUP"
echo "V29_CONDITIONAL_GRID_FLOW_DONE arm=$ARM seed=$SEED"
date
