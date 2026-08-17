#!/bin/bash
#SBATCH --job-name=flow_v30ns
#SBATCH --array=0-1%2
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v30ns_%A_%a.out
set -euo pipefail

# Paired seed-501 screen: the response target is the only training-recipe change.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
ARMS=(v30_m10_c16 v30_m12_c16)
TARGETS=(
  results/v30_rblend_mag10_nosize_c16_target.npz
  results/v30_rblend_mag12_nosize_c16_target.npz
)
ARM=${ARMS[$TASK]:?unknown task}
TARGET=${TARGETS[$TASK]:?unknown task}
SEED=501
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
OUT=$CACHE/measurement_flow_g0_ngmix_${ARM}_s${SEED}.pt
mkdir -p "$CACHE"
for file in "$CAT" "$TARGET" "$COUP"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] && [ ! -e "${OUT%.pt}_swaavg.pt" ] || {
  echo "REFUSING overwrite $OUT"; exit 1;
}

echo "### V3.0 NO-SIZE GRID TRAIN arm=$ARM seed=$SEED all_rows target=$TARGET ###"; date
nvidia-smi -L
"$PY" -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue "$CAT" --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 0 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
  --weight-decay 1e-5 --swa-last-k 8 --seed "$SEED" --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz "$TARGET" \
  --response-error absolute --response-rel-floor 0.05 --response-bin-ema 0.0 \
  --coupling-weight 500 --coupling-target-npz "$COUP"
echo "V30_NOSIZE_FLOW_DONE arm=$ARM seed=$SEED"; date
