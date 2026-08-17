#!/bin/bash
#SBATCH --job-name=flow_v27sh
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v27sh_%A_%a.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
IFS=',' read -r -a SEEDS <<< "${V27_SEEDS:-501,502}"
(( TASK < ${#SEEDS[@]} )) || { echo "NO SEED for task $TASK"; exit 1; }
SEED=${SEEDS[$TASK]}
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=$C/det_meas_sixshell_v27_g0.0_train_v22domain.feather
TARGET=$C/response_target_v27_sixshell.joblib
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
OUT=$CACHE/measurement_flow_g0_ngmix_ablate_s2c_lt500_v27_sixshell_s${SEED}.pt
for f in "$CAT" "$TARGET" "$COUP"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] && [ ! -e "${OUT%.pt}_swaavg.pt" ] || { echo "REFUSING overwrite"; exit 1; }
echo "### V2.7 sixshell seed=$SEED ###"; nvidia-smi -L; date
"$PY" -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue "$CAT" --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz_sixshell \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
  --weight-decay 1e-5 --swa-last-k 8 --seed "$SEED" --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-perobj "$TARGET" \
  --response-error absolute --response-rel-floor 0.05 --response-bin-ema 0.0 \
  --coupling-weight 500 --coupling-target-npz "$COUP"
echo V27_SIXSHELL_FLOW_DONE seed=$SEED; date
