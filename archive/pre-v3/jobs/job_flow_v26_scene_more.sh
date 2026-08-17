#!/bin/bash
#SBATCH --job-name=flow_v26more
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-11
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v26more_%A_%a.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

# Continue the repository's established 16-seed convention: 501--503, skip 504, then 505--517.
# Seeds 501/502/503/505 are the completed blind screen; train only the twelve missing members.
SEEDS=(506 507 508 509 510 511 512 513 514 515 516 517)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?array task required}]}
TAG=ablate_s2c_lt500_v26_scene_target
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
RESP=results/response_target_scene_top2_thirdplus_snc_c0-99_6x6x5x4cond_v26.npz
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_scene_g0.0_train_full.feather
OUT=$CACHE/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
for f in "$RESP" "$COUP" "$CAT"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] && [ ! -e "${OUT%.pt}_swaavg.pt" ] || { echo "REFUSING to overwrite seed $SEED"; exit 1; }

echo "### V2.6 JOINT SCENE TARGET extension seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
"$PY" -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue "$CAT" --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz_top2_thirdplus \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
  --weight-decay 1e-5 --swa-last-k 8 \
  --seed "$SEED" --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz "$RESP" \
  --response-error absolute --response-rel-floor 0.05 --response-bin-ema 0.0 \
  --coupling-weight 500 --coupling-target-npz "$COUP"
echo V26_SCENE_MORE_FLOW_DONE; date
