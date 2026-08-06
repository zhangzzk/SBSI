#!/bin/bash
#SBATCH --job-name=flow_v22
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v22_s%a_%j.out
set -euo pipefail

# Exact V2 flow recipe with one scientific lever: primary true r<25.8 and Re>0.5 arcsec.
# This deliberately keeps V2's 4M cap, 80 epochs, patience 10 and default SWA-8; it does not
# inherit V2.1's longer 120-epoch/SWA-32 recipe.
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
# cip A40-*Q vGPU profiles do not expose the CUDA virtual-memory API used by expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(${SEEDS:-501 502})
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; else SEED=${SEED:-501}; fi
TAG=${TAG:-ablate_s2c_lt500_v22}
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
RESP=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUT="$CACHE/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt"
for f in "$RESP" "$COUP" "$CAT"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] && [ ! -e "${OUT%.pt}_swaavg.pt" ] || { echo "REFUSING to overwrite seed $SEED"; exit 1; }

echo "### V2.2 FLOW seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
"$PY" -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue "$CAT" --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
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
echo "V22_FLOW_DONE seed=$SEED"; date
