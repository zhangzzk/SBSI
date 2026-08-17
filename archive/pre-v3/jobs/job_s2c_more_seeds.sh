#!/bin/bash
#SBATCH --job-name=s2c_more
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_more_s%a_%j.out

# lt=500 coupling model, seeds 505-509 (to pair with existing s2_true4d swaavg seeds
# 505-509 for a seed-noise-cancelled paired test of the pin's effect on constgold m).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
SEEDS=(505 506 507 508 509)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LT=500
FS=g0_meas_crowd_conc_szfl_noz
TAG=ablate_s2c_coupling_lt${LT}
RESP=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
echo "### S2C MORE lt=$LT seed=$SEED ###"; nvidia-smi -L; date
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $CAT --output $OUT \
  --target-column detected --selection-name sextractor_detected --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 --weight-decay 1e-5 \
  --seed "$SEED" --num-workers 8 --gpu-resident \
  --response-weight 450 --response-delta 0.02 --response-difference central --response-target-npz "$RESP" \
  --coupling-weight "$LT" --coupling-target-npz "$COUP" \
  || { echo "FAILED seed=$SEED"; exit 1; }
echo "S2C_DONE lt=$LT seed=$SEED"; date
