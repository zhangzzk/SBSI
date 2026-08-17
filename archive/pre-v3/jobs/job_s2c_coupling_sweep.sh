#!/bin/bash
#SBATCH --job-name=s2c_sweep
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_sweep_t%a_%j.out

# S2 + flux/size orientation-coupling pin (dims 2,3). lam_theta sweep, seed 501.
# Everything else byte-identical to S2 (6x3x5 shape pin UNCHANGED). Coupling target 6x9x5 rblend.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation

LAMS=(20 100 500)
LAMT=${LAMS[$SLURM_ARRAY_TASK_ID]}
SEED=501
FS=g0_meas_crowd_conc_szfl_noz
TAG=ablate_s2c_coupling_lt${LAMT}
RESP=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
OUT=$OUTDIR/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt

echo "### S2C COUPLING job=$SLURM_JOB_ID lam_theta=$LAMT seed=$SEED -> $OUT ###"; nvidia-smi -L; date
[ -f "$RESP" ] || { echo "MISSING RESP"; exit 1; }
[ -f "$COUP" ] || { echo "MISSING COUP"; exit 1; }

python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $CAT --output $OUT \
  --target-column detected --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 --weight-decay 1e-5 \
  --seed "$SEED" --num-workers 8 --gpu-resident \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz "$RESP" \
  --coupling-weight "$LAMT" --coupling-target-npz "$COUP" \
  || { echo "S2C FAILED lam=$LAMT"; exit 1; }
echo "S2C_DONE lam=$LAMT seed=$SEED MODEL=$OUT"; date
