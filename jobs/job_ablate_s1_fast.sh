#!/bin/bash
#SBATCH --job-name=ablate_s1
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ablate_s1_s%a_%j.out

# ABLATION STEP 1 -- measured->true conditioning swap (single knob vs Step 0/V1).
# Uses train_measurement_model_swa_s1_truecond.py, which is byte-identical to the certified
# V1 trainer EXCEPT it swaps the two measured primary conditioners
# (measured_mag_auto, measured_flux_radius) for their TRUE counterparts (r_input_p, Re_input_p).
# Same feature set flag, same 3 seeds, same LAM/DELTA/EPOCHS/catalogue/RESP as Step 0.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation

SEED=$((501 + SLURM_ARRAY_TASK_ID))
FS=g0_meas_crowd_conc_szfl_noz
TAG=ablate_s1_truecond
LAM=450; WD=1e-5; DELTA=0.02; EPOCHS=80
RESP=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
mkdir -p "$OUTDIR"

echo "### ABLATE_S1 (true-cond) job=$SLURM_JOB_ID seed=$SEED FS=$FS TAG=$TAG LAM=$LAM ###"
nvidia-smi -L; date
[ -f "$RESP" ] || { echo "MISSING RESP $RESP"; exit 1; }
[ -f "$CAT" ]  || { echo "MISSING CAT $CAT"; exit 1; }

  OUT=$OUTDIR/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
  echo "===== TRAIN seed=$SEED -> $OUT ====="; date
  python -u scripts/train_measurement_model_swa_s1_truecond.py \
    --catalogue $CAT --output $OUT \
    --target-column detected --selection-name sextractor_detected \
    --feature-set "$FS" \
    --target-features measured_ngmix_g1 measured_ngmix_g2 \
    --flow-type mean_affine --mean-hidden 128 \
    --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 --max-rows 4000000 --epochs "$EPOCHS" --batch-size 8192 \
    --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
    --weight-decay "$WD" \
    --seed "$SEED" --num-workers 8 --gpu-resident \
    --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
    --response-target-npz "$RESP" || { echo "ABLATE_S1 FAILED seed=$SEED"; exit 1; }
  echo "TRAIN_DONE seed=$SEED MODEL=$OUT"; date
echo "ABLATE_S1_ALL_DONE job=$SLURM_JOB_ID seed=$SEED TAG=$TAG"; date
