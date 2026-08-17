#!/bin/bash
#SBATCH --job-name=pilot_train
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/pilot_train_%j.out

# TRAIN-ONLY seed pilot (WORKLOG cont.35). Generalizes job_ensemble_quad: env knobs FS/TAG/LAM/WD so
# one script covers z-drop + response-weight + weight-decay variants. Trains SEEDS serially on 1 GPU
# (workload is CPU/IO-bound; serial gives each seed the full CPU allocation). No inline 45M val here
# (that OOM'd on shared nodes) -> harvest separately with job_pilot_harvest.sh reading these models.
# Model naming (new clean convention): models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI

SEEDS=${SEEDS:?set SEEDS (space-separated, e.g. "501 502 503")}
FS=${FS:-g0_meas_crowd_conc_szfl_noz}
TAG=${TAG:?set TAG (e.g. meas_szfl_noz_lam300)}
LAM=${LAM:-300}; WD=${WD:-1e-5}; DELTA=${DELTA:-0.02}; EPOCHS=${EPOCHS:-80}
RESP=${RESP:-results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather

echo "### PILOT_TRAIN job=$SLURM_JOB_ID seeds='$SEEDS' FS=$FS TAG=$TAG LAM=$LAM WD=$WD RESP=$RESP ###"
nvidia-smi -L; date

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
  echo "===== TRAIN seed=$SEED -> $OUT ====="; date
  python -u scripts/train_measurement_model.py \
    --catalogue $CAT --output $OUT \
    --target-column detected --selection-name sextractor_detected \
    --feature-set "$FS" \
    --target-features measured_ngmix_g1 measured_ngmix_g2 \
    --flow-type mean_affine --mean-hidden 128 \
    --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 --max-rows 0 --epochs "$EPOCHS" --batch-size 8192 \
    --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
    --weight-decay "$WD" \
    --seed "$SEED" --num-workers 8 --gpu-resident \
    --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
    --response-target-npz "$RESP" || { echo "PILOT_TRAIN FAILED seed=$SEED"; exit 1; }
  echo "TRAIN_DONE seed=$SEED MODEL=$OUT"; date
done
echo "PILOT_TRAIN_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS' TAG=$TAG"; date
