#!/bin/bash
#SBATCH --job-name=train_conc
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_conc_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

# ROOT-CAUSE flow retrain (confirm-first positive, WORKLOG cont.15): add nbr_flux_max (flux
# concentration / brightest-neighbour dominance) to the crowd-flux conditioning so the flow can
# bend its self-response for a bright close blend. Same snc100_central02_lam300 recipe as the
# baseline measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1, only feature-set + cat differ.
FS=g0_crowd_flux_conc
TAG=crowdflux_conc_snc100_central02
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN CONC $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
echo "catalogue=$CAT"; echo "response_target=$RESP"; date
python -u scripts/train_measurement_model.py \
  --catalogue $CAT \
  --output $OUT \
  --target-column detected \
  --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine \
  --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 \
  --max-rows 0 \
  --epochs "$EPOCHS" \
  --batch-size 8192 \
  --hidden-dim 256 \
  --condition-layers 3 \
  --n-flows 10 \
  --lr 0.0007 \
  --patience 10 \
  --num-workers 8 \
  --response-weight "$LAM" \
  --response-delta "$DELTA" \
  --response-difference central \
  --response-target-npz "$RESP" || { echo "TRAIN_CONC FAILED"; exit 1; }
date
echo "TRAIN_CONC_DONE $OUT"
