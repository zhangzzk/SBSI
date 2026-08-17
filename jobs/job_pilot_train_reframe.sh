#!/bin/bash
#SBATCH --job-name=train_reframe
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_reframe_%j.out

# cont.101: CONDITIONAL-CALIBRATION ("reframed") loss test. IDENTICAL to job_pilot_train.sh EXCEPT it
# exposes the new gated response-loss knobs: RESPERR (absolute|relative), BWPOW (bin-weight power:
# 1.0=certified count-weight, 0.0=every occupied bin equal), MINCNT (per-batch sparse-bin guard).
# Goal: down-weight the populous bright/small bins so the starved LARGE/FAINT/ISOLATED cells drive the
# fit -> per-cut (conditional) calibration, not just GLOBAL m~0. Built on the szfine non-circular target
# (finer size). Firewall-clean (constgold never read in training). Certified path unchanged.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEEDS=${SEEDS:?set SEEDS}
FS=${FS:-g0_meas_crowd_conc_szfl_noz}
TAG=${TAG:?set TAG}
LAM=${LAM:-450}; WD=${WD:-1e-5}; DELTA=${DELTA:-0.02}; EPOCHS=${EPOCHS:-80}
RESP=${RESP:-results/response_target_crowd_rblend_snc_c0-99_szfine.npz}
RESPERR=${RESPERR:-absolute}      # absolute (certified) | relative (m-targeting)
BWPOW=${BWPOW:-1.0}               # 1.0 certified count-weight | 0.0 equal-per-bin
MINCNT=${MINCNT:-0}               # per-batch sparse-bin mask (use ~30 with BWPOW<1)
ANCHOR=${ANCHOR:-0}               # global-mean anchor weight (0=off; pins GLOBAL m~0 with BWPOW<1)
MHID=${MHID:-128}                 # mean-head hidden width (certified 128; raise to test capacity floor)
NFLOWS=${NFLOWS:-10}              # n coupling flows (certified 10)
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather

echo "### TRAIN_REFRAME job=$SLURM_JOB_ID seeds='$SEEDS' FS=$FS TAG=$TAG LAM=$LAM RESP=$RESP RESPERR=$RESPERR BWPOW=$BWPOW MINCNT=$MINCNT ANCHOR=$ANCHOR MHID=$MHID NFLOWS=$NFLOWS ###"
nvidia-smi -L; date

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
  echo "===== TRAIN seed=$SEED -> $OUT ====="; date
  python -u scripts/train_measurement_model.py \
    --catalogue $CAT --output $OUT \
    --target-column detected --selection-name sextractor_detected \
    --feature-set "$FS" \
    --target-features measured_ngmix_g1 measured_ngmix_g2 \
    --flow-type mean_affine --mean-hidden "$MHID" \
    --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 --max-rows 0 --epochs "$EPOCHS" --batch-size 8192 \
    --hidden-dim 256 --condition-layers 3 --n-flows "$NFLOWS" --lr 0.0007 --patience 10 \
    --weight-decay "$WD" \
    --seed "$SEED" --num-workers 8 --gpu-resident \
    --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
    --response-error "$RESPERR" \
    --response-bin-weight-power "$BWPOW" --response-min-bin-count "$MINCNT" \
    --response-global-anchor "$ANCHOR" \
    --response-target-npz "$RESP" || { echo "TRAIN_REFRAME FAILED seed=$SEED"; exit 1; }
  echo "TRAIN_DONE seed=$SEED MODEL=$OUT"; date
done
echo "TRAIN_REFRAME_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS' TAG=$TAG"; date
