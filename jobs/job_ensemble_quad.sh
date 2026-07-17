#!/bin/bash
#SBATCH --job-name=ens_quad
#SBATCH --time=14:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
# NOTE (cont.28): do NOT pin a40 -- this training is CPU/IO-bound and barely uses the GPU, so ANY GPU
# works (one serial model ~1.6G fits a p5000/2080ti/v100/h100 alike). Pinning a40 queued 3/4 jobs behind
# other users while idle v100/h100 nodes (88-128 idle CPUs, far less contended) sat unused. gpu:1 lets the
# scheduler spread onto whichever node is free -> starts immediately AND runs faster (less I/O contention).
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens_quad_%j.out

# SEED-ENSEMBLE, <=4-GPU polite packing (WORKLOG cont.28). ONE job holds ONE a40 and trains 3 seeds
# SERIALLY on it, writing a per-seed log. 4 such jobs (seeds 501-512) => 12-model ensemble on 4 GPUs.
# NOTE (cont.28): an earlier CONCURRENT variant (3 seeds backgrounded on one a40) was ~8x slower per
# epoch -- this training is CPU/data-pipeline-bound (a40 sat at 0% util, ~1.6G/model), so co-locating 3
# just tripled CPU+I/O contention (3x re-reading the 45M-row feather) instead of filling an idle GPU.
# Serial gives each seed the full 24-CPU allocation -> ~1.5h/seed, ~4.5h/job. Harvest R_flow from the 12 logs.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

SEEDS=${SEEDS:?set SEEDS (space-separated, e.g. "501 502 503")}
FS=${FS:-g0_meas_crowd_conc_szfl}
TAGBASE=${TAG:-meas_szfl}
LAM=300; DELTA=0.02; EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
MP=results/meas_prim_lookup_c0-139.feather
FL=results/crowd_flux_conc_c0-199.feather
PERLOG=/home/z/Zekang.Zhang/logs/ens_seed_${SLURM_JOB_ID}   # per-seed logs: ${PERLOG}_s<seed>.out

echo "### ENS_QUAD job=$SLURM_JOB_ID  seeds='$SEEDS'  FS=$FS  (3 concurrent on 1 a40) ###"; nvidia-smi -L; date

train_and_val () {
  local SEED=$1
  local TAG=${TAGBASE}_ens_s${SEED}
  local OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt
  local LOG=${PERLOG}_s${SEED}.out
  {
    echo "### ENSEMBLE SEED $SEED  FS=$FS  TAG=$TAG ###"; date
    python -u scripts/train_measurement_model.py \
      --catalogue $CAT --output $OUT \
      --target-column detected --selection-name sextractor_detected \
      --feature-set "$FS" \
      --target-features measured_ngmix_g1 measured_ngmix_g2 \
      --flow-type mean_affine --mean-hidden 128 \
      --flow-blind-features e1_input_p e2_input_p \
      --shear-case 0.0 --max-rows 0 --epochs "$EPOCHS" --batch-size 8192 \
      --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
      --seed "$SEED" --num-workers 8 --gpu-resident \
      --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
      --response-target-npz "$RESP" || { echo "ENS_SEED TRAIN FAILED seed=$SEED"; exit 1; }
    echo "TRAIN_DONE seed=$SEED"; date
    stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
      --measurement-model $OUT \
      --catalogue $CD/constant_response_catalogue_c40-139.feather \
      --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
      --crowd-flux-lookup $FL --meas-prim-lookup $MP \
      --ood-lookup results/ood_split_c40-139.feather \
      --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
      --global-only \
      --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
      | grep --line-buffered -vE "module command"
    date; echo "ENS_SEED_DONE seed=$SEED MODEL=$OUT"
  } > "$LOG" 2>&1
}

# SERIAL: one seed at a time so each gets the full CPU allocation (workload is CPU/IO-bound, not GPU-bound).
for s in $SEEDS; do echo "=== starting seed $s -> ${PERLOG}_s${s}.out ==="; date; train_and_val "$s"; done
echo "ENS_QUAD_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS'"; date
