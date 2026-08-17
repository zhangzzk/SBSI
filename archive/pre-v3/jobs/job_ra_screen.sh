#!/bin/bash
#SBATCH --job-name=ra_screen
#SBATCH --time=01:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ra_screen_%j.out

# STAGE P2: warm-start SCREEN for the realisation-aware (RA) head. GO/NO-GO, not a certification.
#
# The RA shift A(c,u) is zero-initialised, so loading a fiducial dom6x6 checkpoint with
# strict=False reproduces that model BIT FOR BIT and the warm start is exact. 20 epochs at ~3 min
# per seed is enough to see whether A engages at all.
#
# P2 PASS: held-out count-weighted rms of rho_model/rho_sim - 1 falls by >=2x versus the fiducial
# (measured in the SAME run by the RA-weight-0 reference below); AND g=0 val NLL does not degrade
# by more than 0.01 nats; AND |A| > 0 with per-draw response std > 0.
#
# NO m MAY BE QUOTED FROM THIS JOB. Warm-started seeds are correlated with their parents.
#
# NEW TAG + NEW DIRECTORY on purpose: writes to sbsi_caches/ra/, tag `ra_dom6x6_v1`, so it cannot
# collide with the certified `ablation/..._dom6x6_s{seed}_swaavg.pt` set.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

A=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra
mkdir -p "$OUTD"
R=/home/z/Zekang.Zhang/SBSI/results
RESP=$R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz
COUP=$R/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
RA=${RA:-$OUTD/ra_target_g0meas_c0-79.npz}
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
TAG=${TAG:-ra_dom6x6_v1}

SEEDS=(${SEEDS:-501 502})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; else SEED=${SEED:-501}; fi
WARM=$A/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${SEED}_swaavg.pt
[ -f "$RA" ]   || { echo "missing RA target $RA -- run jobs/job_ra_target.sh first"; exit 1; }
[ -f "$WARM" ] || { echo "missing warm-start checkpoint $WARM"; exit 1; }

# The RA screen runs TWICE per seed: once with lambda_ra = 0 (the fiducial architecture, RA target
# attached as a READ-ONLY diagnostic -> the rho closure the fiducial model already has), and once
# with the RA head engaged. The pass criterion is the RATIO of those two rho_rms values measured in
# the same session, so nothing is compared across runs or machines.
run () {   # $1=label $2=ra_hidden $3=ra_weight
  local L="$1"; local H="$2"; local LAM="$3"
  local OUT=$OUTD/measurement_flow_g0_ngmix_${TAG}_${L}_s${SEED}.pt
  if [ -f "$OUT" ]; then echo "REFUSING to overwrite existing $OUT"; exit 1; fi
  echo; echo "################ $L  (ra_hidden=$H, lambda_ra=$LAM, seed=$SEED) ################"
  python -u scripts/train_measurement_model_swa_s1_truecond.py \
    --catalogue $CAT --output "$OUT" \
    --target-column detected --selection-name sextractor_detected --feature-set g0_meas_crowd_conc_szfl_noz \
    --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
    --flow-type mean_affine_ra --ra-hidden "$H" --ra-target-npz "$RA" --ra-weight "$LAM" \
    --ra-warm-start "$WARM" \
    --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 --max-rows 4000000 --epochs ${EPOCHS:-20} --batch-size 8192 \
    --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr ${LR:-0.0007} --patience 20 --weight-decay 1e-5 \
    --seed "$SEED" --num-workers 8 --gpu-resident \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --response-weight 450 --response-delta 0.02 --response-difference central --response-target-npz "$RESP" \
    --coupling-weight 500 --coupling-target-npz "$COUP" \
    || { echo "FAILED $L seed=$SEED"; exit 1; }
}

echo "### RA SCREEN tag=$TAG seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
run refblind 0 0                      # fiducial architecture; RA target read out only
run raon ${RAH:-128} ${LAMRA:-1.0}    # RA head engaged
echo "RA_SCREEN_DONE seed=$SEED"; date
