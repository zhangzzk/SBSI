#!/bin/bash
#SBATCH --job-name=s2c_more
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_more_s%a_%j.out

# EIGHT MORE SEEDS for the Gold-V2 coupling-pinned flow, taking the ensemble from 8 to 16.
#
# WHY (WORKLOG 2026-07-28n RESULT 17): the 8-seed ensembles give m = -0.460 +- 0.353% with the
# certified R_blend and -1.248 +- 0.347% with the corrected one. Per-seed sd is ~1.0%, so at n=8 the
# error bar is comparable to the |m| <= 0.3% target itself -- the ABSOLUTE m is simply not resolved.
# n=16 halves the sem to ~0.25%. (The DIFFERENCE between emulators is already pinned at
# -0.787 +- 0.016 and needs no more seeds; only the absolute does.)
#
# Byte-identical to jobs/job_s2c_lt500_seeds.sh, which produced the existing 8 checkpoints
# (501-503, 505-509): same tree (SBSI-ablation), same script, same catalogue, feature set, targets,
# architecture, epochs, lambda, response and coupling targets. ONLY the seed list differs. Run from
# SBSI-ablation rather than this worktree ON PURPOSE: the worktree's
# train_measurement_model_swa_s1_truecond.py is a superset (adds --primary-mag-max/--primary-re-min
# and a response-bin EMA, all no-ops at default), and mixing two script versions inside one ensemble
# is exactly the kind of subtle inconsistency that is not worth risking here.
#
# Writes both ..._s${SEED}.pt and ..._s${SEED}_swaavg.pt; the constgold chain uses _swaavg.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
SEEDS=(510 511 512 513 514 515 516 517)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LT=500
FS=g0_meas_crowd_conc_szfl_noz
TAG=ablate_s2c_coupling_lt${LT}
RESP=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
if [ -f "$OUT" ]; then echo "REFUSING to overwrite existing $OUT"; exit 1; fi
echo "### S2C lt=$LT seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
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
