#!/bin/bash
#SBATCH --job-name=s2cdom
#SBATCH --time=01:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2cdom_%j.out

# Gold-V2 (s2c coupling lt500) retrained INSIDE THE DELIVERABLE DOMAIN: primary true mag < 26.0
# and primary true Re > 0.3 (neighbours stay full-population, per GOALS.md). Everything else is
# byte-identical to jobs/job_s2c_lt500_seeds.sh, which produced the certified lt500 checkpoints:
# same catalogue, feature set, targets, architecture, epochs, lambda, response + coupling targets.
# ONE seed first -- validate in-domain, then decide whether to spend more seeds.
#
# Runs from THIS worktree, not SBSI-ablation: `scripts/train_measurement_model_swa_s1_truecond.py`
# and `sbs_shear/` were verified identical between the two trees (diff -rq, excluding __pycache__),
# so the only difference from the baseline run is the two domain flags.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

# Seed: explicit $SEED, or picked from the array index when run with --array (SEEDS may be
# overridden from the submit line, e.g. SEEDS="502 503").
SEEDS=(${SEEDS:-501 502 503 505 506 507 508 509})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
  SEED=${SEED:-501}
fi
LT=500
FS=g0_meas_crowd_conc_szfl_noz
TAG=${TAG:-ablate_s2c_coupling_lt${LT}_dom}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
# The response target MUST be built on the same domain as the training cut -- a full-population
# target straddles the cut and pins the survivors to a mean that includes rows the trainer never
# sees (WORKLOG 2026-07-27d: that cost 5% of R_flow in-domain).
RESP=${RESP:-/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5_dom.npz}
COUP=/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUT=$D/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt

echo "### S2C-DOMAIN lt=$LT seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $CAT --output $OUT \
  --target-column detected --selection-name sextractor_detected --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden ${MH:-128} --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs 80 --batch-size ${BS:-8192} \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 --weight-decay 1e-5 \
  --seed "$SEED" --num-workers 8 --gpu-resident \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --response-weight ${RW:-450} --response-delta 0.02 --response-difference central --response-target-npz "$RESP" \
  --response-error ${RERR:-absolute} --response-rel-floor ${RFLOOR:-0.05} \
  --coupling-weight "$LT" --coupling-target-npz "$COUP" \
  || { echo "FAILED seed=$SEED"; exit 1; }
echo "S2CDOM_DONE lt=$LT seed=$SEED"; date
