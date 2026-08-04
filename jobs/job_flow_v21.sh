#!/bin/bash
#SBATCH --job-name=flow_v21
#SBATCH --time=03:00:00
#SBATCH --mem=34G          # measured: MaxRSS 3-4.6G on the fiducial runs; cip GPU nodes cap at ~40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_v21_s%a_%j.out

# STEP 4 of V2.1: the measurement flow, retrained on the V2.1 domain. ONE SEED FIRST (owner).
#
# TWO CHANGES vs jobs/job_s2c_domain_train.sh, which produced the fiducial dom6x6 checkpoints.
# Both are deliberate and are listed here because everything else is byte-identical: same
# catalogue, feature set, targets, architecture, batch size, lr, weight decay, response weight,
# response delta/difference, coupling weight and coupling target.
#
# (1) DOMAIN. `--v21-domain` replaces `--primary-mag-max 26.0 --primary-re-min 0.3`. The V2.1
#     domain is primary true Re > 0.5" (2.5 px, resolution 0.474) AND true S/N > 10; the S/N half
#     is a CURVE in (mag, Re), so it comes from sbs_shear.domain rather than from this command
#     line -- see that module for why it cannot be retyped per consumer. The response target is
#     rebuilt on the SAME domain (job 15519919); a target built on a different population pins the
#     kept rows to a mean including rows the trainer never sees (WORKLOG 2026-07-27d, -5% R_flow).
#
# (2) LONGER TRAINING + WIDER SWA: 80 -> 120 epochs, swa-last-k 8 -> 32.
#     Measured, not assumed. Decomposing the 16 fiducial seeds' `*_train_curve.npz` into epoch
#     wobble (averageable) vs genuinely different solutions (not) gives, for the spread in
#     validation response across seeds:
#         SWA window     1 ep    8 ep (fiducial)   16 ep    32 ep
#         spread        0.81%       0.29%          0.25%    0.20%
#     -> an irreducible floor near 0.17%. So SWA-32 is worth roughly a FACTOR 2 IN SEED COUNT
#     (16 -> ~8 for the same ensemble error) and no more; it cannot go below the floor. Training
#
#     MEASURED OUTCOME (job 15527267, 4 V2.1 seeds): per-seed sd on constgold no-cut m = 0.467%,
#     against the fiducial SWA-8's 0.606% on its own domain -- a factor 1.3, in the ballpark of the
#     1.45 predicted above. Do NOT compare it instead to the 1.014% figure quoted elsewhere: that
#     is the fiducial SWA-8 dumps re-masked onto the V2.1 population, a different measurement.
#     The 0.467% has 3 dof (~+-40%); re-measure once more seeds exist.
#     longer is separately justified: validation NLL was still falling at epoch 80 (1.4447 ->
#     1.4377 between the 60-70 and 70-80 bands), i.e. the fiducial stopped while still improving.
#     The LR is constant (no schedule), which is exactly the regime where weight averaging pays --
#     the weights are still wandering at the end rather than frozen.
#
#     PATIENCE 10 -> 20 is a guard, not a lever: early stopping never triggered in the fiducial
#     (all 16 seeds ran the full 80 epochs), but a stop at, say, epoch 45 would make SWA-32 average
#     back to epoch 13 -- i.e. average in under-converged weights and quietly poison the
#     checkpoint. If a V2.1 seed does early-stop, that is a finding to report, not to absorb.
#
# COUPLING TARGET IS CARRIED OVER UNCHANGED, and that is a known inconsistency INHERITED from the
# fiducial rather than introduced here: response_target_theta_coupling_rblend_c0-99_6x9x5.npz is
# built on the FULL population (its size edges start at 0.10"), and job_s2c_domain_train.sh already
# paired it with an in-domain response target. There is no builder for it in scripts/, so
# rebuilding it on V2.1 is a separate piece of work. Flagged so it is not mistaken for a match.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# expandable_segments needs CUDA virtual-memory APIs the cip vGPU slices (A40-16Q etc.) lack.
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(${SEEDS:-501 502 503 505 506 507 508 509})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; else SEED=${SEED:-501}; fi
LT=500
FS=g0_meas_crowd_conc_szfl_noz
TAG=${TAG:-ablate_s2c_lt${LT}_v21}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
R=/home/z/Zekang.Zhang/SBSI/results
# 5x4x5, not the fiducial's 6x6x5: V2.1 keeps 47.5% of V2, so the 6x6x5 grid falls to min-cell
# N_eff = 521 -- below the 611 at which an earlier grid was already rejected, and far below the
# fiducial's 1,875 floor. 5x4x5 is the FINEST grid that clears that floor (min 2,791). Chosen on
# cell occupancy, never on constgold m (the R_blend firewall). Job 15519919.
RESP=${RESP:-$R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21.npz}
COUP=$R/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
OUT=$D/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
if [ -f "$OUT" ]; then echo "REFUSING to overwrite existing $OUT"; exit 1; fi

# ANCHOR (--response-global-anchor). DEFAULT 0 = byte-identical to the certified path and to the
# original seed-501 run (job 15520081), so this addition does not silently change anything already
# produced. Set ANCHOR>0 to switch it on.
#
# WHAT IT DOES. The per-cell response term penalises SQUARED per-bin errors, so the SIGNED aggregate
# sum_b w_b (Rmodel_b - Rsim_b) is not directly controlled (trainer comment at
# scripts/train_measurement_model_swa_s1_truecond.py:419). The anchor pins that signed aggregate.
#
# ON V2.1 IT IS A MEASURED NULL -- DO NOT REACH FOR IT HERE. Arms at ANCHOR=300/1000/3000 (jobs
# 15526057-59) and a control at RW=2000 (15526060) all landed on the ANCHOR=0 baseline:
# <R_model>(val) = 0.8509 / 0.8505 / 0.8499 / 0.8506 vs baseline 0.8498. At ANCHOR=3000 the anchor
# term is ~2.5 against an NLL of ~0.44 -- the largest term in the loss -- and it still moved nothing.
# The reason is simply that there was no signed error to remove: the flow was ALREADY on its
# population-weighted target (0.8509), so the anchor correctly found ~nothing to fix.
#
# The "2.9% deficit" that motivated these arms WAS AN ARTIFACT of the old epoch readout, which
# printed the cells-UNWEIGHTED bin_targets.mean() (0.8753) next to the POPULATION mean
# <R_model>(val). Those are different quantities; on this grid they differ by 2.79%. The readout now
# prints both, labelled. WORKLOG 2026-08-04j retracts 2026-08-04i.
#
# Weights default to TRAINING cell counts (--response-pop-weight-npz unset). That is right here
# because the training population and the deliverable population are the same V2.1 domain. It is
# still an approximation -- training is half-shear, the deliverable is constgold, so cell occupancy
# can differ between the two sims -- and a pop-weight npz would remove it if it ever matters.
#
# FIREWALL: the anchor matches the HALF-SHEAR target's weighted mean. No constgold response and no m
# enters training. Do not tune ANCHOR against constgold m.
echo "### V2.1 FLOW lt=$LT seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
echo "anchor: ${ANCHOR:-0}   response-weight: ${RW:-450}"
echo "resp target: $RESP"
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $CAT --output $OUT \
  --target-column detected --selection-name sextractor_detected --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden ${MH:-128} --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 4000000 --epochs ${EPOCHS:-120} --batch-size ${BS:-8192} \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience ${PATIENCE:-20} \
  --weight-decay 1e-5 --swa-last-k ${SWAK:-32} \
  --seed "$SEED" --num-workers 8 --gpu-resident \
  --v21-domain \
  --response-weight ${RW:-450} --response-delta 0.02 --response-difference central \
  --response-target-npz "$RESP" \
  --response-error ${RERR:-absolute} --response-rel-floor ${RFLOOR:-0.05} \
  --response-global-anchor ${ANCHOR:-0} \
  --coupling-weight "$LT" --coupling-target-npz "$COUP" \
  || { echo "FAILED seed=$SEED"; exit 1; }
echo "V21_FLOW_DONE lt=$LT seed=$SEED"; date
