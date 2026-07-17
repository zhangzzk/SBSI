#!/bin/bash
#SBATCH --job-name=val_meas
#SBATCH --time=18:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_meas_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# Validate a REALISTIC (measured-primary) flow on constgold (WORKLOG cont.19). Parametrized by MTAG
# (meas_szfl = V1 bridge, meas_full = V2 realistic), passed via sbatch --export=ALL,MTAG=...
# IDENTICAL to job_validate_conc.sh EXCEPT MODEL and the added --meas-prim-lookup (measured primary
# observables averaged over +/- signs). Direct A/B vs conc-v1: 100-case(40-139) +0.11% ; held-out(0-39)
# -0.07% ; per-bin spread ~6.7/10.7%. Success = |m|<=0.3% on BOTH splits AND per-bin spread not >1.5x conc-v1.
MTAG=${MTAG:-meas_full}
MODEL=models/measurement_flow_g0_ngmix_${MTAG}_snc100_central02_lam300_v1.pt
FL=results/crowd_flux_conc_c0-199.feather           # TRUE neighbour flux (theta_blending stays true)
MP=results/meas_prim_lookup_c0-139.feather          # MEASURED primary observables (this experiment)
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
echo "### VALIDATE realistic flow MTAG=$MTAG  MODEL=$MODEL ###"; date

echo "### STEP 1: 100-case (40-139), NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_c40-139.feather \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup $FL \
  --meas-prim-lookup $MP \
  --ood-lookup results/ood_split_c40-139.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date

echo "### STEP 2: emulator-HELD-OUT (0-39), NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_train.rmax10.feather \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup $FL \
  --meas-prim-lookup $MP \
  --ood-lookup results/ood_split_c0-39.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo "VAL_MEAS_DONE MTAG=$MTAG"
