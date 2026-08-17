#!/bin/bash
#SBATCH --job-name=qdiag_ho
#SBATCH --time=13:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/qdiag_ho_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# PORTABILITY / GENERALIZATION TEST.
# Correction deficit(R_blend) was FIT on cases 40-59, which are IN the emulator's (extnbrho) training set.
# Cases 0-39 are the emulator's TRUE HELD-OUT set (extnbrho excludes 0-39). Apply the 40-59-fit correction
# to 0-39: if q3 flattens & global stays sub-percent, the correction generalizes to data the EMULATOR itself
# never saw -> strongest within-sim portability statement.
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.rmax10.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
NPZ=results/deficit_rblend_c40-59.npz
COMMON="--measurement-model $MODEL --catalogue $CAT \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather \
  --n-samples 48 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 200"

echo "### STEP 1: emulator-HELD-OUT cases 0-39 WITH deficit(R_blend) correction (fit on 40-59) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON \
  --apply-rblend-corr "$NPZ" 2>&1 | grep --line-buffered -vE "module command"
date

echo "### STEP 2: CONTROL - held-out 0-39 with NO correction (baseline) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo QDIAG_HO_DONE
