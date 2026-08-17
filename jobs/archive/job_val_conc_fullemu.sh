#!/bin/bash
#SBATCH --job-name=val_fullemu
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_fullemu_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# ATTRIBUTION TEST (flow vs emulator on held-out 0-39): re-validate conc-v1 on cases 0-39 using the
# FULL emulator (blend_lookup_c0-199, trained on 0-199 -> in-distribution on 0-39) instead of the ho
# emulator (blend_lookup_extnbrho, trained 40-199 -> OOD on 0-39). The flow trains on 0-39 either way.
# If q3 drops from the ho-emulator's +8.0% back toward the in-training +4.1%, the held-out inflation was
# EMULATOR-OOD, not a flow limitation -> confirms the flow is the lever and the emulator pivot was wrong.
MODEL=models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant

echo "### conc-v1 on 0-39 with FULL emulator (in-dist), NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_train.rmax10.feather \
  --blend-lookup results/blend_lookup_c0-199.feather \
  --crowd-flux-lookup results/crowd_flux_conc_c0-199.feather \
  --n-samples 48 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 200 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo VAL_FULLEMU_DONE
