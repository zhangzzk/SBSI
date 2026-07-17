#!/bin/bash
#SBATCH --job-name=qdiag_c100
#SBATCH --time=16:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/qdiag_c100_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# Headline: apply deficit(R_blend) correction (fit on cases 40-59) to the FULL 100-case set (40-139).
# Compare against the known UNCORRECTED baseline job 15013994: global +0.81%, q3 +6.8% (same catalogue/binning).
# Cases 40-59 are in-sample (20%); 60-139 (80%) are out-of-sample. Out-of-sample transfer already shown on 60-79.
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_c40-139.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
NPZ=results/deficit_rblend_c40-59.npz
COMMON="--measurement-model $MODEL --catalogue $CAT \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c40-139.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300"

echo "### 100-CASE (40-139) WITH deficit(R_blend) correction fit on 40-59 ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON \
  --apply-rblend-corr "$NPZ" 2>&1 | grep --line-buffered -vE "module command"
date; echo QDIAG_C100_DONE
