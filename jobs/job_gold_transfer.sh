#!/bin/bash
#SBATCH --job-name=goldxfer
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/goldxfer_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
COMMON="--measurement-model $MODEL --catalogue $CAT --blend-lookup results/blend_lookup_extnbrho_c40-79.feather --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c40-79.feather --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 12000000 --n-boot 200"
echo "### GOLD-STANDARD held-out 40-79: BASELINE (no correction) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON 2>&1 | grep -vE "module command" | grep -E "matched|GLOBAL|WITH blend m|bare-flow|Rbl q|ISO\("
echo "### GOLD-STANDARD held-out 40-79: WITH correction (fit on 0-39) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON --apply-rblend-corr results/rblend_corr_c0-39.npz 2>&1 | grep -vE "module command" | grep -E "applied R_blend|GLOBAL|WITH blend m|bare-flow|Rbl q|ISO\("
date; echo GOLDXFER_DONE
