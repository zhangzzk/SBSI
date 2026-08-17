#!/bin/bash
#SBATCH --job-name=t002
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/t002_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date
echo "### build 0.02 R_blend lookup ###"
python -u scripts/build_rblend002_lookup.py 2>&1 | grep -vE "module command"
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
COMMON="--measurement-model $MODEL --catalogue $CAT --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c40-79.feather --min-case 40 --max-case 60 --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 12000000 --n-boot 100"
echo "### BASELINE (emulator R_blend, cases 40-59) ###"
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON --blend-lookup results/blend_lookup_extnbrho_c40-79.feather 2>&1 | grep -vE "module command" | grep -E "GLOBAL|WITH blend m|bare-flow|Rbl q|ISO\("
echo "### TEST (g=0.02-scaled R_blend, cases 40-59) ###"
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON --blend-lookup results/blend_lookup_002_c40-59.feather 2>&1 | grep -vE "module command" | grep -E "GLOBAL|WITH blend m|bare-flow|Rbl q|ISO\("
date; echo T002_JOBDONE
