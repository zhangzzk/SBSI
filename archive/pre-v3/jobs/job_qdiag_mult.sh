#!/bin/bash
#SBATCH --job-name=qdiag_mult
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/qdiag_mult_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

MULT=results/blend_multiplicity_extnbrho_c40-79.feather
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_c40-79.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt

echo "### STEP 1: build multiplicity lookup c40-79 ###"; date
if [ ! -f "$MULT" ]; then
  stdbuf -oL -eL python -u scripts/build_blend_multiplicity.py \
    --cases $(seq 40 79) --output "$MULT" --tag lsst_r_extnbr_ho 2>&1 | grep -vE "module command"
else
  echo "multiplicity lookup already exists: $MULT"
fi
date

echo "### STEP 2: validation c40-79 WITH multiplicity discriminator ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" --catalogue "$CAT" \
  --blend-lookup results/blend_lookup_extnbrho_c40-79.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c40-79.feather \
  --mult-lookup "$MULT" \
  --n-samples 48 --n-blend 4 --blend-eps 0.02 --max-rows 20000000 --n-boot 200 2>&1 \
  | grep -vE "module command"
date; echo QDIAG_MULT_DONE
