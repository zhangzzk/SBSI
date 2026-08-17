#!/bin/bash
#SBATCH --job-name=fig2_dump
#SBATCH --time=00:50:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/fig2_dump_%j.out
# cip full-a40 (cip-cl-nv01) starts immediately; the inter a40 queue is days out.
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments on cip a40-*gb
# vGPU slices (unsupported -> CUDA "operation not supported" at torch.load);
# not needed for this 8M-row dump either, so it is left unset.

# ONE small per-object dump for Figure 2 (response-vs-property). Current fixresp
# convention, seed 501. Writes per-object (case,input_index,r_input_p,r_sim,
# R_flow,R_blend,neighbored,distance) so the plot script can join property
# columns (Re_input_p size, S/N_plus flux, nbr_flux_near blend flux) offline.
# Mirrors job_pilot_harvest.sh lookups so R_blend matches the certified harvest.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEED=501
TAG=meas_szfl_noz_lam450_fixresp
OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
CAT=$CD/constant_response_catalogue_train.feather
DUMP=/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s${SEED}_fixresp.feather

echo "### FIG2_DUMP job=$SLURM_JOB_ID seed=$SEED TAG=$TAG ###"; nvidia-smi -L; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $OUT \
  --catalogue $CAT --min-case 40 \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup results/crowd_flux_conc_c0-199.feather \
  --meas-prim-lookup results/meas_prim_lookup_c0-139.feather \
  --ood-lookup results/ood_split_c40-139.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --global-only --flow-seed 12345 \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --batch-size 16384 --n-boot 50 \
  --dump $DUMP 2>&1 | grep --line-buffered -vE "module command"
echo "FIG2_DUMP_DONE dump=$DUMP"; date
