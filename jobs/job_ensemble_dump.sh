#!/bin/bash
#SBATCH --job-name=ens_dump
#SBATCH --time=00:50:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=cip
#SBATCH --array=502-516
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens_dump_%A_%a.out
# cont.100: per-object R_flow dumps for the OTHER 15 certified seeds (s502-516), on the
# SAME constgold matched sample as the existing s501 dump. Averaging all 16 offline gives
# the certified 16-seed ENSEMBLE R_flow per object -> the honest PER-CUT flow closure
# (the s501 single-seed dump inflates it ~1sigma high; +0.91% vs certified ensemble +0.245%).
# Byte-for-byte mirror of job_fig2_dump.sh (same lookups/flags/full-a40) -> comparable R_flow.
# Full a40 on cip-cl-nv01: do NOT set PYTORCH_CUDA_ALLOC_CONF (fine here, but matched to fig2 job).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

SEED=${SLURM_ARRAY_TASK_ID}
TAG=meas_szfl_noz_lam450_fixresp
OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
CAT=$CD/constant_response_catalogue_train.feather
DUMP=/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s${SEED}_fixresp.feather

if [ ! -f "$OUT" ]; then echo "MISSING MODEL seed=$SEED"; exit 1; fi
echo "### ENS_DUMP job=$SLURM_JOB_ID seed=$SEED TAG=$TAG ###"; nvidia-smi -L; date
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
echo "ENS_DUMP_DONE seed=$SEED dump=$DUMP"; date
