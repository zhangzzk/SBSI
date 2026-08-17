#!/bin/bash
#SBATCH --job-name=dump_szfine
#SBATCH --time=00:50:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/dump_szfine_%j.out

# cont.87 (task #38): per-object dump of the FINER-SIZE-retrained measurement flow (seed 501), so the
# realistic-family eval can read its R_flow column. BYTE-IDENTICAL to job_fig2_dump.sh (same catalogue,
# same lookups => same r_sim validation truth + same emulator R_blend) EXCEPT the model TAG and the
# output DUMP path -> ONLY R_flow differs from the certified dump. Routed to inter (cip full-a40 is
# drained). expandable_segments left UNSET (fine on inter full-a40). constgold read = validation r_sim only.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEED=501
TAG=meas_szfl_noz_lam450_szfine
OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
CAT=$CD/constant_response_catalogue_train.feather
DUMP=/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s${SEED}_szfine.feather

echo "### FIG2_DUMP_SZFINE job=$SLURM_JOB_ID seed=$SEED TAG=$TAG ###"; nvidia-smi -L; date
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
echo "FIG2_DUMP_SZFINE_DONE dump=$DUMP"; date
