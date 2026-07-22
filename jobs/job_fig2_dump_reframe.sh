#!/bin/bash
#SBATCH --job-name=dump_reframe
#SBATCH --time=00:50:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/dump_reframe_%j.out

# cont.101: per-object R_flow dump of a REFRAMED-loss measurement flow. BYTE-IDENTICAL to
# job_fig2_dump_szfine.sh (same constgold catalogue + lookups => same r_sim validation truth + same
# emulator R_blend) EXCEPT the model TAG and DUMP path -> ONLY R_flow differs. Env: TAG, SEED, DUMPTAG.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

SEED=${SEED:-501}
TAG=${TAG:?set TAG}
DUMPTAG=${DUMPTAG:?set DUMPTAG (short label for the dump file)}
OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
CAT=$CD/constant_response_catalogue_train.feather
DUMP=/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s${SEED}_${DUMPTAG}.feather

if [ ! -f "$OUT" ]; then echo "MISSING MODEL $OUT"; exit 1; fi
echo "### FIG2_DUMP_REFRAME job=$SLURM_JOB_ID seed=$SEED TAG=$TAG DUMPTAG=$DUMPTAG ###"; nvidia-smi -L; date
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
echo "FIG2_DUMP_REFRAME_DONE dump=$DUMP"; date
