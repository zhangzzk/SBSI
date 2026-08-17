#!/bin/bash
#SBATCH --job-name=s2cdomev
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2cdomev_%j.out

# Certify the domain-trained checkpoint on constgold and report m under the SAME four masks used for
# the 8-seed baseline (scripts/eval_v2_indomain_m.py), so the two are directly comparable.
#
# The EVALUATION population is deliberately left unchanged (DEFAULT_SELECTION_CUTS, min-case 40) and
# the domain is applied as a MASK afterwards. That keeps the constgold rows row-for-row identical to
# the baseline dumps, so the only thing that differs between the two m tables is the model.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(${SEEDS:-501 502 503 505 506 507 508 509})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
  SEED=${SEED:-501}
fi
TAG=${TAG:-ablate_s2c_coupling_lt500_dom}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
DUMPDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
mkdir -p $DUMPDIR
CK=$D/measurement_flow_g0_ngmix_${TAG}_s${SEED}_swaavg.pt
# DUMPNAME lets a re-score of an EXISTING tag write somewhere else. Without it a control run (e.g.
# re-scoring the fiducial checkpoint on different GPU hardware) would silently OVERWRITE the real
# fiducial dump, destroying the baseline it is meant to be compared against. Defaults to TAG, so
# every existing invocation is byte-identical.
DUMPNAME=${DUMPNAME:-$TAG}

echo "### S2C-DOMAIN EVAL seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  --dump "$DUMPDIR/${DUMPNAME}_perobj_s${SEED}.feather" || { echo "FAILED eval seed=$SEED"; exit 1; }

echo; echo "### m under the four masks (same script as the 8-seed baseline) ###"
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$DUMPDIR/${DUMPNAME}_perobj_s${SEED}.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.3 --mag-max 26.0
echo "S2CDOMEV_DONE seed=$SEED"; date
