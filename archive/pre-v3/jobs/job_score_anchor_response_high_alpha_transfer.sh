#!/bin/bash
#SBATCH --job-name=ab_higha_xfer
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-99%32
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_higha_xfer_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_higha_xfer_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_weak_weight_transfer_g002_c400-899
TAG=${AB_MODEL_TAG:?submit with AB_MODEL_TAG}
case "$TAG" in
  lsst_r_extnbr_v22_rpowposa010|lsst_r_extnbr_v22_rpowposw0200) ;;
  *) echo "unsupported AB_MODEL_TAG=$TAG"; exit 2 ;;
esac
START=$((400 + 5 * SLURM_ARRAY_TASK_ID))
END=$((START + 4))
HUNDRED_START=$((400 + 100 * (SLURM_ARRAY_TASK_ID / 20)))
HUNDRED_END=$((HUNDRED_START + 99))
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${HUNDRED_START}-${HUNDRED_END}
REFERENCE=$ROOT/results/anchorblend_g002_response_v22_c${HUNDRED_START}-${HUNDRED_END}.feather

export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$PARTS/$TAG"
cd "$ROOT"
for anchor_case in $(seq "$START" "$END"); do
  env SLURM_PROCID=0 python -u scripts/score_anchor_response_model.py \
    --base "$BASE" \
    --reference "$REFERENCE" \
    --model-tag "$TAG" \
    --output-dir "$PARTS/$TAG" \
    --case-offset "$anchor_case" \
    --n-cases 1 \
    --sign 0.02
done
echo "ANCHOR_RESPONSE_HIGH_ALPHA_TRANSFER_PART_DONE tag=$TAG cases=$START-$END"
