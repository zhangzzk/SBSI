#!/bin/bash
#SBATCH --job-name=abfr_rpow2
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-59%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_rpow2_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_rpow2_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399
REFERENCE=$ROOT/results/anchorblend_g005_response_v22_c300-399.feather
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_refined_c300-399
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
BLOCK=$((SLURM_ARRAY_TASK_ID % 20))
case "$MODEL_INDEX" in
  0) TAG=lsst_r_extnbr_v22_rpowa010 ;;
  1) TAG=lsst_r_extnbr_v22_rpowa015 ;;
  2) TAG=lsst_r_extnbr_v22_rpowa020 ;;
  *) echo "unexpected model index $MODEL_INDEX"; exit 2 ;;
esac
START=$((300 + 5 * BLOCK))
END=$((START + 4))
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$PARTS/$TAG"
cd "$ROOT"
for case in $(seq "$START" "$END"); do
  env SLURM_PROCID=0 python -u scripts/score_anchor_response_model.py \
    --base "$BASE" --reference "$REFERENCE" --model-tag "$TAG" \
    --output-dir "$PARTS/$TAG" --case-offset "$case" --n-cases 1
done
echo "ANCHOR_RESPONSE_WEIGHTED_REFINED_FRESH_PART_DONE tag=$TAG cases=$START-$END"
