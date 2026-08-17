#!/bin/bash
#SBATCH --job-name=abdev_rpow
#SBATCH --time=00:50:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-39%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/abdev_rpow_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abdev_rpow_%A_%a.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TAG=lsst_r_extnbr_v22_rpowa0065
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_c0-199
START=$((5 * SLURM_ARRAY_TASK_ID))
END=$((START + 4))
if [ "$START" -lt 100 ]; then
  BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005
  REFERENCE=$ROOT/results/anchorblend_g005_response_v22_c0-99.feather
else
  BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
  REFERENCE=$ROOT/results/anchorblend_g005_response_v22_c100-299.feather
fi
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
echo "ANCHOR_RESPONSE_WEIGHTED_DEVELOPMENT_PART_DONE cases=$START-$END"
