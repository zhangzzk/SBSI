#!/bin/bash
#SBATCH --job-name=ab_o3score
#SBATCH --array=0-3%4
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_o3score_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_o3score_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in
  0)
    BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
    RESPONSE=results/anchorblend_g005_response_v22_c100-299.feather
    OUT=results/anchorblend_g005_other3abs_c200-299.feather
    ;;
  1)
    BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_local10_c200-299
    RESPONSE=results/anchorblend_random_local10_response_v22_c200-299.feather
    OUT=results/anchorblend_random_local10_other3abs_c200-299.feather
    ;;
  2)
    BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_layer1_local10_c200-299
    RESPONSE=results/anchorblend_random_layer1_local10_response_v22_c200-299.feather
    OUT=results/anchorblend_random_layer1_local10_other3abs_c200-299.feather
    ;;
  3)
    BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_layer2_local10_c200-299
    RESPONSE=results/anchorblend_random_layer2_local10_response_v22_c200-299.feather
    OUT=results/anchorblend_random_layer2_local10_other3abs_c200-299.feather
    ;;
  *) exit 1 ;;
esac
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/rescore_anchorblend_other3abs.py \
  --response "$RESPONSE" --base "$BASE" \
  --scene-lookup results/neighbor_flux_shells_anchor_c200-299.feather \
  --case-min 200 --case-max 299 --g 0.05 --output "$OUT"
echo ANCHORBLEND_OTHER3ABS_RESCORE_JOB_DONE
date
