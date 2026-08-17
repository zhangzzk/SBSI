#!/bin/bash
#SBATCH --job-name=ab_scenecorr
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_scenecorr_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_scenecorr_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
BASE=/project/ls-gruen/users/zekang.zhang
cd "$ROOT"
OUT=results/anchorblend_scene_correction_v22_c200-299.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
"$PY" -u scripts/fit_anchorblend_scene_correction.py \
  --base "$BASE/lsst_sims_fs2_25876_anchorblend_random_local10_c200-299" \
  --response10 results/anchorblend_random_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_local15_response_v22_c200-299.feather \
  --response10 results/anchorblend_random_layer1_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_layer1_local15_response_v22_c200-299.feather \
  --response10 results/anchorblend_random_layer2_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_layer2_local15_response_v22_c200-299.feather \
  --coherent-response results/anchorblend_g005_response_v22_c100-299.feather \
  --output "$OUT"
echo ANCHORBLEND_SCENE_CORRECTION_JOB_DONE; date
