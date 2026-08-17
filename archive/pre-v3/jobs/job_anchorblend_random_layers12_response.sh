#!/bin/bash
#SBATCH --job-name=ab_rlresp
#SBATCH --array=0-3
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlresp_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_rlresp_%A_%a.err
set -euo pipefail
TASK=${SLURM_ARRAY_TASK_ID:?array task required}; LAYER=$((TASK / 2 + 1))
if (( TASK % 2 == 0 )); then RADIUS=10; else RADIUS=15; fi
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_layer${LAYER}_local${RADIUS}_c200-299
OUT=$ROOT/results/anchorblend_random_layer${LAYER}_local${RADIUS}_response_v22_c200-299.feather
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}; cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases $(seq 200 299) --g 0.05 --anchor-directions \
  --tags lsst_r_extnbr_v21 lsst_r_extnbr_v22 --output "$OUT"
echo "ANCHORBLEND_RANDOM_LAYER${LAYER}_LOCAL${RADIUS}_RESPONSE_DONE"; date
