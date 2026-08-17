#!/bin/bash
#SBATCH --job-name=abv2c_resp
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-4%2
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2c_resp_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2c_resp_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
START=$((400 + 100 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 99))
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c${START}-${STOP}
OUT=$ROOT/results/anchorblend_g002_response_v2complement_c${START}-${STOP}.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
python -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases $(seq "$START" "$STOP") --g 0.02 \
  --primary-domain v2-complement --tags lsst_r_extnbr_indom_tuned \
  --output "$OUT"
test -s "$OUT"
echo "ANCHOR_V2_COMPLEMENT_RESPONSE_DONE cases=$START-$STOP"
