#!/bin/bash
# Build archived V2-complement response references after shape production.
#SBATCH --job-name=lv2as_resp
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-4%2
#SBATCH --output=/home/z/Zekang.Zhang/logs/lv2as_resp_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lv2as_resp_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
LEGACY=$ROOT/archive/pre-v3
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
START=$((400 + 100 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 99))
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c${START}-${STOP}
OUT=$LEGACY/results/anchorblend_g002_response_v2complement_c${START}-${STOP}.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LEGACY:$LEGACY/scripts:$BE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$LEGACY"
test ! -e "$OUT"
python -u scripts/build_anchorblend_response.py --base "$BASE" --cases $(seq "$START" "$STOP") \
  --g 0.02 --primary-domain v2-complement --tags lsst_r_extnbr_indom_tuned --output "$OUT"
test -s "$OUT"
echo "LEGACY_V2_ANCHOR_RESPONSE_DONE cases=$START-$STOP"
