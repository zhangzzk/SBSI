#!/bin/bash
#SBATCH --job-name=abg002x_resp
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

: "${AB_START:?AB_START not set}"
: "${AB_STOP:?AB_STOP not set}"

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${AB_START}-${AB_STOP}
OUT=$ROOT/results/anchorblend_g002_response_v22_c${AB_START}-${AB_STOP}.feather
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
python -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases $(seq "$AB_START" "$AB_STOP") --g 0.02 \
  --tags lsst_r_extnbr_v22 --output "$OUT"
echo "ANCHORBLEND_G002_EXT_RESPONSE_DONE cases=${AB_START}-${AB_STOP}"
date
