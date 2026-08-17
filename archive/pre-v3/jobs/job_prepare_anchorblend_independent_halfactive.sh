#!/bin/bash
#SBATCH --job-name=ab_halfprep
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_halfprep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_halfprep_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SRC=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
OUT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_halfactive_local10_c200-299
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
python -u scripts/prepare_anchorblend_independent_neighbours.py \
  --source "$SRC" --output "$OUT" --cases $(seq 200 299) \
  --g 0.05 --radius 10 --active-mode upper_index_half
echo ANCHORBLEND_INDEPENDENT_HALF_ACTIVE_PREP_DONE
date
