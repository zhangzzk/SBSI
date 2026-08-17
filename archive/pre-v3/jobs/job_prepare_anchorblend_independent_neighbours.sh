#!/bin/bash
#SBATCH --job-name=ab_indprep
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indprep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indprep_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SRC=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
OUT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/prepare_anchorblend_independent_neighbours.py \
  --source "$SRC" --output "$OUT" --cases $(seq 200 299) --g 0.05 --radius 10
echo ANCHORBLEND_INDEPENDENT_NEIGHBOUR_PREP_JOB_DONE
date
