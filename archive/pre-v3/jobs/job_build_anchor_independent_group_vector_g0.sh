#!/bin/bash
#SBATCH --job-name=ab_indgroupg0
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indgroupg0_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indgroupg0_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299
TAG=lsst_r_extnbr_v22_groupvecg0
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_${TAG}_c200-299.feather
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"
"$PY" -u scripts/build_anchorblend_independent_response.py \
  --base "$BASE" --cases $(seq 200 299) --g 0.05 --tag "$TAG" --output "$OUT"
