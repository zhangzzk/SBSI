#!/bin/bash
#SBATCH --job-name=ab_rlplan
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlplan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_rlplan_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"
"$PY" -u scripts/plan_anchorblend_random_layers.py \
  --source /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --case-min 200 --case-max 299 --min-separation 30.01 \
  --output results/anchorblend_random_layer_plan_c200-299.json
echo ANCHORBLEND_RANDOM_LAYER_PLAN_JOB_DONE; date
