#!/bin/bash
#SBATCH --job-name=poolScenePlot
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_pair_pooled_scene_moment_case100_100_v1
OUTPUT=$ROOT/results/case100_100_pair_pooled_scene_selection_c160-199
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/plot_case100_100_pair_pooled_scene_selection.py \
  --run-dir "$RUN" --output-stem "$OUTPUT"
echo "PLOT_PAIR_POOLED_SCENE_SELECTION_JOB_DONE"
