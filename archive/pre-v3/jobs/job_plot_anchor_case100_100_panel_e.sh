#!/bin/bash
#SBATCH --job-name=panelE100
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_case100_100_v1
SCORES=$RUN/anchor_scores_c400_899
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OLD_JSON=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final.json
OUTPUT=$ROOT/results/anchor_case100_100_residual_vs_raw_scene_c700-899

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$RUN/summary.json"
test -s "$FEATURES"
test -s "$OLD_JSON"
python -u scripts/plot_anchor_case100_100_panel_e.py \
  --anchor-features "$FEATURES" \
  --score-dir "$SCORES" \
  --original-json "$OLD_JSON" \
  --output-stem "$OUTPUT"
echo "PLOT_ANCHOR_CASE100_100_PANEL_E_JOB_DONE"
date
