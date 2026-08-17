#!/bin/bash
#SBATCH --job-name=anchor_nb4x4
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
SCORES=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_2x2_ablation_v1/anchor_scores_c400_899
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OLD_JSON=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final.json
OLD_CURVES=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final_curves.csv
OUTPUT=$ROOT/results/anchor_newbase_raw_corrected_residual_1d_all_c700-899
PANEL_E=$ROOT/results/anchor_newbase_raw_corrected_residual_vs_newbase_scene_c700-899

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

python -u scripts/plot_anchor_newbase_original_cases_4x4.py \
  --anchor-features "$FEATURES" \
  --score-dir "$SCORES" \
  --original-json "$OLD_JSON" \
  --original-curves "$OLD_CURVES" \
  --output-stem "$OUTPUT" \
  --panel-e-output-stem "$PANEL_E"
echo "PLOT_ANCHOR_NEWBASE_ORIGINAL_CASES_4X4_JOB_DONE"
date
