#!/bin/bash
#SBATCH --job-name=ab_dist_an
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_dist_an_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"

OUT=results/anchorblend_g005_distance_models_c0-299.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }

"$PY" -u scripts/analyze_anchorblend_distance_models.py \
  results/anchorblend_g005_distance_models_c0-99.feather \
  results/anchorblend_g005_distance_models_c100-299.feather \
  --overlap-audit results/anchorblend_training_overlap_audit_v2.json \
  --output "$OUT"

echo ANCHORBLEND_DISTANCE_ANALYSIS_JOB_DONE; date
