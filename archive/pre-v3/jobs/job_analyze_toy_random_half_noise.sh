#!/bin/bash
#SBATCH --job-name=rhalf_an
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/toy_random_half_noise_analyze_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT="$ROOT/results/toy_random_half_response_noise_summary.json"
cd "$ROOT"

test ! -e "$OUT"
"$PY" -u scripts/analyze_toy_random_half_noise.py \
  --inputs "$ROOT/results/toy_random_half_response_noise_s17??.json" \
  --output-json "$OUT"
test -s "$OUT"
