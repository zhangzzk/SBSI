#!/bin/bash
#SBATCH --job-name=ab3g_prep
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
PREFIX=$ROOT/results/anchor_highp_threeleg_gaussian_manifest_v22_c700-899
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$PREFIX.feather" "$PREFIX.sources.feather" "$PREFIX.json"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/prepare_anchor_highp_threeleg_gaussian_toys.py \
  --features results/anchorblend_g002_bias_features_v22_c400-899.feather \
  --pair-design \
    results/anchorblend_g002_pairs_renderer_v22_c700-799.json \
    results/anchorblend_g002_pairs_renderer_v22_c800-899.json \
  --case-min 700 --case-max 899 \
  --prediction-min 0.2 --primary-mag-min 24.5 --primary-mag-max 25.8 \
  --close-radius 4.0 --multiplicities 2 4 8 --n-per-multiplicity 16 \
  --seed 20260814 \
  --output-templates "$PREFIX.feather" \
  --output-sources "$PREFIX.sources.feather" \
  --output-json "$PREFIX.json"
test -s "$PREFIX.feather"
test -s "$PREFIX.sources.feather"
test -s "$PREFIX.json"
echo ANCHOR_HIGHP_THREELEG_GAUSSIAN_PREP_JOB_DONE
date
