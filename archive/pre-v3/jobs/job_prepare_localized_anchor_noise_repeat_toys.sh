#!/bin/bash
#SBATCH --job-name=abnr_prep
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
OUT=$ROOT/results/localized_anchor_noise_repeat_toy_manifest_v22_typical20
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$OUT.feather" "$OUT.json"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/prepare_localized_anchor_noise_repeat_toys.py \
  --tail-manifest results/localized_anchor_noiseless_toy_manifest_c700-899.feather \
  --features results/anchorblend_g002_bias_features_v22_c400-899.feather \
  --n-per-stratum 10 --central-quantile 0.02 --seed 20260813 \
  --output-feather "$OUT.feather" --output-json "$OUT.json"
test -s "$OUT.feather"
test -s "$OUT.json"
echo LOCALIZED_ANCHOR_NOISE_REPEAT_PREP_JOB_DONE
date
