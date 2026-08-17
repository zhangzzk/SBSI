#!/bin/bash
#SBATCH --job-name=abtoy_prep
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
OUT_FEATHER=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.feather
OUT_JSON=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.json
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$OUT_FEATHER" "$OUT_JSON"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/prepare_localized_anchor_noiseless_toys.py \
  --features results/anchorblend_g002_bias_features_v22_c400-899.feather \
  --parent-curves results/anchorblend_g002_bias_emulator_v22_c400-899_final_curves.csv \
  --case-min 700 --case-max 899 --compact-size 0.4 --expected-rows 56691 \
  --output-feather "$OUT_FEATHER" --output-json "$OUT_JSON"
test -s "$OUT_FEATHER"
test -s "$OUT_JSON"
echo LOCALIZED_ANCHOR_NOISELESS_TOY_PREP_JOB_DONE
date
