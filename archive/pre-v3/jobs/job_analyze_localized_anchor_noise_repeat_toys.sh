#!/bin/bash
#SBATCH --job-name=abnr_ana
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
MANIFEST=$ROOT/results/localized_anchor_noise_repeat_toy_manifest_v22_typical20.feather
INPUT=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noise_repeat_toy_v22_typical20_r400}
OUT=$ROOT/results/localized_anchor_noise_repeat_toy_v22_typical20_r400
NREAL=${TOY_NREAL:-400}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$OUT.feather" "${OUT}_pairs.feather" "$OUT.json" "$OUT.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/analyze_localized_anchor_noise_repeat_toys.py \
  --manifest "$MANIFEST" --input-dir "$INPUT" \
  --noiseless-results results/localized_anchor_noiseless_toy_v22_c700-899.feather \
  --nreal "$NREAL" --g 0.02 --stamp 48 --pixel-rms 0.312 \
  --sesoi 0.02 --target-ci-halfwidth 0.01 --min-success-fraction 0.8 \
  --output-scenes "$OUT.feather" --output-pairs "${OUT}_pairs.feather" \
  --output-json "$OUT.json" --output-md "$OUT.md"
test -s "$OUT.json"
test -s "$OUT.md"
echo LOCALIZED_ANCHOR_NOISE_REPEAT_ANALYSIS_JOB_DONE
date
