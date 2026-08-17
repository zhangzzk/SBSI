#!/bin/bash
#SBATCH --job-name=v22_domphys
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PAIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_pair_features_c40-139.feather
HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
CONTROL_BINS=${CONTROL_BINS:-4}
AMPLITUDE_CONTROL=${AMPLITUDE_CONTROL:-dominant_pair}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
OUT_JSON=$ROOT/results/v22_dominant_physics_anchor_c400-899_constgold_c40-139${OUTPUT_SUFFIX}.json
OUT_MD=$ROOT/results/v22_dominant_physics_anchor_c400-899_constgold_c40-139${OUTPUT_SUFFIX}.md
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
for input in "$PAIR" "$HALF" results/v22_constgold_gap_features_c40-139.feather; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
cd "$ROOT"

RESPONSES=()
DOMINANCE=()
for block in 400-499 500-599 600-699 700-799 800-899; do
  RESPONSES+=("results/anchorblend_g002_response_v22_c${block}.feather")
  DOMINANCE+=("results/anchorblend_g002_dominance_v22_c${block}.feather")
done
for input in "${RESPONSES[@]}" "${DOMINANCE[@]}"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done

python -u scripts/diagnose_v22_dominant_physics.py \
  --anchor-response "${RESPONSES[@]}" \
  --anchor-dominance "${DOMINANCE[@]}" \
  --constgold-gap results/v22_constgold_gap_features_c40-139.feather \
  --constgold-pairs "$PAIR" --half-selfresp "$HALF" \
  --anchor-development-max 599 --constgold-development-max 89 \
  --n-control-bins "$CONTROL_BINS" --amplitude-control "$AMPLITUDE_CONTROL" \
  --output-json "$OUT_JSON" --output-md "$OUT_MD"
echo V22_DOMINANT_PHYSICS_JOB_DONE
date
