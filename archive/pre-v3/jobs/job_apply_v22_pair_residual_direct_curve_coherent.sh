#!/bin/bash
#SBATCH --job-name=v22direct
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
CAL=$ROOT/results/v22_halfshear_rblend_emulator_label_calibration_training_c40-199
OUT=$ROOT/results/v22_pair_residual_direct_curve_transfer_anchor_g002_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

REFERENCES=()
DESIGNS=()
for block in 400-499 500-599 600-699 700-799 800-899; do
  REFERENCES+=("results/anchorblend_g002_response_v22_c${block}.feather")
  DESIGNS+=("results/anchorblend_g002_pairs_renderer_v22_c${block}.json")
done
for input in "$CAL.csv" "$CAL.json" "${REFERENCES[@]}" "${DESIGNS[@]}"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in json md png pdf; do
  test ! -e "$OUT.$suffix" || { echo "REFUSING existing $OUT.$suffix"; exit 1; }
done
for suffix in cases calibration_bins anchor_deciles; do
  test ! -e "${OUT}_${suffix}.csv" || {
    echo "REFUSING existing ${OUT}_${suffix}.csv"; exit 1; }
done

python -u scripts/apply_v22_pair_residual_calibration_coherent.py \
  --calibration-csv "$CAL.csv" --calibration-json "$CAL.json" \
  --reference "${REFERENCES[@]}" --pair-design "${DESIGNS[@]}" \
  --case-min 400 --case-max 899 --replay-tolerance 3e-7 \
  --output-prefix "$OUT"
echo V22_PAIR_RESIDUAL_DIRECT_CURVE_COHERENT_JOB_DONE
date
