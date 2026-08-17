#!/bin/bash
#SBATCH --job-name=v22_sharedpilot
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_sharedpilot_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_sharedpilot_%j.err
set -euo pipefail

# Early path/feature validation on the already rendered 100-case anchor block.
# This is explicitly a diagnostic pilot; the production result uses the
# predeclared c400--599 / c600--899 split in job_localize_v22_shared_gap.sh.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PAIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_pair_features_c40-139.feather
HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
OUT_JSON=$ROOT/results/v22_shared_gap_localization_pilot_dualsearch_anchor_g002_c400-499_constgold_c40-139.json
OUT_MD=$ROOT/results/v22_shared_gap_localization_pilot_dualsearch_anchor_g002_c400-499_constgold_c40-139.md
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
for input in "$PAIR" "$HALF" results/v22_constgold_gap_features_c40-139.feather \
  results/anchorblend_g002_response_v22_c400-499.feather \
  results/anchorblend_g002_dominance_v22_c400-499.feather; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
cd "$ROOT"
python -u scripts/localize_v22_shared_gap.py \
  --anchor-response results/anchorblend_g002_response_v22_c400-499.feather \
  --anchor-dominance results/anchorblend_g002_dominance_v22_c400-499.feather \
  --constgold-gap results/v22_constgold_gap_features_c40-139.feather \
  --constgold-pairs "$PAIR" --half-selfresp "$HALF" \
  --anchor-development-max 449 --constgold-development-max 89 \
  --output-json "$OUT_JSON" --output-md "$OUT_MD"
echo V22_SHARED_GAP_LOCALIZATION_PILOT_DONE
date
