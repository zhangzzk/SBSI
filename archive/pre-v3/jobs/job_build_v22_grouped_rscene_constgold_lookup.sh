#!/bin/bash
#SBATCH --job-name=v22grs_cg_lookup
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22grs_cg_lookup_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22grs_cg_lookup_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1
OUT=$RUN/constgold_c40-139/lookup.feather
SUMMARY=$ROOT/results/v22_grouped_rscene_constgold_lookup_c40-139.json
MODEL_SUMMARY=$RUN/run/final.summary.json
REFERENCE=$ROOT/results/blend_lookup_v22_c40-139.feather
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT" "$SUMMARY"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
python -u scripts/build_v22_grouped_rscene_constgold_lookup.py \
  --cases $(seq 40 139) \
  --model-summary "$MODEL_SUMMARY" \
  --reference-lookup "$REFERENCE" \
  --output "$OUT" \
  --summary-json "$SUMMARY"
test -s "$OUT"
test -s "$SUMMARY"
echo V22_GROUPED_RSCENE_CONSTGOLD_LOOKUP_JOB_DONE
date
