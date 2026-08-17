#!/bin/bash
#SBATCH --job-name=seqpair_cg50_lookup
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
MODEL_SUMMARY=$RUN/pair/summary.json
OUTDIR=$RUN/pair/constgold_c40-89
LOOKUP=$OUTDIR/lookup.feather
SUMMARY=$ROOT/results/v22_crossfit_pair_constgold_lookup_c40-89.json
REFERENCE=$ROOT/results/blend_lookup_v22_c40-139.feather
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$LOOKUP" "$SUMMARY"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
python -u scripts/build_crossfit_sequential_constgold_lookup.py \
  --cases $(seq 40 89) \
  --model-summary "$MODEL_SUMMARY" \
  --reference-lookup "$REFERENCE" \
  --output "$LOOKUP" --summary-json "$SUMMARY"
test -s "$LOOKUP"
test -s "$SUMMARY"
echo CROSSFIT_PAIR_CONSTGOLD_LOOKUP_C40_89_JOB_DONE
date
