#!/bin/bash
#SBATCH --job-name=v22q3rhoana
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3rhoana_%j.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3rhoana_%j.%N.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
cd "$ROOT"
"$PY" -u scripts/analyze_v22_q3_purity.py \
  --truth "$CACHE/v22_constgold_q3_decomp_c40-139_truth.feather" \
  --flow "$CACHE/v22_constgold_q3_decomp_c40-139_flow.feather" \
  --purity-glob "$CACHE/v22_constgold_q3_purity_adaptive_c*-*.feather" \
  --output-prefix results/v22_constgold_q3_purity_informativeness_final_c40-139 \
  --prediction-output "$CACHE/v22_constgold_q3_purity_cv_predictions_final_c40-139.feather"
echo V22_Q3_PURITY_ANALYZE_JOB_DONE
