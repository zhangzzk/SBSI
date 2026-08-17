#!/bin/bash
#SBATCH --job-name=posw_tv_v21
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/posw_tv_v21_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/posw_tv_v21_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
OUTPUT=$ROOT/results/v22_pair_residual_vs_prediction_weight_scan_official_validation_v21domain_c40-199_overlay
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "$ROOT"
"$PY" -u scripts/plot_positive_weighted_pair_residual_official_validation_overlay.py \
  --cache "$CACHE" \
  --case-min 40 --case-max 199 --n-bins 20 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --v21-primary-domain \
  --output-prefix "$OUTPUT"
