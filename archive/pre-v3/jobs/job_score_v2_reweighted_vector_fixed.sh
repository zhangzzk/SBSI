#!/bin/bash
#SBATCH --job-name=score_v2_rwv
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/score_v2_rwv_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/score_v2_rwv_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SUMMARY=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/summary.json
OUTPUT=results/v2_reweighted_vector_fixed_halfshear_score_c0-39.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

test -s "$SUMMARY"
[ ! -e "$OUTPUT" ] || { echo "REFUSING to overwrite $OUTPUT"; exit 1; }
nvidia-smi -L
"$PY" -u scripts/score_v2_reweighted_vector_fixed.py \
  --config configs/fs2_lsst_r_extnbr_indom_tuned.yaml \
  --training-summary "$SUMMARY" \
  --output "$OUTPUT"
test -s "$OUTPUT"
echo V2_REWEIGHTED_VECTOR_FIXED_SCORE_JOB_DONE
