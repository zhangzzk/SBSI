#!/bin/bash
#SBATCH --job-name=v22_bootscore
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=24
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_bootscore_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_bootscore_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
MODEL_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_gap_v22_training_bootstrap
TABLE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_v22_training_bootstrap/anchor_c200-299_scores.feather
OUT=$ROOT/results/anchorblend_v22_training_bootstrap_c200-299.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
cd "$ROOT"
[ ! -e "$TABLE" ] || { echo "REFUSING existing $TABLE"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/score_anchorblend_v22_bootstrap.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --model-dir "$MODEL_DIR" --n-bootstrap 16 \
  --table-output "$TABLE" --output "$OUT"
echo V22_CASE_BOOTSTRAP_SCORE_JOB_DONE; date
