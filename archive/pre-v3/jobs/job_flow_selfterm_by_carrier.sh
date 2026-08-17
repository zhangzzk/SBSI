#!/bin/bash
#SBATCH --job-name=flow_selfterm
#SBATCH --time=01:00:00
#SBATCH --mem=96G
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
OUT=$ROOT/results/flow_selfterm_by_carrier_c40-139_s16.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
python -u scripts/diag_flow_selfterm_by_carrier.py \
  --half-selfresp "$HALF" --constgold-pairs "$PAIR" \
  --seeds 501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517 \
  --case-min 40 --case-max 139 --development-max 89 \
  --output "$OUT"
echo FLOW_SELFTERM_JOB_DONE; date
