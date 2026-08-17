#!/bin/bash
#SBATCH --job-name=ab_fixedpos
#SBATCH --time=08:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_fixedpos_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_fixedpos_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_fixedpos_c200-299
export PATH="$SIMS/bin:$PATH"
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
cd "$ROOT"
srun --ntasks=100 "$SIMS/bin/python" -u scripts/measure_anchor_fixed_positions.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --output-dir "$OUT" --case-offset 200 --n-cases 100

count=$(find "$OUT" -maxdepth 1 -name 'case*.feather' -type f | wc -l)
[ "$count" -eq 100 ] || { echo "expected 100 case files, found $count"; exit 1; }
echo ANCHOR_FIXED_POSITION_MEASUREMENT_DONE; date
