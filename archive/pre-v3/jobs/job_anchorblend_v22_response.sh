#!/bin/bash
#SBATCH --job-name=ab_v22
#SBATCH --array=0-1%2
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_v22_%A_%a.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$TASK" -eq 0 ]; then
  BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005
  CASES=$(seq 0 99)
  OUT=results/anchorblend_g005_response_v22_c0-99.feather
else
  BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
  CASES=$(seq 100 299)
  OUT=results/anchorblend_g005_response_v22_c100-299.feather
fi
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases $CASES --g 0.05 \
  --tags lsst_r_extnbr_v21 lsst_r_extnbr_v22 --output "$OUT"

echo ANCHORBLEND_V22_RESPONSE_DONE; date
