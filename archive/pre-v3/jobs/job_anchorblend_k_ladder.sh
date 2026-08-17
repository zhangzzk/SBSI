#!/bin/bash
#SBATCH --job-name=ab_klad
#SBATCH --array=0-1%2
#SBATCH --time=04:00:00
#SBATCH --mem=44G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_klad_%A_%a.out
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
  RESPONSE=results/anchorblend_g005_response_v22_c0-99.feather
  OUT=results/anchorblend_g005_k_ladder_c0-99.feather
else
  BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
  RESPONSE=results/anchorblend_g005_response_v22_c100-299.feather
  OUT=results/anchorblend_g005_k_ladder_c100-299.feather
fi

"$PY" -u scripts/rescore_anchorblend_k_ladder.py \
  --response "$RESPONSE" --base "$BASE" --g 0.05 --ks 20 32 48 64 --output "$OUT"

echo ANCHORBLEND_K_LADDER_JOB_DONE; date
