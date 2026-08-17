#!/bin/bash
#SBATCH --job-name=rblfullctx
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SIMULATION=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE/metadata.json"
test -d "$SIMULATION/case0_0.2"
python -u scripts/prepare_v22_fullneighbour_context_cache.py \
  --source-cache "$SOURCE" \
  --simulation-root "$SIMULATION" \
  --output-dir "$OUTPUT"
echo "PREPARE_V22_FULLNEIGHBOUR_CONTEXT_JOB_DONE"
date
