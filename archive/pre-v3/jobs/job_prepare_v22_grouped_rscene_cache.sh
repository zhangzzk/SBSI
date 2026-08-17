#!/bin/bash
#SBATCH --job-name=v22grpcache
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENES=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_validation_tail_scenes_c40-199.feather
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE/metadata.json"
test -s "$SCENES"
test ! -e "$OUTPUT" || { echo "REFUSING existing $OUTPUT"; exit 1; }
python -u scripts/prepare_v22_grouped_rscene_cache.py \
  --source-cache "$SOURCE" --scene-audit "$SCENES" --output-dir "$OUTPUT"
echo V22_GROUPED_RSCENE_CACHE_JOB_DONE
date
