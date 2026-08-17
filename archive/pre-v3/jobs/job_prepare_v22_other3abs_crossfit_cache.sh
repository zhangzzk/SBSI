#!/bin/bash
#SBATCH --job-name=rblb3prep
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
AUGMENTED=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_other3abs/response_catalogue_train.feather
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_other3abs_crossfit_features_v1/cache

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE/metadata.json"
test -s "$AUGMENTED"
python -u scripts/prepare_v22_other3abs_crossfit_cache.py \
  --source-cache "$SOURCE" \
  --augmented-catalogue "$AUGMENTED" \
  --output-dir "$OUTPUT"
echo V22_OTHER3ABS_CROSSFIT_CACHE_JOB_DONE
date
