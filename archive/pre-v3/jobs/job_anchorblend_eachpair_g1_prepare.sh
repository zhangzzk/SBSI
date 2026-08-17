#!/bin/bash
#SBATCH --job-name=abep_prep
#SBATCH --time=03:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abep_prep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abep_prep_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_u_c400-499
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g1
SOURCE_MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_eachpair_g1_pilot_c400-409
test ! -e "$MANIFEST" || { echo "REFUSING existing $MANIFEST"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/prepare_anchorblend_eachpair_g1.py \
  --reference-base "$REFERENCE" --output-prefix "$PREFIX" \
  --source-manifest "$SOURCE_MANIFEST" --manifest-dir "$MANIFEST" \
  --cases $(seq 400 409) --g 0.05 --max-rank 18
echo ANCHORBLEND_EACHPAIR_G1_PREP_JOB_DONE
date
