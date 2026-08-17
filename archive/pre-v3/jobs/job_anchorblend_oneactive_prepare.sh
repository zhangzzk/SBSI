#!/bin/bash
#SBATCH --job-name=ab1n_prep
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_prep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_prep_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
BASE_U=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_u_c400-499
BASE_V=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_v_c400-499
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
test ! -e "$MANIFEST" || { echo "REFUSING existing $MANIFEST"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/prepare_anchorblend_oneactive_orthogonal.py \
  --reference-base "$REFERENCE" --base-u "$BASE_U" --base-v "$BASE_V" \
  --manifest-dir "$MANIFEST" --cases $(seq 400 499) --g 0.05 \
  --tag lsst_r_extnbr_v22
echo ANCHORBLEND_ONEACTIVE_PREP_JOB_DONE
date
