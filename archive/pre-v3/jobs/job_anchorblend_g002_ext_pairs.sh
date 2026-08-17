#!/bin/bash
#SBATCH --job-name=abg002x_pairs
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

# Deployed per-pair V2.2 responses for one g=0.02 anchor block, written beside
# the anchors so the dominance builder can consume the tree directly.
: "${AB_START:?AB_START not set}"
: "${AB_STOP:?AB_STOP not set}"

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${AB_START}-${AB_STOP}
SUMMARY=$ROOT/results/anchorblend_g002_pairs_v22_c${AB_START}-${AB_STOP}.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -d "$BASE" || { echo "MISSING anchor tree $BASE"; exit 1; }
test ! -e "$SUMMARY" || { echo "REFUSING existing $SUMMARY"; exit 1; }
cd "$ROOT"
python -u scripts/build_anchorblend_pair_responses.py \
  --base "$BASE" --manifest-dir "$BASE" \
  --cases $(seq "$AB_START" "$AB_STOP") --g 0.02 \
  --tag lsst_r_extnbr_v22 --summary-json "$SUMMARY"
echo "ANCHORBLEND_G002_EXT_PAIRS_DONE cases=${AB_START}-${AB_STOP}"
date
