#!/bin/bash
#SBATCH --job-name=abg002x_dom
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

# Per-anchor dominance table for one g=0.02 block.  The dominance threshold is
# the value frozen for the published cases 400--499 table; it only sets the
# `is_response_dominant_tail` flag and does not enter the
# `dominant_to_runner_up_abs_response` ratio the amplitude analysis splits on.
: "${AB_START:?AB_START not set}"
: "${AB_STOP:?AB_STOP not set}"

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${AB_START}-${AB_STOP}
STEM=$ROOT/results/anchorblend_g002_dominance_v22_c${AB_START}-${AB_STOP}
DEV_MAX=$(( AB_START + (AB_STOP - AB_START) / 2 ))
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$STEM.feather" "$STEM.json" "$STEM.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
test -s "$BASE/pairs_case${AB_START}.feather" || {
  echo "MISSING pair manifest in $BASE"; exit 1; }
cd "$ROOT"
python -u scripts/localize_anchorblend_response_dominance.py \
  --manifest-dir "$BASE" --reference-base "$BASE" --reference-shear 0.02 \
  --tag lsst_r_extnbr_v22 \
  --case-min "$AB_START" --case-max "$AB_STOP" --development-max "$DEV_MAX" \
  --dominance-threshold 0.7752772106835227 \
  --output-feather "$STEM.feather" --output-json "$STEM.json" \
  --output-md "$STEM.md"
echo "ANCHORBLEND_G002_EXT_DOMINANCE_DONE cases=${AB_START}-${AB_STOP}"
date
