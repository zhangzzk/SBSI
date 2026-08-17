#!/bin/bash
#SBATCH --job-name=abep2_prep
#SBATCH --time=03:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abep2_prep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abep2_prep_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
PAIR_PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g2
COHERENT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_coherent_g2_pilot_c400-409
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_eachpair_g1_pilot_c400-409
DESIGN=$ROOT/results/anchorblend_eachpair_g2_design_pilot_c400-409.json
test -d "$MANIFEST"
test ! -e "$DESIGN" || { echo "REFUSING existing $DESIGN"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/prepare_anchorblend_eachpair_g2.py \
  --reference-base "$REFERENCE" --pair-prefix "$PAIR_PREFIX" \
  --coherent-base "$COHERENT" --manifest-dir "$MANIFEST" \
  --design-json "$DESIGN" --cases $(seq 400 409) --g 0.05 --max-rank 18
echo ANCHORBLEND_EACHPAIR_G2_PREP_JOB_DONE
date
