#!/bin/bash
#SBATCH --job-name=abg002x_gate
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

# Cheap replication gate, meant to run BEFORE the four extension renders.
#
# The extension scores the dominance split through a new pair-response path
# (`build_anchorblend_pair_responses.py` reading the g=0.02 tree) instead of the
# one-active derisk manifest that produced the published cases 400--499 table.
# This job rebuilds cases 400--499 through the new path and requires it to
# reproduce the published tail / bulk gaps.  If it does not, the extension is
# not worth rendering until the discrepancy is understood.
#
# The comparison must be made PAIRED.  The published analysis inner-joins the
# g=0.02 and g=0.05 response tables, which silently drops the 958 anchors (0.28%)
# that the g=0.02 render has and the g=0.05 render does not.  The extension is
# deliberately unpaired, so scoring it unrestricted moves the tail gap by 4.1e-4
# for population reasons alone -- that is the intended design difference, not a
# path defect, and the first version of this gate wrongly failed on it.
# `--restrict-keys` puts both sides on the same anchor set; the unpaired value is
# then reported alongside as information only.

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c400-499
NEW=$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c400-499_newpath_paired.json
UNPAIRED=$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c400-499_newpath.json
G005=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
PUBLISHED=$ROOT/results/anchorblend_amplitude_by_dominance_v22_c400-499.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$NEW" || { echo "REFUSING existing $NEW"; exit 1; }
test -s "$PUBLISHED" || { echo "MISSING $PUBLISHED"; exit 1; }
cd "$ROOT"

AB_START=400 AB_STOP=499
if [ ! -s "$BASE/pairs_case400.feather" ]; then
  python -u scripts/build_anchorblend_pair_responses.py \
    --base "$BASE" --manifest-dir "$BASE" --cases $(seq 400 499) --g 0.02 \
    --tag lsst_r_extnbr_v22 \
    --summary-json "$ROOT/results/anchorblend_g002_pairs_v22_c400-499.json"
fi

STEM=$ROOT/results/anchorblend_g002_dominance_v22_c400-499
if [ ! -s "$STEM.feather" ]; then
  python -u scripts/localize_anchorblend_response_dominance.py \
    --manifest-dir "$BASE" --reference-base "$BASE" --reference-shear 0.02 \
    --tag lsst_r_extnbr_v22 \
    --case-min 400 --case-max 499 --development-max 449 \
    --dominance-threshold 0.7752772106835227 \
    --output-feather "$STEM.feather" --output-json "$STEM.json" \
    --output-md "$STEM.md"
fi

# (1) the graded comparison: same anchor set as the published paired analysis
python -u scripts/analyze_anchor_g002_gap_by_dominance.py \
  --g002 results/anchorblend_g002_response_v22_c400-499.feather \
  --dominance "$STEM.feather" --restrict-keys "$G005" \
  --ratio-threshold 20 --case-min 400 --case-max 499 \
  --replicate-window 400 499 \
  --output "$NEW"

# (2) the unrestricted value the extension will actually report, for reference
if [ ! -e "$UNPAIRED" ]; then
  python -u scripts/analyze_anchor_g002_gap_by_dominance.py \
    --g002 results/anchorblend_g002_response_v22_c400-499.feather \
    --dominance "$STEM.feather" \
    --ratio-threshold 20 --case-min 400 --case-max 499 \
    --replicate-window 400 499 \
    --output "$UNPAIRED"
fi

NEW="$NEW" UNPAIRED="$UNPAIRED" PUBLISHED="$PUBLISHED" python - <<'PY'
import json
import os

new = json.load(open(os.environ["NEW"]))
free = json.load(open(os.environ["UNPAIRED"]))
old = json.load(open(os.environ["PUBLISHED"]))
print(f"kept {new['restricted_fraction']:.4%} of g=0.02 anchors when matched to "
      f"the g=0.05 partner set")
print(f"{'group':6s} {'published':>12s} {'paired new':>12s} {'delta':>10s} "
      f"{'unpaired new':>13s} {'popn shift':>11s}")
worst = 0.0
for name in ("all", "tail", "rest"):
    a = old[name]["gap_g002"]["mean"]
    b = new[name]["gap"]["mean"]
    c = free[name]["gap"]["mean"]
    worst = max(worst, abs(a - b))
    print(f"{name:6s} {a:12.6f} {b:12.6f} {a - b:10.2e} {c:13.6f} {c - b:11.2e}")
if worst > 1e-6:
    raise SystemExit(
        f"GATE FAILED: on the SAME anchor set the new pair-response path moves a "
        f"group gap by {worst:.3e} (> 1e-6); do not render the extension until "
        "this is understood"
    )
print(f"GATE PASSED: worst same-population group-gap difference {worst:.3e}")
print("the 'popn shift' column is the deliberate paired->unpaired population "
      "change, not a path error")
PY
echo ANCHORBLEND_G002_EXT_GATE_DONE
date
