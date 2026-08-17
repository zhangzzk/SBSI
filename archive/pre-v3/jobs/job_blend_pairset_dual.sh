#!/bin/bash
#SBATCH --job-name=blpair2
#SBATCH --time=04:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blpair2_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/blpair2_%j.err

# STEP 1 OF MERGING FLOWS #1 AND #2 (WORKLOG 2026-08-02i).
#
# Same pair set as `job_blend_pairset.sh`, with three columns added:
#   self_truth  the primary's OWN shear response -- the identical measured-shape difference the
#               blend label uses, projected on the PRIMARY's shear direction instead of the
#               neighbour's. No new leg and no new join: the both-sheared leg carries both shear
#               directions and they are independent, so one extra projection buys the second label.
#   self_null   its 45-degree null, tested alongside the blend null. Both must sit at zero or the
#               decorrelation argument that makes EITHER label unbiased has broken.
#   k           the primary's annotated-neighbour count. The self response is a per-PRIMARY quantity
#               repeated across k rows, so it is averaged with weight 1/k; the blend label is
#               per-pair and is not. Crowding suppresses the self response, so getting this wrong is
#               a bias, not just an inefficiency -- the build prints both means so the size of it is
#               on the record rather than assumed small.
#
# Written to a NEW path. `blend_pairset_ap7.feather` is what the round-2 grids model trained on and
# is left untouched, so that result stays reproducible.
#
# No smoke run: `stream()` scans every batch of both 130 GB legs regardless of --max-case, so a
# small-case pass costs essentially the same wall time as the full build. The build's own null tests
# and label statistics are the check.
#
# FIREWALL: half-shear legs only, constgold never opened.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blend_pairset_ap7_dual.feather}
echo "### DUAL-LABEL BLEND PAIRSET job=$SLURM_JOB_ID -> $OUT ###"; date
python -u scripts/build_blend_pairset.py --output "$OUT" ${MAXCASE:+--max-case $MAXCASE} \
  || { echo "FAILED"; exit 1; }
echo "BLPAIR2_DONE"; date
