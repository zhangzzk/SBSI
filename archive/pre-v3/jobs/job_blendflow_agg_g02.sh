#!/bin/bash
#SBATCH --job-name=bfagg02
#SBATCH --time=4:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfagg02_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfagg02_%j.err

# Aggregate BOTH 16-seed ensembles under identical treatment (WORKLOG 2026-08-03e).
#
# READ THE chi2/dof COMPARISON WITH CARE -- IT IS NOT APPLES TO APPLES ACROSS SHEARS.
# chi2/dof measures the model-vs-anchor difference in units of the ANCHOR's own noise. At g=0.2 the
# label noise is ~4x lower, so the anchor sem is ~4x smaller and chi2/dof rises ~16x for an
# UNCHANGED model error. A higher chi2/dof at g=0.2 therefore does NOT mean the model got worse --
# it means errors that were previously buried under the noise floor are now resolved. That is the
# whole point of the exercise, and it is also the easiest thing in this report to misread.
#
# What IS directly comparable across shears: the summed bias in PERCENT, and the per-seed spread.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow

echo "############ g = 0.2 (NEW) ############"; date
python -u scripts/agg_blendflow_ensemble.py --glob "$CACHE/eval_ens_g02_s*.npz" --expect 16 || exit 1

echo; echo "############ g = 0.05 (INCUMBENT, same treatment) ############"; date
python -u scripts/agg_blendflow_ensemble.py --glob "$CACHE/eval_ens_s*.npz" --expect 16 || exit 1

echo "BFAGG02_DONE"; date
