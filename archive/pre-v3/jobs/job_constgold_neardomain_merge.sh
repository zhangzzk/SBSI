#!/bin/bash
#SBATCH --job-name=cg_nd_m
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nd_m_%j.out

# Merge the per-seed dumps written by jobs/job_constgold_neardomain_array.sh into the printed table
# and results/constgold_neardomain_table.npz. No GPU and no catalogue read -- the dumps carry the sim
# side already, so this is seconds of work.
#
# Chain it to the array so it cannot run on a partial set:
#   A=$(sbatch --parsable jobs/job_constgold_neardomain_array.sh)
#   sbatch --dependency=afterok:$A jobs/job_constgold_neardomain_merge.sh
#
# The merge REFUSES on: a fingerprint mismatch between dumps (different population), a repeated
# checkpoint (would double-weight a seed), or fewer than 16 dumps (`m` is an e-response quantity and
# 16 is the standard -- see AGENTS.md).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### CONSTGOLD NEAR-DOMAIN MERGE job=$SLURM_JOB_ID ###"; date
# DUMPS must track whatever DUMPDIR the array wrote to. A variant cut list (--complements) uses its
# own directory, and merging it against the canonical dumps would silently combine two different cut
# lists -- the fingerprint checks the POPULATION, not the cuts, so it would not catch that.
# EXPECT defaults to 16 -- unchanged behaviour. Lower it ONLY for a run whose reported quantity is a
# model-vs-model difference (column (4), dm, the leg-split boundary block), where the common-mode
# seed offset cancels and 4 seeds is genuinely enough. Never lower it for an absolute `m`.
python -u scripts/merge_neardomain_seeds.py \
  --dumps "${DUMPS:-results/nd_seeds/s*.json}" \
  --expect-seeds ${EXPECT:-16} \
  --save-npz ${OUT:-results/constgold_neardomain_table.npz} \
  2>&1 | grep -v --line-buffered "module command" || { echo CG_ND_M_FAILED; exit 1; }
echo CG_ND_M_ALL_DONE; date
