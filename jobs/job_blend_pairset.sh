#!/bin/bash
#SBATCH --job-name=blpair
#SBATCH --time=04:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blpair_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/blpair_%j.err

# Training set for Gold-V3 flow #2 (the blend-response flow). Streams the two ap7 legs (130 GB
# each), keeps the both-sheared rows in the deliverable domain, and writes one row per
# (primary, neighbour) pair with the per-pair blending-response label.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blend_pairset_ap7.feather}
echo "### BLEND PAIRSET job=$SLURM_JOB_ID ###"; date
python -u scripts/build_blend_pairset.py --output "$OUT" ${MAXCASE:+--max-case $MAXCASE} \
  || { echo "FAILED"; exit 1; }
date
