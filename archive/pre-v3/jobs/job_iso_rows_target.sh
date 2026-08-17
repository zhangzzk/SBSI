#!/bin/bash
#SBATCH --job-name=isorows
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/isorows_%j.out
set -eo pipefail

# De-risk the "add the isolated primaries back with zero neighbour flux" fix BEFORE any GPU spend.
# Establishes: which primaries the np7 target drops, whether the SNC g=0 lookup covers them, their
# response vs the kept rows, and how far the target moves if they are added -- compared against the
# +0.027 the measured target defect (B) requires. CPU only; no GPU, no training, no constgold m.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### ISO ROWS TARGET job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_iso_rows_target.py --max-case ${MAXCASE:-9} 2>&1 | grep -v "module command"
echo "ISOROWS_DONE"; date
