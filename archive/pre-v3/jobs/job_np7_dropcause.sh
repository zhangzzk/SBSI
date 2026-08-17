#!/bin/bash
#SBATCH --job-name=np7drop
#SBATCH --time=01:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7drop_%j.out
set -eo pipefail

# Separate the two explanations for np7 dropping half the detected primaries: a 7" GEOMETRY cut
# (isolated galaxies wrongly excluded -> the fix is real) vs a MEASUREMENT cut (ngmix failures ->
# nothing to add). CPU only; no GPU, no training, no constgold.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### NP7 DROP CAUSE job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_np7_dropcause.py --max-case ${MAXCASE:-4} 2>&1 | grep -v "module command"
echo "NP7DROP_DONE"; date
