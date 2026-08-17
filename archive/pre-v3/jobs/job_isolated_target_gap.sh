#!/bin/bash
#SBATCH --job-name=isogap
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/isogap_%j.out
# Quantify RESULT 10: how much does a NEIGHBOURED-ONLY response target mis-state the response for a
# 24%-isolated deliverable population? Measured on the ruler catalogue, the only one containing both
# classes. Half-shear only -- no constgold response is read (its property occupancy is used for
# reweighting, which is occupancy, not response).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### ISOLATED TARGET GAP job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_isolated_target_gap.py --max-case ${MAXCASE:-9} \
  2>&1 | grep -v --line-buffered "module command" || { echo FAILED; exit 1; }
echo ISOGAP_DONE; date
