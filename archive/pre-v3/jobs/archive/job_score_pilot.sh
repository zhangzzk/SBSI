#!/bin/bash
#SBATCH --job-name=score_pil
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/score_pil_%j.out
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SCORE PILOT job=$SLURM_JOB_ID ###"; date
python -u scripts/score_pin_pilot.py 2>&1 | grep -v --line-buffered "module command" || exit 1
echo SCORE_PIL_DONE; date
