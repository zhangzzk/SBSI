#!/bin/bash
#SBATCH --job-name=mshift
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/mshift_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### M SHIFT job=$SLURM_JOB_ID ###"; date
python -u scripts/estimate_m_shift.py --corrected results/blend_lookup_${SUFFIX:-indist}_c40-139.feather \
    2>&1 | grep -v --line-buffered "module command"
echo MSHIFT_DONE; date
