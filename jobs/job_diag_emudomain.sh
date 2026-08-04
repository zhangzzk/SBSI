#!/bin/bash
#SBATCH --job-name=diagemud
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/diagemud_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/diagemud_%j.err

# Why did R_flow move 0.7268 -> 0.6886 when R_sim did not move at all? See the script docstring.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u scripts/diag_emudomain_subset.py || exit 1
echo "DIAGEMUD_DONE"; date
