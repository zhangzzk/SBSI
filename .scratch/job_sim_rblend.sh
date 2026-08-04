#!/bin/bash
#SBATCH --job-name=sim_rb
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sim_rb_%j.out
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u .scratch/sim_rblend.py 2>&1 | grep -v --line-buffered "module command"
echo SIM_RB_DONE
