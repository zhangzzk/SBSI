#!/bin/bash
#SBATCH --job-name=v2figs23
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2figs23_%j.out

# figv2_fig2 (response vs galaxy properties) + figv2_fig3 (per-seed m) from the 8 per-object
# constgold dumps. Too heavy for the login node: 8 x 27M rows plus the catalogue/crowd joins.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### V2 FIGS 2+3 job=$SLURM_JOB_ID ###"; date
python -u plotting/plot_v2_flow_figures.py
echo "V2FIGS23_DONE"; date
