#!/bin/bash
#SBATCH --job-name=fid_figs
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/fid_figs_%j.out

# Figures 1-4 under the FIDUCIAL model: V2 dom6x6 flow + tuned in-domain blend emulator.
# 16 seeds x 26.9M rows plus a 26.9M-row join is well past login-node work.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### FIDUCIAL FIGURES job=$SLURM_JOB_ID ###"; date
python -u plotting/plot_fid_flow_figures.py 2>&1 | grep -v --line-buffered "module command" || { echo FID_FIGS_FAILED; exit 1; }
echo FID_FIGS_ALL_DONE; date
