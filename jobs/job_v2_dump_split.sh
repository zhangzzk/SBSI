#!/bin/bash
#SBATCH --job-name=v2split
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2split_%j.out

# Split the 8 Gold-V2 per-seed constgold dumps (the ones figv2_fig3 is drawn from) into the
# GOALS.md acceptance population, to test whether that figure's -0.46% is a closure on the
# deliverable or a cancellation between accepted and rejected rows -- the effect cont.161
# established for V1.  8 x 27M rows plus a catalogue join: too heavy for the login node.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b
echo "### V2_SPLIT job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/split_v2_dumps_acceptance.py
echo "### V2_SPLIT_DONE job=$SLURM_JOB_ID ###"; date
