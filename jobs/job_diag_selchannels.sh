#!/bin/bash
#SBATCH --job-name=selchan
#SBATCH --time=02:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/selchan_%j.out

# Why is the flow's magnitude-cut selection response flat? Measures the spin-2 orientation
# coupling b_mag / b_size (the flow's ONLY shear channel into measured mag/size) three ways:
# the pinned training target, the constgold sim, and the flow itself. Also compares the
# training population to the constgold evaluation population under identical cuts.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SELECTION CHANNEL DIAG job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/diag_selection_channels.py
echo "SELCHAN_DONE"; date
