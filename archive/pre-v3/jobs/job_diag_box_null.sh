#!/bin/bash
#SBATCH --job-name=boxnull
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/boxnull_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
E=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval
set -e
python -u scripts/diag_box_null.py --npz $E/rblend_measured_allnbr_v21dom_ap7g0.2_ladder.npz
echo BOXNULL_DONE
