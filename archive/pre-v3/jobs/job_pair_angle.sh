#!/bin/bash
#SBATCH --job-name=pair_angle
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/pairangle_%j.out

# Two decision-independent checks for Gold-V3 flow #2 (see scripts/eval_pair_angle.py docstring):
#   1. does the sim shear POSITIONS as well as shapes -> must the loss shift move the separation
#      vector, or is the existing shape-only shift already faithful to the sim?
#   2. does the pair ORIENTATION carry real spin-2 signal, and does the -41% close-pair deficit
#      survive averaging over it (it should, if the deficit is a selection effect rather than an
#      omitted-variable effect).
# FIREWALL: half-shear legs only; constgold is never read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PAIR ANGLE + POSITION SHEAR job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_pair_angle.py \
    --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval/pair_angle.npz \
    2>&1 | grep -v "module command"
echo PAIRANGLE_DONE; date
