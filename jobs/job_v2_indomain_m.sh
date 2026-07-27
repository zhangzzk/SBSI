#!/bin/bash
#SBATCH --job-name=v2indom
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2indom_%j.out

# Gold-V2 8-seed constgold m under the certified convention vs inside the FLOW TRAINING DOMAIN
# (true Re>0.3, true mag<26.0). Reads the existing per-object dumps; no GPU pass needed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### V2 IN-DOMAIN m job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_v2_indomain_m.py
echo "V2INDOM_DONE"; date
