#!/bin/bash
#SBATCH --job-name=seedscale
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/seedscale_%j.out
# DIAGNOSTIC: does the measured-size selection bias shrink with more flow seeds?
# Not a gate; introduces no criterion. See scripts/eval_seed_scaling_sizebias.py.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SEED SCALING DIAGNOSTIC job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_seed_scaling_sizebias.py 2>&1 | grep -v --line-buffered "module command" \
  || { echo SEEDSCALE_FAILED; exit 1; }
echo SEEDSCALE_DONE; date
