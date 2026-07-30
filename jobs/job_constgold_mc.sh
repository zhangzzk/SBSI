#!/bin/bash
#SBATCH --job-name=cgmc
#SBATCH --time=00:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgmc_%j.out

# Shift m on constgold in three shape columns (unsheared intrinsic | sheared intrinsic | measured).
# CPU only: reads one feather, no model. FIREWALL: constgold is EVALUATION only -- nothing is
# trained, fitted or selected on it here.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### CONSTGOLD m,c job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_constgold_mc.py 2>&1 | grep -v "module command" \
  || { echo CGMC_FAILED; exit 1; }
echo CGMC_ALL_DONE; date
