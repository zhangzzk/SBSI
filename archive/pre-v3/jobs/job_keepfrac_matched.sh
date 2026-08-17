#!/bin/bash
#SBATCH --job-name=keepfrac
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/keepfrac_%j.out
# Phase 0c: compare TRUE-size and MEASURED-size cuts at matched KEEP FRACTION, not matched threshold.
# All cuts are FROZEN (read off the catalogue), so no GPU is needed -- the dumps already have R_flow.
# CPU only -- reads existing dumps, scores nothing. 13G of per-object dumps x 16 seeds.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PHASE-0c KEEP-FRACTION MATCHED job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_keepfrac_matched.py "$@" || { echo "KEEPFRAC_FAILED"; exit 1; }
echo "KEEPFRAC_DONE"; date
