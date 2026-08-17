#!/bin/bash
#SBATCH --job-name=add_iso
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/add_iso_%j.out
# Phase 0a: does the model hold up where the emulator contributes nothing (neighbored=False)?
# Gates the 2026-08-01s flow-vs-emulator attribution, and with it Phase 2.
# CPU only -- reads existing dumps, scores nothing. 13G of per-object dumps x 16 seeds.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PHASE-0a ADDITIVITY ISOLATED job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_additivity_isolated.py "$@" || { echo "ADD_ISO_FAILED"; exit 1; }
echo "ADD_ISO_DONE"; date
