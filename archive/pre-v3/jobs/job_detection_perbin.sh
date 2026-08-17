#!/bin/bash
#SBATCH --job-name=det_perbin
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/det_perbin_%j.out
# Phase 0d: is the both-detected leg-matching bias structured per bin (esp. in TRUE SIZE)?
# Also serves Gold-V3 outstanding test 3 (close-pair detection selection) via the distance axis.
# CPU only. Reads the two per-leg constgold detection catalogues (~6G each).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PHASE-0d DETECTION PER-BIN job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_perbin.py "$@" || { echo "DET_PERBIN_FAILED"; exit 1; }
echo "DET_PERBIN_DONE"; date
