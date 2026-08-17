#!/bin/bash
#SBATCH --job-name=extr_gdep
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/extr_gdep_%j.out
# Phase 0b: is the forward-vs-antithetic extraction difference size-dependent?
# Measured as dR_fwd/dg from the g=0.02 and g=0.05 half-shear legs against the shared g=0 leg.
# CPU only. Reads three half-shear legs restricted to cases 0-19 (the g0.02 leg's coverage).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PHASE-0b EXTRACTION g-DEPENDENCE job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_extraction_gdep.py "$@" || { echo "EXTR_GDEP_FAILED"; exit 1; }
echo "EXTR_GDEP_DONE"; date
