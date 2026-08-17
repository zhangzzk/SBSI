#!/bin/bash
#SBATCH --job-name=blendap
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/blendap_%j.out
# Does the missing +0.0277 of <R_blend> live outside the emulator's k=20 / r_max=10" summation
# aperture? Rows at r_max=10 test k-truncation INSIDE the trained aperture (trustworthy); rows beyond
# are extrapolation and bound the far-field term from above. CPU-only, 3 constgold cases.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### BLEND APERTURE SCAN job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_blend_aperture.py --cases ${CASES:-40 41 42} \
  2>&1 | grep -v --line-buffered "module command" || { echo FAILED; exit 1; }
echo BLENDAP_DONE; date
