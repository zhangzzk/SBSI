#!/bin/bash
#SBATCH --job-name=popclose
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/popclose_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/popclose_%j.err

# Read-only, CPU-only truth closure. One shared intrinsic LSST pair mask is
# applied to the flow half-shear, R_blend half-shear, and constgold. No model is
# trained or tuned, and constgold affects no output other than this diagnostic.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUT=results/consistent_population_closure_model_v2_c40-99.npz
if [ -e "$OUT" ]; then
  echo "REFUSING to overwrite $OUT"
  exit 1
fi

echo "CONSISTENT POPULATION CLOSURE job=$SLURM_JOB_ID"
date
python -u scripts/eval_consistent_population_closure.py --output "$OUT"
echo CONSISTENT_POPULATION_JOB_DONE
date
