#!/bin/bash
#SBATCH --job-name=errbudget
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/errbudget_%j.out

# Decompose the uncertainty on the constgold ensemble m into (A) which fields were simulated and
# (B) which flow training seeds, and fix the R_flow-frozen bootstrap in validate_constant_with_blend
# (it holds the denominator fixed while resampling the very objects it averages over, which discards
# a real cancellation and inflates the sim-side error). Reads the per-object dumps already on disk.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### ERROR BUDGET job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_error_budget.py \
    --dump-dir "${DUMPDIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps}" \
    --pattern "${PATTERN:-indist_perobj_s*.feather}" \
    --label "${LABEL:-indist_wc5}" 2>&1 | grep -v --line-buffered "module command"
echo ERRBUDGET_DONE; date
