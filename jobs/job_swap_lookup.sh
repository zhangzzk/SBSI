#!/bin/bash
#SBATCH --job-name=swaplook
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/swaplook_%j.out

# Fill the missing flow x emulator cell: the CERTIFIED Gold-v1 flow with the CORRECTED R_blend
# emulator. No GPU and no flow rerun -- the per-object dumps already carry r_sim and R_flow, and
# validate_constant_with_blend uses the lookup R_blend unmodified (rb_add = rb_i), so the swap is a
# join. Deliberately NO --gres: this is pandas/numpy only, and asking for a GPU would put it behind
# the saturated inter queue for nothing.
set -o pipefail   # python is piped into grep, so without this a python crash exits 0 and the job
                  # reports COMPLETED with a truncated result. That happened on 15348051/2, which
                  # died reading a dump the constgold array was concurrently rewriting.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SWAP LOOKUP job=$SLURM_JOB_ID ###"; date

# PROVENANCE GUARD (added 2026-07-30 after a near-miss). A 2-trial SMOKE TEST (job 15355997) wrote
# real files at models/*_lsst_r_extnbr_indom_tuned.*, and the 100-trial search (15355998) only
# overwrites them when it FINISHES. So between those two events the path named "_indom_tuned" holds
# an essentially UNTUNED model (global R2 +0.000009 over inherited) -- promoting it would look like
# "tuning bought nothing" when tuning had not actually been run.
#
# The metadata records its own provenance in metrics.n_trials, so check it. Set TUNED_TAG to the tag
# being promoted; unset (the default) skips the guard for non-tuned lookups.
if [ -n "${TUNED_TAG}" ]; then
  python -u scripts/check_emulator_provenance.py --tag "${TUNED_TAG}" \
    || { echo "SWAPLOOK_ABORTED (provenance guard)"; exit 1; }
fi

python -u scripts/eval_swap_lookup.py \
    --lookup "${LOOKUP:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/blend_lookup_wc5_c40-139.feather}" \
    --new-label "${NEWLABEL:-wc5}" ${EXTRA} 2>&1 | grep -v --line-buffered "module command"
echo SWAPLOOK_DONE; date
