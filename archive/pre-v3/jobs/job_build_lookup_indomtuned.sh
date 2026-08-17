#!/bin/bash
#SBATCH --job-name=lk_indomt
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_indomt_%j.out

# Per-(case,input_index) summed R_blend on constgold cases 40-139 from the OPTUNA-TUNED in-domain
# emulator `lsst_r_extnbr_indom_tuned` (73 trials, finalized 2026-07-30 21:43), so the certified
# constgold m can be recomputed with it by a pure join (scripts/eval_swap_lookup.py).
#
# Byte-for-byte the same call as the certified lookup build (jobs/job_build_4079.sh +
# job_build_100.sh, and jobs/archive/job_build_lookup_indist.sh) -- same script, same cases 40-139,
# same --sign 0.02 input fields -- with ONLY --tag changed. Any difference in the resulting m is
# therefore the emulator and nothing else.
#
# COVERAGE NOTE (expected, not a bug): `_indom_tuned` inherits `_indom`'s regression cuts
# (primary mag 18-26, Re 0.3-1.5), and BlendingPredictor applies the stored cuts at inference, so
# this lookup will cover ~13.4M of the ~56.3M rows the certified `_ho` lookup covers. It is valid
# on the IN-DOMAIN population only; on the wide certified convention the unmatched rows fall back
# to R_blend = 0, which is a coverage artifact, not a tuning effect.
#
# FIREWALL: constgold is read only for POSITIONS and TRUE properties to evaluate the emulator, as
# the certified lookup build does. No constgold measurement enters the emulator or its training,
# and nothing is fitted here.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
mkdir -p results

echo "### blend_lookup 40-139, tag=lsst_r_extnbr_indom_tuned job=$SLURM_JOB_ID ###"; date

# Provenance guard: refuse if the path holds the 2-trial smoke-test artifact rather than the real
# search (WORKLOG 2026-07-30m).
python -u scripts/check_emulator_provenance.py --tag lsst_r_extnbr_indom_tuned \
  || { echo "LK_INDOMT_ABORTED (provenance guard)"; exit 1; }

python -u scripts/build_blend_lookup.py --cases $(seq 40 139) --tag lsst_r_extnbr_indom_tuned \
  --output results/blend_lookup_indomtuned_c40-139.feather 2>&1 | grep -v "module command" \
  || { echo "LK_INDOMT_FAILED (build)"; exit 1; }

if [ -f results/blend_lookup_indomtuned_c40-139.feather ]; then echo LK_INDOMT_DONE;
else echo "LK_INDOMT_FAILED (no output)"; exit 1; fi
date
