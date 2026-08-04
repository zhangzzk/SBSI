#!/bin/bash
#SBATCH --job-name=lk_v21
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v21_%j.out

# STEP 5 of V2.1: per-(case,input_index) summed R_blend on constgold cases 40-139, from the V2.1
# emulator `lsst_r_extnbr_v21`, so the V2.1 flow's constgold m can be formed by a pure join.
#
# ONE LEVER vs jobs/job_build_lookup_indomtuned.sh: --tag and the output name. Same script, same
# cases, same inputs, so any difference in the resulting m is the emulator and nothing else.
#
# COVERAGE IS THE THING TO CHECK, and it is expected to be LOW here -- much lower than the
# fiducial's. BlendingPredictor applies the emulator's stored cuts at inference, and V2.1's box is
# tighter than V2's on both axes (Re > 0.5 vs 0.3, mag < 25.72 vs 26). The fiducial `_indom_tuned`
# lookup already covered only ~13.4M of the ~56.3M rows in the wide `_ho` lookup; V2.1 will cover
# less again.
#
# THAT IS ONLY SAFE BECAUSE THE V2.1 EVALUATION IS RESTRICTED TO THE V2.1 DOMAIN. AGENTS.md "Two
# traps" #1 is exactly this: on a wider population the unmatched rows fall back to R_blend = 0,
# which collapsed <R_blend> from 0.1593 to 0.0589 and manufactured a spurious +28.9% m. So the
# evaluator MUST assert the match fraction inside the domain and DROP unmatched rows rather than
# zero-fill them. Do not reuse this lookup on the wide population.
#
# FIREWALL: constgold is read only for POSITIONS and TRUE properties, as the certified build does.
# Nothing is fitted; no constgold measurement enters the emulator.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
mkdir -p results
echo "### blend_lookup 40-139, tag=lsst_r_extnbr_v21 job=$SLURM_JOB_ID ###"; date
python -u scripts/build_blend_lookup.py --cases $(seq 40 139) --tag lsst_r_extnbr_v21 \
  --output results/blend_lookup_v21_c40-139.feather 2>&1 | grep -v --line-buffered "module command" \
  || { echo "LK_V21_FAILED (build)"; exit 1; }
if [ -f results/blend_lookup_v21_c40-139.feather ]; then echo LK_V21_DONE;
else echo "LK_V21_FAILED (no output)"; exit 1; fi
date
