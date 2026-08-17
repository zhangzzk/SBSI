#!/bin/bash
#SBATCH --job-name=rbsim
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbsim_%j.out
set -euo pipefail

# Measure R_blend straight from the half-shear sim, with NO emulator involved.
#
# After 2026-08-07e removed the (mag, size) population term (+0.28%), a -2.39% level gap remains
# between the half-shear self-response (0.8146) and constgold's demand R_sim - R_blend (0.8369).
# Either the emulator's R_blend (0.1257) is too small -- the per-pair ruler already says it is low,
# v22 -9.78% / _ho -6.89% -- or the two sims genuinely disagree about the self response, in which
# case no emulator change closes V2.2.
#
# The half-shear legs give every galaxy an independent random shear direction, so the same rows
# yield the self response under a ghat_p projection and the blend response under a ghat_s
# projection onto the NEIGHBOUR's direction. Self is AVERAGED over an object's rows (identical
# values, one detection); blend is SUMMED over pairs (additive over neighbours). That gives a
# sim-measured R_blend the emulator never touches.
#
# Reported as a cumulative curve in pair separation, never as a single number: R_blend is a sum
# over a pair list, and AGENTS.md records a case where changing that list moved a verdict ninefold.
#
# MUST run against an ALL-PAIRS leg. The first attempt (job 15596566) used det_meas_crowd_*, which
# annotates only the NEAREST neighbour -- 15.70M rows for 15.70M objects -- so the sum ran over one
# neighbour and returned 0.0145 against a needed 0.148. That was a pair-list artefact, not a
# measurement, and is discarded. det_meas_ngmix_ap7_* is the all-pairs build (25.66M rows, 7" cap).
# The script now prints rows-per-object and refuses to let a ~1.0 build pass unremarked.
#
# The closing comparison against constgold R_sim crosses catalogues and conventions, so it is a
# flag rather than a proof; the known convention term (-0.55% +- 0.61) is printed, not applied.
# Nothing here is corrected. No `m` is reported. CPU only.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### R_blend MEASURED FROM THE SIM (ghat_s projection), no emulator ###"; date
"$PY" -u scripts/diag_rblend_from_sim.py \
  --g 0.02 --mag-max 25.8 --re-min 0.5 --max-case 99
echo RBSIM_DONE; date
