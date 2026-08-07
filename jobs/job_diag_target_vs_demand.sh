#!/bin/bash
#SBATCH --job-name=tgtdem
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tgtdem_%j.out
set -euo pipefail

# Why is the V2.2 response TARGET ~2.5% below what constgold demands?
#
# 2026-08-07d measured the two levels on the V2.2 box: the half-shear self-response (the target's
# own level) is 0.8161 +- 0.0036, while constgold's demand R_sim - R_blend is 0.8369. The flow sits
# slightly ABOVE its target and still falls short -- so the flow is not the culprit, the target is.
# Shear nonlinearity is already excluded as the carrier (+0.16% +- 0.39, job 15595564).
#
# This job separates the POPULATION term from the LEVEL term by re-averaging the half-shear
# self-response under constgold's (mag, size) cell counts. That reweighting stays inside the
# half-shear catalogue, so it crosses no convention and is clean; whatever survives it is level,
# i.e. R_blend and/or the forward-vs-antithetic convention, which this job does NOT try to
# separate from each other. The per-cell table then says whether the survivor is flat (level) or
# tilted (a population axis the 2D grid misses, e.g. crowding).
#
# Nothing is corrected here. Per AGENTS.md "Numerical Integrity", a gap measured is not a gap to
# fold into anything.
#
# Seeds: R_sim and R_blend are both seed-independent, so one dump carries the demand exactly. The
# R_flow line is single-seed and labelled as context only. No `m` is reported, so the 16-seed rule
# does not apply. CPU only.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### TARGET LEVEL vs CONSTGOLD DEMAND: population or level? ###"; date
"$PY" -u scripts/diag_target_vs_demand.py \
  --g 0.02 --mag-max 25.8 --re-min 0.5 --max-case 99 --min-case 40
echo TGTDEM_DONE; date
