#!/bin/bash
#SBATCH --job-name=crowdcmp
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/crowdcmp_%j.out
set -euo pipefail

# Are constgold and the half-shear legs equally blended?
#
# Job 15596684 measured, emulator-free, that the half-shear sim's own total response inside 7"
# (R_self 0.8165 + R_blend 0.0865 = 0.9030) sits 6.2% below constgold's R_sim 0.9626, and that
# closing it from R_self would need a blend of 0.1460 -- 69% more than the sim delivers. Either
# constgold scenes are genuinely more crowded (a population difference, fixable by retargeting),
# or the two sims disagree about the response itself (nothing about the emulator or the target
# fixes V2.2). Both catalogues are pair-annotated with a `distance` column, so crowding can be
# compared like for like.
#
# Read the RISING part of both curves only: each build has its own k and r_max, and the ap7 leg is
# capped at 7", so a flat tail reports the build cap rather than the sky. Same pair-list discipline
# that invalidated job 15596566.
#
# Nothing is corrected. No model, no emulator, no `m`. CPU only.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### CROWDING: constgold vs half-shear, pairs per object ###"; date
"$PY" -u scripts/diag_crowding_compare.py --mag-max 25.8 --re-min 0.5
echo CROWDCMP_DONE; date
