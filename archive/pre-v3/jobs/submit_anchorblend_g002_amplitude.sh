#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c400-499
OUT=$ROOT/results/anchorblend_g002_response_v22_c400-499.feather
ANALYSIS=$ROOT/results/anchorblend_shear_amplitude_v22_c400-499.json
test ! -e "$BASE"
test ! -e "$OUT"
test ! -e "$ANALYSIS"
cd "$ROOT"
j1=$(/opt/slurm/bin/sbatch --parsable jobs/job_anchorblend_g002_catalog.sh)
j2=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_g002_sim.sh)
j3=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_g002_shape.sh)
j4=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_g002_response.sh)
j5=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j4" jobs/job_analyze_anchor_shear_amplitude.sh)
printf '%s\n' "catalog=$j1" "sim=$j2" "shape=$j3" "response=$j4" "analysis=$j5"
