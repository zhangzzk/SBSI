#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g1
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_eachpair_g1_pilot_c400-409
OUT=$ROOT/results/anchorblend_eachpair_g1_v22_pilot_c400-409.json
for rank in $(seq 0 18); do
  base=$(printf '%s_rank%02d_pilot_c400-409' "$PREFIX" "$rank")
  test ! -e "$base" || { echo "REFUSING existing $base"; exit 1; }
done
for path in "$MANIFEST" "$OUT"; do
  test ! -e "$path" || { echo "REFUSING existing $path"; exit 1; }
done
cd "$ROOT"
j1=$(/opt/slurm/bin/sbatch --parsable jobs/job_anchorblend_eachpair_g1_catalog.sh)
j2=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_eachpair_g1_prepare.sh)
j3=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_eachpair_g1_sim.sh)
j4=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_eachpair_g1_shape.sh)
j5=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j4" jobs/job_anchorblend_eachpair_g1_analyze.sh)
printf '%s\n' "catalog=$j1" "prepare=$j2" "sim=$j3" "shape=$j4" "analysis=$j5"
