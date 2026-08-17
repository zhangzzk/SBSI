#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PAIR_PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g2
COHERENT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_coherent_g2_pilot_c400-409
DESIGN=$ROOT/results/anchorblend_eachpair_g2_design_pilot_c400-409.json
OUT=$ROOT/results/anchorblend_eachpair_orthogonal_v22_pilot_c400-409.json
for rank in $(seq 0 18); do
  base=$(printf '%s_rank%02d_pilot_c400-409' "$PAIR_PREFIX" "$rank")
  test ! -e "$base" || { echo "REFUSING existing $base"; exit 1; }
done
for path in "$COHERENT" "$DESIGN" "$OUT"; do
  test ! -e "$path" || { echo "REFUSING existing $path"; exit 1; }
done
cd "$ROOT"
j1=$(/opt/slurm/bin/sbatch --parsable jobs/job_anchorblend_eachpair_g2_catalog.sh)
j2=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_eachpair_g2_prepare.sh)
j3=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_eachpair_g2_sim.sh)
j4=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_eachpair_g2_shape.sh)
j5=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j4" jobs/job_anchorblend_eachpair_orthogonal_analyze.sh)
printf '%s\n' "catalog=$j1" "prepare=$j2" "sim=$j3" "shape=$j4" "analysis=$j5"
