#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BASE_U=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_u_c400-499
BASE_V=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_v_c400-499
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
OUT_FEATHER=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.feather
OUT_JSON=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.json
for path in "$BASE_U" "$BASE_V" "$MANIFEST" "$OUT_FEATHER" "$OUT_JSON"; do
  test ! -e "$path" || { echo "REFUSING existing $path"; exit 1; }
done
cd "$ROOT"
j1=$(/opt/slurm/bin/sbatch --parsable jobs/job_anchorblend_oneactive_catalog.sh)
j2=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_oneactive_prepare.sh)
j3=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_oneactive_sim.sh)
j4=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_oneactive_shape.sh)
j5=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j4" jobs/job_anchorblend_oneactive_analyze.sh)
printf '%s\n' "catalog=$j1" "prepare=$j2" "sim=$j3" "shape=$j4" "analysis=$j5"
