#!/bin/bash
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005
if [ -e "$BASE" ] || [ -e results/anchorblend_g005_response.feather ]; then
  echo "REFUSING: anchorblend output already exists; audit before recovery"
  exit 1
fi
j1=$(sbatch --parsable jobs/job_anchorblend_catalog.sh)
j2=$(sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_sim.sh)
j3=$(sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_shape.sh)
j4=$(sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_response.sh)
echo "catalog=$j1 sim=$j2 shape=$j3 response=$j4"
