#!/bin/bash
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399
OUT=results/anchorblend_g005_response_v22_c300-399.feather
FINAL=results/anchorblend_coherent_scene_correction_v22_fresh_c300-399.json
if [ -e "$BASE" ] || [ -e "$OUT" ] || [ -e "$FINAL" ]; then
  echo "REFUSING: fresh-validation output already exists; audit before recovery"
  exit 1
fi
j1=$(/opt/slurm/bin/sbatch --parsable jobs/job_anchorblend_fresh_catalog.sh)
j2=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j1" jobs/job_anchorblend_fresh_sim.sh)
j3=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j2" jobs/job_anchorblend_fresh_shape.sh)
j4=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j3" jobs/job_anchorblend_fresh_response.sh)
j5=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$j4" jobs/job_anchorblend_fresh_scene_validation.sh)
echo "catalog=$j1 sim=$j2 shape=$j3 response=$j4 validation=$j5"
