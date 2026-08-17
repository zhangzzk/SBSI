#!/bin/bash
# Submit the firewall-clean random-direction antithetic target DAG.
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SIM=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
NEG=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_gm002_antithetic_c0-99.feather
TARGET=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_antithetic_self_c0-99_5x4x5_v21.npz

# A partial pre-existing leg needs an explicit recovery audit; blindly layering
# a new DAG on it could mix extraction versions. The initial run starts empty.
if find "$SIM" -maxdepth 1 -type d -name 'case*_-0.02' -print -quit | grep -q .; then
  echo "REFUSING: a -0.02 case tree already exists; audit/recovery must be explicit"
  exit 1
fi
if [ -e "$NEG" ] || [ -e "$TARGET" ]; then
  echo "REFUSING: negative catalogue or target output already exists"
  exit 1
fi

j1=$(sbatch --parsable jobs/job_antithetic_self_catalog.sh)
j2=$(sbatch --parsable --dependency=afterok:"$j1" jobs/job_antithetic_self_sim.sh)
j3=$(sbatch --parsable --dependency=afterok:"$j2" jobs/job_antithetic_self_shape.sh)
j4=$(sbatch --parsable --dependency=afterok:"$j3" jobs/job_antithetic_self_build.sh)
j5=$(sbatch --parsable --dependency=afterok:"$j4" jobs/job_antithetic_self_target.sh)

echo "catalog=$j1 sim=$j2 shape=$j3 build=$j4 target=$j5"
