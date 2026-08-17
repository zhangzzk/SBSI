#!/bin/bash
#SBATCH --job-name=hs_coherence
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_coherence_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_coherence_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/audit_halfshear_shear_coherence.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --primary-table /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_detection_c0-39.feather \
  --table-output /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --output results/halfshear_shear_coherence_corrected_v22_c0-39.json
echo HALFSHEAR_SHEAR_COHERENCE_JOB_DONE
date
