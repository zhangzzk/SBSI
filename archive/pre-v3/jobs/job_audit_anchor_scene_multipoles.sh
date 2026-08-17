#!/bin/bash
#SBATCH --job-name=ab_multipole
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_multipole_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_multipole_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"
"$PY" -u scripts/audit_anchor_scene_multipoles.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_detection_c200-299.feather \
  --table-output /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_multipoles_c200-299.feather \
  --output results/anchor_scene_multipoles_v22_c200-299.json
echo ANCHOR_SCENE_MULTIPOLE_JOB_DONE
date
