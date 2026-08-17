#!/bin/bash
#SBATCH --job-name=ab_l10shape
#SBATCH --time=20:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_l10shape_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_l10shape_%j.err
set -euo pipefail

SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py \
  --config /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_g005_local10_c200-299.yaml \
  --steps 3 --n-mpi 100

BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_local10_c200-299
missing=0
for case in $(seq 200 299); do
  for shear in 0.05 -0.05; do
    shape="$BASE/case${case}_${shear}/real0/catalogues/Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    [ -s "$shape" ] || {
      echo "Missing shape catalogue: case=$case shear=$shear"
      missing=$((missing + 1))
    }
  done
done
[ "$missing" -eq 0 ] || { echo "Missing shape catalogues: $missing"; exit 1; }

echo ANCHORBLEND_LOCAL10_SHAPE_JOB_DONE; date
