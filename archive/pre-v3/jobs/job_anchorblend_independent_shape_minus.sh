#!/bin/bash
#SBATCH --job-name=ab_indshpminus
#SBATCH --time=03:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indshpminus_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indshpminus_%j.err
set -euo pipefail

# Run after the two-leg render completes, independently of the +0.05 shape
# extraction.  The guarded pipeline shape stage waits for both one-leg jobs
# and then performs its usual completeness checks without rewriting them.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299
export PATH=$SIMS/bin:$PATH
export LD_LIBRARY_PATH=$SIMS/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
cd /home/z/Zekang.Zhang/blendemu/scripts
srun -n 100 --mpi=pmi2 python -u run_shape.py "$BASE" \
  --case_start 200 --realizations d,0,1 --shear_case=-0.05 \
  --stamp_size 48 --use_pos detect --pixel_scale 0.2 \
  --tile_name tile180.0_-0.5 --targets all --centering subpixel
for case in $(seq 200 299); do
  test -s "$BASE/case${case}_-0.05/real0/catalogues/Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
done
echo ANCHORBLEND_INDEPENDENT_SHAPE_MINUS_DONE
date
