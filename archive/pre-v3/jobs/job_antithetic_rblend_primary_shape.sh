#!/bin/bash
#SBATCH --job-name=anti_rbshape
#SBATCH --time=04:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_rbshape_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_rbshape_%j.err
set -euo pipefail

# The -0.02 images already exist from the antithetic self-response audit, but
# only secondary shapes were extracted then.  Measure the primaries using the
# historical geometric-centre convention of the existing 0 and +0.02 primary
# catalogues, so the three-leg finite-difference comparison changes no other
# extraction choice.
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$SIMS/bin:$PATH
export LD_LIBRARY_PATH=$SIMS/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd /home/z/Zekang.Zhang/blendemu/scripts
srun -n 100 --mpi=pmi2 python -u run_shape.py \
  /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --case_start 0 --realizations d,0,1 --shear_case=-0.02 \
  --stamp_size 48 --use_pos detect --pixel_scale 0.2 \
  --tile_name tile180.0_-0.5 --targets primaries \
  --centering geometric

for case in $(seq 0 99); do
  test -s "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/case${case}_-0.02/real0/catalogues/Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather"
done
echo ANTITHETIC_RBLEND_PRIMARY_SHAPE_DONE
date
