#!/bin/bash
#SBATCH --job-name=anti_shape
#SBATCH --time=20:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_shape_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_shape_%j.err

# Measure ngmix shapes for the sheared secondary half only. The existing +0.02
# target shapes predate the sub-pixel-centre fix, so --centering geometric is
# REQUIRED here for an extraction-matched estimator comparison.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/blendemu/scripts
echo "ANTITHETIC SELF SHAPES job=$SLURM_JOB_ID"; date
srun -n 100 --mpi=pmi2 python -u run_shape.py \
  /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --case_start 0 --realizations d,0,1 --shear_case=-0.02 \
  --stamp_size 48 --use_pos detect --pixel_scale 0.2 \
  --tile_name tile180.0_-0.5 --targets secondaries \
  --centering geometric
echo ANTITHETIC_SELF_SHAPE_DONE; date
