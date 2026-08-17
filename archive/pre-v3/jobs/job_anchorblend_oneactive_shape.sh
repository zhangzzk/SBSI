#!/bin/bash
#SBATCH --job-name=ab1n_shape
#SBATCH --array=0-3%4
#SBATCH --time=02:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_shape_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_shape_%A_%a.err
set -euo pipefail

BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$TASK" -lt 2 ]; then MODE=u; else MODE=v; fi
if [ $((TASK % 2)) -eq 0 ]; then SHEAR=0.05; else SHEAR=-0.05; fi
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_${MODE}_c400-499
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
srun -n 100 --mpi=pmi2 python -u run_shape.py "$BASE" \
  --case_start 400 --realizations d,0,1 --shear_case="$SHEAR" \
  --stamp_size 48 --use_pos detect --pixel_scale 0.2 \
  --tile_name tile180.0_-0.5 --targets all --centering subpixel
for case in $(seq 400 499); do
  test -s "$BASE/case${case}_${SHEAR}/real0/catalogues/Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
done
echo "ANCHORBLEND_ONEACTIVE_SHAPE_DONE mode=$MODE shear=$SHEAR"
date
