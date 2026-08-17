#!/bin/bash
#SBATCH --job-name=abv2c_shape
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-9%2
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2c_shape_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2c_shape_%A_%a.err
set -euo pipefail

ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BLOCK=$((SLURM_ARRAY_TASK_ID / 2))
LEG=$((SLURM_ARRAY_TASK_ID % 2))
START=$((400 + 100 * BLOCK))
STOP=$((START + 99))
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c${START}-${STOP}
if [ "$LEG" -eq 0 ]; then SHEAR=0.02; else SHEAR=-0.02; fi

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd /home/z/Zekang.Zhang/blendemu/scripts
srun -n 100 --mpi=pmi2 python -u run_shape.py "$BASE" \
  --case_start "$START" --realizations d,0,1 --shear_case="$SHEAR" \
  --stamp_size 48 --use_pos detect --pixel_scale 0.2 \
  --tile_name tile180.0_-0.5 --targets all --centering subpixel
for case in $(seq "$START" "$STOP"); do
  test -s "$BASE/case${case}_${SHEAR}/real0/catalogues/Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
done
echo "ANCHOR_V2_COMPLEMENT_SHAPE_DONE cases=$START-$STOP shear=$SHEAR"
