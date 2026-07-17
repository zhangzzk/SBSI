#!/bin/bash
#SBATCH --job-name SBSI_SEL_G0
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selection_g0.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selection_g0.%j.err

echo "START - SBSI g=0 shear-free selection MLP"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# g=0 detection+measured catalogue re-derived from the live blendemu run
# (lsst_sims_fs2_25876) via SBSI/scripts/build_detection_measurement_catalogue.py.
# NOTE: the pipeline's own detection_catalogue_train.feather is g=0.05 only and
# has no measured columns, so it is NOT used for the shear-free forward model.
CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather}"
OUTPUT="${OUTPUT:-SBSI/models/selection_mlp_g0_shearfree_v1.pt}"

python -u SBSI/scripts/train_selection_model.py \
    --catalogue "$CATALOGUE" \
    --output "$OUTPUT" \
    --target-column detected \
    --selection-name sextractor_detected \
    --feature-set g0_shearfree \
    --shear-case 0.0 \
    --max-rows 4000000 \
    --epochs 60 \
    --batch-size 8192 \
    --hidden-dim 256 \
    --n-layers 4 \
    --dropout 0.02 \
    --loss bce \
    --lr 0.0007 \
    --patience 10 \
    --num-workers 8

echo "FINISH"
date
