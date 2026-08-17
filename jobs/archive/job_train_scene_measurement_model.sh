#!/bin/bash
#SBATCH --job-name SBSI_SCENE_MEAS
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_scene_measurement_flow.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_scene_measurement_flow.%j.err

echo "START - SBSI scene-conditioned measurement likelihood flow"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather}"
GEOMETRY_MODE="${GEOMETRY_MODE:-full}"
OUTPUT="${OUTPUT:-SBSI/models/scene_measurement_flow_detected_v1_${GEOMETRY_MODE}.pt}"
MAX_SCENES="${MAX_SCENES:-1000000}"
STOP_AFTER_SCENES="${STOP_AFTER_SCENES:-0}"

python -u SBSI/scripts/train_scene_measurement_model.py \
    --catalogue "${CATALOGUE}" \
    --output "${OUTPUT}" \
    --target-column detected \
    --selection-name sextractor_detected \
    --geometry-mode "${GEOMETRY_MODE}" \
    --aperture 3.0 \
    --max-neighbors 32 \
    --max-scenes "${MAX_SCENES}" \
    --stop-after-scenes "${STOP_AFTER_SCENES}" \
    --epochs 60 \
    --batch-size 4096 \
    --context-dim 128 \
    --set-hidden-dim 128 \
    --flow-hidden-dim 256 \
    --n-flows 8 \
    --lr 0.0007 \
    --patience 10 \
    --num-workers 8

echo "FINISH"
date
