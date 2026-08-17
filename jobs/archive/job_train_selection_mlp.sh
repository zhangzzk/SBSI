#!/bin/bash
#SBATCH --job-name SBS_SEL_MLP
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbs_selection_mlp.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbs_selection_mlp.%j.err

echo "START - SBS selection MLP"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python -u SBSI/scripts/train_selection_model.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather \
    --output SBSI/models/selection_mlp_detected_v6_primary_frame.pt \
    --target-column detected \
    --selection-name sextractor_detected \
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
