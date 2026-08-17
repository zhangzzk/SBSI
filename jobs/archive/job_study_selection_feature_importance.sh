#!/bin/bash
#SBATCH --job-name SBS_SEL_FI
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbs_selection_feature_importance.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbs_selection_feature_importance.%j.err

echo "START - SBS selection feature importance"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python -u SBSI/scripts/study_selection_feature_importance.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather \
    --output-dir SBSI/results/selection_feature_importance_v6_primary_frame \
    --pilot-output SBSI/models/selection_mlp_feature_importance_pilot_v6_primary_frame.pt \
    --target-column detected \
    --selection-name sextractor_detected \
    --max-rows 1500000 \
    --importance-rows 250000 \
    --epochs 20 \
    --batch-size 8192 \
    --predict-batch-size 65536 \
    --hidden-dim 256 \
    --n-layers 4 \
    --dropout 0.02 \
    --loss bce \
    --lr 0.0007 \
    --patience 5 \
    --permutation-repeats 3 \
    --num-workers 8

echo "FINISH"
date
