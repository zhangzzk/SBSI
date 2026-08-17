#!/bin/bash
#SBATCH --job-name SBS_SEL_GRAD
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbs_selection_gradients.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbs_selection_gradients.%j.err

echo "START - SBS selection-gradient study"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -u SBSI/scripts/study_selection_gradients.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather \
    --model SBSI/models/selection_mlp_detected_v6_primary_frame.pt \
    --output-dir SBSI/results/selection_gradient_study_v6_primary_frame \
    --target-column detected \
    --radii 1 2 3 \
    --max-rows 1000000 \
    --gradient-rows 150000 \
    --finite-diff-rows 20000 \
    --finite-diff-eps 0.001 \
    --batch-size 65536 \
    --gradient-batch-size 16384

echo "FINISH"
date
