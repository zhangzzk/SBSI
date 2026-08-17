#!/bin/bash
#SBATCH --job-name SBS_SEL_BLEND
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbs_selection_blends.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbs_selection_blends.%j.err

echo "START - SBS close-blend selection diagnostics"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -u SBSI/scripts/evaluate_selection_blends.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather \
    --model SBSI/models/selection_mlp_detected_v6_primary_frame.pt \
    --output-dir SBSI/results/selection_blends_v6_primary_frame \
    --target-column detected \
    --radii 2 3 \
    --max-rows 1000000 \
    --gradient-rows 100000 \
    --batch-size 65536 \
    --gradient-batch-size 16384

echo "FINISH"
date
