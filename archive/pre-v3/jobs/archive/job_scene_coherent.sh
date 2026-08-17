#!/bin/bash
#SBATCH --job-name=scene_coh
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/scene_coh_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

NEST=${1:-600}
DEPTH=${2:-7}
LR=${3:-0.05}
FOLDS=${4:-5}

echo "### SCENE-LEVEL COHERENT FORWARD MODEL  n_est=$NEST depth=$DEPTH lr=$LR folds=$FOLDS ###"
date
python -u scripts/scene_coherent_model.py \
  --max-rows 0 --n-folds "$FOLDS" \
  --n-estimators "$NEST" --max-depth "$DEPTH" --lr "$LR" \
  || { echo "SCENE_COHERENT FAILED"; exit 1; }
date
echo "SCENE_COHERENT_JOB_DONE"
