#!/bin/bash
#SBATCH --job-name=scene_field
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/scene_field_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

NEST=${1:-600}
DEPTH=${2:-7}
echo "### SCENE-LEVEL COHERENT FORWARD MODEL v2 (field features)  n_est=$NEST depth=$DEPTH ###"
date
python -u scripts/scene_coherent_field.py --cases $(seq -s ' ' 0 39) \
  --n-folds 5 --n-estimators "$NEST" --max-depth "$DEPTH" \
  || { echo "SCENE_FIELD FAILED"; exit 1; }
date
echo "SCENE_FIELD_JOB_DONE"
