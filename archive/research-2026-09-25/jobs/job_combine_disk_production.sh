#!/bin/bash
#SBATCH --job-name=sbsi-v36-combine
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v36_combine_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
PARTS=()
for ((INDEX=0; INDEX<20; INDEX++)); do
  printf -v PART '%02d' "$INDEX"
  PARTS+=(--part "$RUN_ROOT/production_lru_v1/part_$PART")
done
"$PYTHON" scripts/combine_inference_partitions.py "${PARTS[@]}" \
  --output "$RUN_ROOT/production_lru_v1/combined"
