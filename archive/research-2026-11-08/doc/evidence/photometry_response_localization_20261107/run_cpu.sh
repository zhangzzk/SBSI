#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=photo_response_check
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_response_localization_20261107/logs/cpu_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py -o /project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_response_localization_20261107/resources.json
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/photometry_response_localization_20261107/pipeline.py "${1:?stage}"
