#!/bin/bash
#SBATCH --job-name=sbsi-auxiliary-probe
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_auxiliary_probe_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_auxiliary_probe_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests/test_disk_auxiliary_probe.py -q
"$PYTHON" scripts/probe_disk_auxiliary.py --run "$RUN_ROOT" \
  --proxy "${PROXY:-gaussian_aux}" \
  --degrees-of-freedom "${STUDENT_DF:-3}" \
  --exact "$RUN_ROOT/curvature_probe_16617682_exact/report.json" \
  --output "$RUN_ROOT/auxiliary_probe_${SLURM_JOB_ID}"
