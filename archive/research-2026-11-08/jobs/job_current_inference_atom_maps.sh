#!/bin/bash
#SBATCH --job-name=sbsi-atom-maps
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/current_atom_maps_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-atoms-pycache-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/sbsi-atoms-mpl-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
SBSI_PLOT_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI_PLOT_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_atom_maps_20260925_v1
"$SBSI_PLOT_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$SBSI_PLOT_ROOT/resources_${SLURM_JOB_ID}.json"
"$SBSI_PLOT_PYTHON" "$SBSI_PLOT_ROOT/plot_current_inference_atoms.py" \
  --runtime-root "$SBSI_PLOT_ROOT/runtime" \
  --run /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1 \
  --output "$SBSI_PLOT_ROOT/results_${SLURM_JOB_ID}"
