#!/bin/bash
#SBATCH --job-name=sbsi-joint-plots
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v36_joint_plots_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
# Avoid reusing stale shared-filesystem bytecode after an immediate resubmit.
export PYTHONPYCACHEPREFIX
PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/sbsi_joint_plot_pycache.XXXXXX)
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_joint_plot_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests/test_disk_joint_likelihood_plot.py -q
"$PYTHON" scripts/plot_disk_joint_likelihood.py --run "$RUN_ROOT" \
  --diagnostics "$RUN_ROOT/likelihood_probe_16621068" \
  --output "/home/z/Zekang.Zhang/SBSI/doc/figures/v36_joint_likelihood_${SLURM_JOB_ID}" --view local
"$PYTHON" scripts/plot_disk_joint_likelihood.py --run "$RUN_ROOT" \
  --diagnostics "$RUN_ROOT/likelihood_probe_16621068" \
  --render-only /home/z/Zekang.Zhang/SBSI/doc/figures/v36_joint_likelihood_16621509 \
  --output "/home/z/Zekang.Zhang/SBSI/doc/figures/v36_joint_likelihood_${SLURM_JOB_ID}/overview"
