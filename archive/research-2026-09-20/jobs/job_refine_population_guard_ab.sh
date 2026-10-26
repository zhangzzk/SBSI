#!/usr/bin/env bash
# A/B isolates shared versus independent response-row replicas.
#SBATCH --job-name=sbsi_popguard_ab
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/population_guard_ab_20260917_v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
options=()
if [[ ${SLURM_ARRAY_TASK_ID} == 0 ]]; then
  arm=shared
  options+=(--shared-row-control)
else
  arm=independent
fi
"$python" -u -m scripts.refine_full_domain_guard_flow \
  --domain-root "$cache/full_domain_true_mag26_guard_v1/domain" \
  --initial-flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --output-root "$root/$arm" --population-objective --wide-guards \
  --epochs 12 --learning-rate 0.00001 --guard-steps 8 "${options[@]}"
