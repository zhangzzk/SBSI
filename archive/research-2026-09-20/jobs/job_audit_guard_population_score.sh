#!/usr/bin/env bash
#SBATCH --job-name=sbsi_guard_audit
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_population_audit_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_population_audit_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/guard_population_audit_20260917_v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$root/resources_${SLURM_ARRAY_TASK_ID}.json"
if [[ ${SLURM_ARRAY_TASK_ID} == 0 ]]; then
  flowroot=$cache/full_domain_true_mag26_guard_v1/flow
  flow=$flowroot/guard/selected.pt
  options=(--radius-softness 0.05 --magnitude-softness 0.05 --response-scale 0.02 --training-pairs 32768)
else
  flowroot=$cache/full_domain_true_mag26_guard_refine_v1/flow
  flow=$flowroot/selected.pt
  options=(--radius-softness 0.005 --magnitude-softness 0.01 --response-scale 0.01 --training-pairs 65536)
fi
"$python" -u -m scripts.audit_guard_population_score \
  --domain-root "$cache/full_domain_true_mag26_guard_v1/domain" \
  --flow "$flow" \
  --reference-indices "$flowroot/validation_guard_pair_indices.npy" \
  --output "$root/audit_${SLURM_ARRAY_TASK_ID}.json" "${options[@]}"
