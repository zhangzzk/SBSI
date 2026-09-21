#!/usr/bin/env bash
# Validate the joint hard radius/magnitude cut on all held-out cases.

#SBATCH --job-name=sbsi_guard_joint
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/hard_cut_joint_validation_c40_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/hard_cut_joint_validation_c40_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1
output=$root/hard_cut_joint_validation_c40_v1

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

mapfile -t heldout_cases < <(
  "$python" - "$root/domain/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
for case in manifest["split"]["validation_cases"]:
    print(int(case))
PY
)
cases=()
for case in "${heldout_cases[@]}"; do
  cases+=(--case "$case")
done

for model in nll guard; do
  if [[ "$model" == nll ]]; then
    checkpoint=$root/flow/nll/selected.pt
  else
    checkpoint=$root/flow/guard/selected.pt
  fi
  "$python" -u -m scripts.diagnose_flow_generated_radius_coupling \
    --domain-root "$root/domain" \
    --flow "$checkpoint" \
    "${cases[@]}" \
    --output "$output/${model}_heldout.json" \
    --draws 64 --sampling-seed 57301 --batch-size 1024 --device cuda
done

echo "FULL_DOMAIN_JOINT_HARD_CUT_VALIDATION_COMPLETE output=$output"
