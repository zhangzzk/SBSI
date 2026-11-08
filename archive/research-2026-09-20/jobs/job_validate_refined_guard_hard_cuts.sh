#!/usr/bin/env bash
# Validate the sharp-guard refinement at the hard deployment cuts on all held-out cases.

#SBATCH --job-name=sbsi_refine_hardcut
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/hard_cut_validation_c40_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/hard_cut_validation_c40_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
refine_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1
output=$refine_root/hard_cut_validation_c40_v1

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

mapfile -t heldout_cases < <(
  "$python" - "$domain_root/manifest.json" <<'PY'
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

"$python" -u -m scripts.diagnose_flow_generated_radius_coupling \
  --domain-root "$domain_root" \
  --flow "$refine_root/flow/selected.pt" \
  "${cases[@]}" \
  --output "$output/refined_heldout.json" \
  --draws 64 --sampling-seed 57301 --batch-size 1024 --device cuda

echo "REFINED_GUARD_HARD_CUT_VALIDATION_COMPLETE output=$output"
