#!/usr/bin/env bash
# Compare NLL-only and guard-selected full-domain flows at hard deployment cuts.

#SBATCH --job-name=sbsi_guard_hardcut
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/hard_cut_validation_c40x2_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/hard_cut_validation_c40x2_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1
output=$root/hard_cut_validation_c40x2_v1

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

mapfile -t train_cases < <(
  "$python" - "$root/domain/manifest.json" train 40 <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
key = "train_cases" if sys.argv[2] == "train" else "validation_cases"
for case in manifest["split"][key][: int(sys.argv[3])]:
    print(int(case))
PY
)
mapfile -t heldout_cases < <(
  "$python" - "$root/domain/manifest.json" heldout 40 <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
key = "train_cases" if sys.argv[2] == "train" else "validation_cases"
for case in manifest["split"][key][: int(sys.argv[3])]:
    print(int(case))
PY
)

for model in nll guard; do
  if [[ "$model" == nll ]]; then
    checkpoint=$root/flow/nll/selected.pt
  else
    checkpoint=$root/flow/guard/selected.pt
  fi
  for split in train heldout; do
    if [[ "$split" == train ]]; then
      selected=("${train_cases[@]}")
      seed=37301
    else
      selected=("${heldout_cases[@]}")
      seed=47301
    fi
    cases=()
    for case in "${selected[@]}"; do
      cases+=(--case "$case")
    done
    "$python" -u -m scripts.diagnose_flow_generated_radius_coupling \
      --domain-root "$root/domain" \
      --flow "$checkpoint" \
      "${cases[@]}" \
      --output "$output/${model}_${split}.json" \
      --draws 64 --sampling-seed "$seed" --batch-size 1024 --device cuda
  done
done

echo "FULL_DOMAIN_HARD_CUT_VALIDATION_COMPLETE output=$output"
