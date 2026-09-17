#!/usr/bin/env bash
# Evaluate the no-radius-cut control on 40 train and all 40 held-out cases.

#SBATCH --job-name=sbsi_norcut_eval
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_no_radius_cut_v1/radius_test_c40x2_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_no_radius_cut_v1/radius_test_c40x2_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_no_radius_cut_v1
output=$root/radius_test_c40x2_v1

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

for split in train heldout; do
  if [[ "$split" == train ]]; then
    selected=("${train_cases[@]}")
    seed=17301
  else
    selected=("${heldout_cases[@]}")
    seed=27301
  fi
  cases=()
  for case in "${selected[@]}"; do
    cases+=(--case "$case")
  done
  "$python" -u -m scripts.diagnose_flow_generated_radius_coupling \
    --domain-root "$root/domain" \
    --flow "$root/flow/paired/selected.pt" \
    "${cases[@]}" \
    --output "$output/${split}.json" \
    --draws 64 --sampling-seed "$seed" --batch-size 1024 --device cuda
done

echo "NO_RADIUS_CUT_EVALUATION_COMPLETE output=$output"
