#!/usr/bin/env bash
# Build the complete nine-view measured-selection normalization stencil at the
# canonical shear origin.  Active prior atoms are split over every allocated
# GPU; no observation likelihood is evaluated.

#SBATCH --job-name=sbsi_norm_g0_size075
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1
output=${NORMALIZATION_OUTPUT:-$root/selection_normalization_g0_size075_v1}
threshold=1.3217558399823195
compact=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$output" "$root/logs"

if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  n_gpus=$SLURM_GPUS_ON_NODE
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  n_gpus=${#visible_gpus[@]}
else
  n_gpus=$($python -c 'import torch; print(torch.cuda.device_count())')
fi
if ! [[ "$n_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "no allocated GPU found" >&2
  exit 2
fi
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / n_gpus ))
if (( OMP_NUM_THREADS < 1 )); then export OMP_NUM_THREADS=1; fi
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

common=(
  --inference-config "$repo/configs/inference_v1_2.json"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$compact/scene_store"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --model-cache "$compact/model_cache"
  --proposal-cache "$compact/proposal_cache"
  --mock-input "$root/prepare_g002/mock"
  --blend-response-cache /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
  --cut-abs-ehat 0.6
  --cut-bound measured_mag_auto::25.8
  --cut-bound "measured_log_flux_radius:${threshold}:"
  --initial-strategy fixed
  --initial 0,0
  --device cuda
)

pids=()
for ((gpu = 0; gpu < n_gpus; gpu++)); do
  shard="$output/shard_p$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/run_inference.py" \
    "${common[@]}" \
    --selection-normalization-shard-index "$gpu" \
    --selection-normalization-shards "$n_gpus" \
    --output "$shard" > "$output/shard_p$gpu.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
  echo "one or more normalization shards failed" >&2
  exit 1
fi

combine_args=()
for ((gpu = 0; gpu < n_gpus; gpu++)); do
  combine_args+=(--shard "$output/shard_p$gpu")
done
exact="$output/selection_normalization_exact_g0.json"
"$python" "$repo/scripts/combine_selection_normalization_shards.py" \
  "${combine_args[@]}" --output "$exact" > "$output/combine.log" 2>&1

"$python" - "$exact" <<'REPORT'
import json
import math
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
h = float(payload["finite_difference_step"])
points = {
    (round(float(row["g1"]), 10), round(float(row["g2"]), 10)):
    math.log(float(row["detected_and_selected_mass"]))
    for row in payload["points"]
}

def value(g1, g2):
    return points[(round(g1, 10), round(g2, 10))]

zero = value(0.0, 0.0)
gradient = np.array([
    (value(h, 0.0) - value(-h, 0.0)) / (2.0 * h),
    (value(0.0, h) - value(0.0, -h)) / (2.0 * h),
])
hessian = np.array([
    [
        (value(h, 0.0) - 2.0 * zero + value(-h, 0.0)) / h**2,
        (value(h, h) - value(h, -h) - value(-h, h) + value(-h, -h)) / (4.0 * h**2),
    ],
    [0.0, (value(0.0, h) - 2.0 * zero + value(0.0, -h)) / h**2],
])
hessian[1, 0] = hessian[0, 1]
report = {
    "center": [0.0, 0.0],
    "finite_difference_step": h,
    "log_mass_at_center": zero,
    "gradient": gradient.tolist(),
    "hessian": hessian.tolist(),
    "exact_cache": str(path.resolve()),
}
(path.parent / "derivatives.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
REPORT

echo "done: $exact"
