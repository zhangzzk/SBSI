#!/usr/bin/env bash
# Recompute the first ConstGold recentering with measured m<24 instead of m<22.
# This reuses the completed pass-0 moments and the exact pass-1 normalization
# stencil, so only the expanded observation subset incurs GPU likelihood work.

#SBATCH --job-name=sbsi_cg_p1_m24
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1
base="$root/hybrid/constgold_plus100k_proxytop_k1024_m16384_iter0_4_exactnorm_bfgs_v1"
run="$root/diagnostics/pass1_mag24_v1"
mock="$root/input"
normalization="$base/pass1_selection_normalization.json"
threshold=1.3217558399823195

if [[ -e "$run" ]]; then
  echo "refusing to overwrite $run" >&2
  exit 2
fi
for required in \
  "$base/pass0_p0/result.json" "$base/pass0_p1/result.json" \
  "$base/pass0/result.json" "$base/pass1_hybrid.json" "$normalization"; do
  [[ -f "$required" ]] || { echo "missing prerequisite $required" >&2; exit 2; }
done
mkdir -p "$run"

export OMP_NUM_THREADS=8
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

centre=$(
  "$python" -c "import json; print('%.10f,%.10f' % tuple(json.load(open('$base/pass0/result.json'))['summary']['estimate']))"
)

"$python" - "$mock" "$run" <<'SUBSET'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

mock, run = Path(sys.argv[1]), Path(sys.argv[2])
mag = pd.read_parquet(
    mock / "measurements.parquet", columns=["measured_mag_auto"]
)["measured_mag_auto"].to_numpy()
rows = np.flatnonzero(mag < 24.0).astype(np.int64)
if rows.size != 53_431:
    raise SystemExit(f"expected 53,431 m<24 rows, found {rows.size}")
for gpu in range(2):
    np.save(run / f"mag24_p{gpu}.npy", rows[gpu::2])
print(f"m<24: {rows.size} of {mag.size} ({rows.size / mag.size:.4%}); "
      f"GPU split {[rows[gpu::2].size for gpu in range(2)]}")
SUBSET

common=(
  --inference-config "$repo/configs/inference_v1_2.json"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$source_root/compact_global/scene_store"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --model-cache "$source_root/compact_global/model_cache"
  --proposal-cache "$source_root/compact_global/proposal_cache"
  --mock-input "$mock"
  --blend-response-cache /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
  --cut-abs-ehat 0.6
  --cut-bound measured_mag_auto::25.8
  --cut-bound "measured_log_flux_radius:${threshold}:"
  --proposal-candidate-source whole_catalogue_gaussian_proxy
  --adaptive-draw-ladder 512 1024 2048 4096 8192 16384
  --selection-normalization-cache "$normalization"
  --allow-indefinite-partition-summary
  --initial-strategy fixed
  --initial "$centre"
  --device cuda
)

pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/run_inference.py" \
    "${common[@]}" \
    --object-subset "$run/mag24_p${gpu}.npy" \
    --output "$run/pass1_p${gpu}" \
    > "$run/pass1_p${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
(( status == 0 )) || exit 1

"$python" "$repo/scripts/combine_hybrid_pass.py" \
  --base "$base/pass0_p0" "$base/pass0_p1" \
  --recomputed "$run/pass1_p0" "$run/pass1_p1" \
  --score-root-bfgs \
  --output "$run/pass1_hybrid.json" \
  > "$run/pass1_combine.log" 2>&1

"$python" - "$base" "$run" <<'SUMMARY'
import json
import sys
from pathlib import Path

base, run = map(Path, sys.argv[1:])
old = json.loads((base / "pass1_hybrid.json").read_text())
new = json.loads((run / "pass1_hybrid.json").read_text())
payload = {
    "status": "complete",
    "diagnostic": "pass-1 recompute threshold comparison at the identical pass-0 centre",
    "center": new["hybrid"]["center"],
    "injected_shear": [0.02, 0.0],
    "m_lt_22": {
        "n_recomputed": old["n_recomputed"],
        "recomputed_fraction": old["recomputed_fraction"],
        "recomputed_information_share": old["recomputed_information_share"],
        "estimate": old["hybrid"]["estimate"],
        "robust_standard_error": old["hybrid"]["robust_standard_error"],
    },
    "m_lt_24": {
        "n_recomputed": new["n_recomputed"],
        "recomputed_fraction": new["recomputed_fraction"],
        "recomputed_information_share": new["recomputed_information_share"],
        "estimate": new["hybrid"]["estimate"],
        "robust_standard_error": new["hybrid"]["robust_standard_error"],
    },
    "m_lt_24_minus_m_lt_22": [
        float(a - b)
        for a, b in zip(new["hybrid"]["estimate"], old["hybrid"]["estimate"])
    ],
}
(run / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
SUMMARY
