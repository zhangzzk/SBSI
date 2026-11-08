#!/usr/bin/env bash
# Freeze the matched 20,000-object complete-likelihood mock for the direct
# whole-catalogue-proxy shortlist experiment.  The measured size target is the
# natural log of FLUX_RADIUS in pixels, so R >= 0.75 arcsec at 0.2 arcsec/pixel
# is log(3.75) = 1.3217558399823195.

#SBATCH --job-name=sbsi_prep_size075_20k
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1
threshold=1.3217558399823195

export PREP_ROOT="$root"
export PREP_OUTPUT="$root/prepare_g002"
export SELECTION_CACHE="$root/selection_cache"
export N_DETECTED=20000
export CUT_BOUND_EXTRA="measured_log_flux_radius:${threshold}:"

bash "$repo/jobs/job_prepare_complete_likelihood_mock.sh"

"$python" - "$PREP_OUTPUT" "$threshold" <<'CHECK'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

output = Path(sys.argv[1])
threshold = float(sys.argv[2])
mock = output / "mock"
manifest = json.loads((mock / "likelihood_mock_manifest.json").read_text())
measurements = pd.read_parquet(mock / "measurements.parquet")
expected = (
    "|xhat|<0.6;measured_mag_auto:None:25.8;"
    f"measured_log_flux_radius:{threshold!r}:None"
)
actual = manifest["generation_identity"]["selection_cut_key"]
if actual != expected:
    raise SystemExit(f"cut mismatch: {actual!r} != {expected!r}")
shape = np.hypot(
    measurements["measured_ngmix_g1"].to_numpy(float),
    measurements["measured_ngmix_g2"].to_numpy(float),
)
passed = (
    (shape < 0.6)
    & (measurements["measured_mag_auto"].to_numpy(float) < 25.8)
    & (measurements["measured_log_flux_radius"].to_numpy(float) >= threshold)
)
if len(measurements) != 20000 or not passed.all():
    raise SystemExit(
        f"stored mock failed selection: n={len(measurements)}, "
        f"passing={int(passed.sum())}"
    )
print(
    "validated size-cut mock:",
    len(measurements),
    "rows; minimum measured radius arcsec =",
    float(0.2 * np.exp(measurements["measured_log_flux_radius"].min())),
)
CHECK
