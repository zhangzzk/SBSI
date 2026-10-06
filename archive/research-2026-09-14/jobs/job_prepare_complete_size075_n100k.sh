#!/usr/bin/env bash
# Generate the 100,000-object extension of the measured-size-cut complete-
# likelihood mock and verify that it preserves the existing 20k prefix.

#SBATCH --job-name=sbsi_prep_size075_100k
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
old_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1
threshold=1.3217558399823195

export PREP_ROOT="$root"
export PREP_OUTPUT="$root/prepare_g002"
export SELECTION_CACHE="$root/selection_cache"
export N_DETECTED=100000
export CUT_BOUND_EXTRA="measured_log_flux_radius:${threshold}:"

bash "$repo/jobs/job_prepare_complete_likelihood_mock.sh"

"$python" - "$PREP_OUTPUT" "$old_root/prepare_g002" "$threshold" <<'CHECK'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

output, old_output = Path(sys.argv[1]), Path(sys.argv[2])
threshold = float(sys.argv[3])
mock, old_mock = output / "mock", old_output / "mock"
manifest = json.loads((mock / "likelihood_mock_manifest.json").read_text())
measurements = pd.read_parquet(mock / "measurements.parquet")
truth = pd.read_parquet(mock / "truth.parquet")
old_measurements = pd.read_parquet(old_mock / "measurements.parquet")
old_truth = pd.read_parquet(old_mock / "truth.parquet")
expected = (
    "|xhat|<0.6;measured_mag_auto:None:25.8;"
    f"measured_log_flux_radius:{threshold!r}:None"
)
if manifest["generation_identity"]["selection_cut_key"] != expected:
    raise SystemExit("100k mock has the wrong selection identity")
shape = np.hypot(
    measurements["measured_ngmix_g1"].to_numpy(float),
    measurements["measured_ngmix_g2"].to_numpy(float),
)
passed = (
    (shape < 0.6)
    & (measurements["measured_mag_auto"].to_numpy(float) < 25.8)
    & (measurements["measured_log_flux_radius"].to_numpy(float) >= threshold)
)
if len(measurements) != 100000 or len(truth) != 100000 or not passed.all():
    raise SystemExit(
        f"stored mock failed selection: measurements={len(measurements)}, "
        f"truth={len(truth)}, passing={int(passed.sum())}"
    )
# GPU flow evaluation is batched differently at 20k and 100k, so measured
# float32 values need not be bit-identical.  The truth/random stream must be
# exact, and measurement differences are limited to a strict numerical bound.
new_prefix = measurements.iloc[: len(old_measurements)].reset_index(drop=True)
old_prefix = old_measurements.reset_index(drop=True)
for column in old_prefix.columns:
    difference = np.abs(
        new_prefix[column].to_numpy(float) - old_prefix[column].to_numpy(float)
    )
    if not np.isfinite(difference).all() or float(difference.max()) > 5e-6:
        raise SystemExit(
            f"20k measurement prefix changed materially in {column}: "
            f"max abs difference {float(difference.max())}"
        )
pd.testing.assert_frame_equal(
    truth.iloc[: len(old_truth)].reset_index(drop=True),
    old_truth.reset_index(drop=True),
    check_exact=True,
)
print(
    "validated 100k size-cut mock, exact truth prefix, and measurement prefix "
    "within 5e-6; "
    "minimum measured radius arcsec =",
    float(0.2 * np.exp(measurements["measured_log_flux_radius"].min())),
)
CHECK
