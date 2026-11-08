#!/usr/bin/env python
"""Compare matched complete-likelihood pass-0 screens for two measurement flows."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(root):
    result_path = root / "hybrid" / "constgold_plus20k_k1024_m8192_exactnorm_pass0" / "pass0" / "result.json"
    payload = json.loads(result_path.read_text())
    estimate = payload["summary"]["estimate"]
    truth = payload["injected_shear"]
    return {
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "measurement_model_sha256": payload["identity"]["model_sha256"]["measurement"],
        "detection_model_sha256": payload["identity"]["likelihood_component_sha256"]["detection"],
        "measurement_sha256": payload["identity"]["mock_input_sha256"]["measurements.parquet"],
        "truth_sha256": payload["identity"]["mock_input_sha256"]["truth.parquet"],
        "n_observations": payload["n_observations"],
        "center": payload["summary"]["center"],
        "estimate": estimate,
        "injected_shear": truth,
        "bias": [estimate[index] - truth[index] for index in range(2)],
        "robust_standard_error": payload["summary"]["robust_standard_error"],
        "information_eigenvalues": payload["summary"]["information_eigenvalues"],
        "ladder": {
            draws: {
                "estimate": row["estimate"],
                "robust_standard_error": row["robust_standard_error"],
            }
            for draws, row in payload["full_ladder"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    arms = {
        name: _load_arm(args.root / name)
        for name in ("original_transition", "selected_retrain_transition")
    }
    original = arms["original_transition"]
    selected = arms["selected_retrain_transition"]
    for key in ("detection_model_sha256", "measurement_sha256", "truth_sha256", "n_observations", "center", "injected_shear"):
        if selected[key] != original[key]:
            raise ValueError(f"matched-screen identity differs for {key}")
    result = {
        "purpose": "matched 20k complete-likelihood flow gate with common transition detector",
        "arms": arms,
        "selected_minus_original_estimate": [
            selected["estimate"][index] - original["estimate"][index]
            for index in range(2)
        ],
        "selected_reduces_absolute_bias": [
            abs(selected["bias"][index]) < abs(original["bias"][index])
            for index in range(2)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
