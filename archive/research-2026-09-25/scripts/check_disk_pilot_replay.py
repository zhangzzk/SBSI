#!/usr/bin/env python
"""Require bitwise identical retained pilot arrays after inference-only caching."""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads((root / "result.json").read_text()) for root in (args.reference, args.replay)]
    for key in ("initial_center", "injected_shear", "model_sha256", "model_cache_sha256", "scene_sha256",
                "proposal_cache_sha256", "mock_input_sha256", "observation_partition", "config"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"pilot replay identity differs: {key}")
    for root, report in zip((args.reference, args.replay), reports):
        if file_hash(root / "one_step_moments.npz") != report["one_step_moments"]["sha256"]:
            raise ValueError("pilot moment hash mismatch")
    before = np.load(args.reference / "one_step_moments.npz")
    after = np.load(args.replay / "one_step_moments.npz")
    if set(before.files) != set(after.files):
        raise ValueError("pilot array names differ")
    for name in before.files:
        if before[name].dtype != after[name].dtype or before[name].shape != after[name].shape:
            raise ValueError(f"pilot layout differs: {name}")
        if before[name].tobytes() != after[name].tobytes():
            raise ValueError(f"pilot array not bitwise identical: {name}")
    result = dict(status="bitwise_identical", arrays=before.files,
        reference_result_sha256=file_hash(args.reference / "result.json"),
        replay_result_sha256=file_hash(args.replay / "result.json"),
        old_seconds=reports[0]["result"]["elapsed_seconds"], new_seconds=reports[1]["result"]["elapsed_seconds"])
    destination = args.replay / "replay_validation.json"
    if destination.exists():
        raise ValueError("refusing to overwrite replay validation")
    write_json(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
