#!/usr/bin/env python
"""Freeze a selected ConstGold sample as a provenance-checked image input."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_NAMES = [
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--likelihood-config", type=Path, required=True)
    parser.add_argument("--injected-g1", type=float, default=0.02)
    parser.add_argument("--injected-g2", type=float, default=0.0)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    source = pd.read_parquet(args.source_sample)
    required = {"case", "input_index", *TARGET_NAMES}
    missing = sorted(required - set(source))
    if missing:
        raise KeyError(f"ConstGold sample lacks columns: {missing}")
    if source[["case", "input_index"]].duplicated().any():
        raise ValueError("ConstGold sample contains duplicate source identities")
    measurements = source[TARGET_NAMES].copy()
    values = measurements.to_numpy(float)
    radius = args.pixel_size * np.exp(values[:, 3])
    selected = (
        np.isfinite(values).all(axis=1)
        & (np.hypot(values[:, 0], values[:, 1]) < 0.6)
        & (values[:, 2] < 25.8)
        & (radius >= 0.75)
    )
    if not selected.all():
        raise ValueError(f"source sample has {int((~selected).sum())} rows outside cuts")

    likelihood = json.loads(args.likelihood_config.read_text())
    shear_transform = likelihood["shear_transform"]
    expected_model_hash = likelihood["measurement_model"]["sha256"]
    actual_model_hash = _sha256(args.measurement_model)
    if actual_model_hash != expected_model_hash:
        raise ValueError(
            "measurement model does not match the likelihood configuration: "
            f"expected {expected_model_hash}, found {actual_model_hash}"
        )
    truth = pd.DataFrame(
        {
            "mock_kind": "image",
            "shear_transform": shear_transform,
            "injected_g1": float(args.injected_g1),
            "injected_g2": float(args.injected_g2),
            "source_case": source["case"].to_numpy(np.int64),
            "source_input_index": source["input_index"].to_numpy(np.int64),
        }
    )
    args.output.mkdir(parents=True)
    measurements.to_parquet(args.output / "measurements.parquet", index=False)
    truth.to_parquet(args.output / "truth.parquet", index=False)
    output_hashes = {
        name: _sha256(args.output / name)
        for name in ("measurements.parquet", "truth.parquet")
    }
    manifest = {
        "mock_kind": "image",
        "catalogue": "constgold",
        "leg": "plus",
        "shear_transform": shear_transform,
        "measurement_model": str(args.measurement_model),
        "measurement_model_sha256": actual_model_hash,
        "target_names": TARGET_NAMES,
        "injected_g1": float(args.injected_g1),
        "injected_g2": float(args.injected_g2),
        "n_objects": int(len(measurements)),
        "source_case_range": [int(source["case"].min()), int(source["case"].max()) + 1],
        "source_cases_present": int(source["case"].nunique()),
        "source_sample": str(args.source_sample),
        "source_sample_sha256": _sha256(args.source_sample),
        "selection_cut_key": (
            "|xhat|<0.6;measured_mag_auto:None:25.8;"
            "measured_log_flux_radius:1.3217558399823195:None"
        ),
        "output_sha256": output_hashes,
    }
    (args.output / "image_mock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
