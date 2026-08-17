#!/usr/bin/env python3
"""Extract pair-excluded near/mid/far fluxes in the V2.2 OOF row order.

The 200-case ``other3abs`` response catalogue preserves the original response
catalogue row order and adds three intrinsic-scene coordinates.  This script
reapplies the frozen V2.2 regression support, checks every selected row against
the immutable OOF cache, and writes only the three added float32 columns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from blendemu import data_utils  # noqa: E402
from blendemu.scene_features import (  # noqa: E402
    PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS,
)
from scripts.v22_grouped_rscene_common import sha256  # noqa: E402


RAW_FEATURES = [
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
]
NEEDED = [
    "case",
    "delta_et1",
    *RAW_FEATURES,
    *PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS,
]


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--augmented-catalogue", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    catalogue = Path(args.augmented_catalogue).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary path already exists: {temporary}")
    temporary.mkdir()

    try:
        source_metadata_path = source_cache / "metadata.json"
        with source_metadata_path.open(encoding="utf-8") as handle:
            source_metadata = json.load(handle)
        if source_metadata["case_window"] != [0, 199]:
            raise RuntimeError("source cache is not the complete 200-case cache")
        if source_metadata["raw_features"] != RAW_FEATURES:
            raise RuntimeError("source-cache raw feature order has drifted")
        n_rows = int(source_metadata["n_rows"])
        source_case = np.load(
            source_cache / source_metadata["arrays"]["case"], mmap_mode="r"
        )
        source_label = np.load(
            source_cache / source_metadata["arrays"]["label"], mmap_mode="r"
        )
        source_raw = np.load(
            source_cache / source_metadata["arrays"]["x_raw"], mmap_mode="r"
        )
        if source_raw.shape != (n_rows, len(RAW_FEATURES)):
            raise RuntimeError("source raw-feature shape has drifted")

        with Path(source_metadata["source_metadata"]).open(
            encoding="utf-8"
        ) as handle:
            emulator_metadata = json.load(handle)
        cuts = emulator_metadata["tasks"]["regression"]["cuts"]
        shear = float(source_metadata["shear"])
        if not np.isclose(shear, 0.2, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"unexpected response shear {shear}")

        feature_path = temporary / "x_blendness.npy"
        output_features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, len(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS)),
        )
        cursor = 0
        case_counts = np.zeros(200, dtype=np.int64)
        feature_sum = np.zeros(3, dtype=np.float64)
        feature_square_sum = np.zeros(3, dtype=np.float64)
        feature_min = np.full(3, np.inf, dtype=np.float64)
        feature_max = np.full(3, -np.inf, dtype=np.float64)

        with ipc.open_file(catalogue) as reader:
            missing = set(NEEDED) - set(reader.schema.names)
            if missing:
                raise KeyError(f"augmented catalogue lacks {sorted(missing)}")
            for batch_index in range(reader.num_record_batches):
                frame = (
                    pa.Table.from_batches([reader.get_batch(batch_index)])
                    .select(NEEDED)
                    .to_pandas()
                )
                case = frame.case.to_numpy(np.int64, copy=False)
                frame = frame.loc[(case >= 0) & (case <= 199)]
                if frame.empty:
                    continue
                frame = data_utils.source_select_reg(frame, cuts=cuts)
                if frame.empty:
                    continue
                n_batch = len(frame)
                stop = cursor + n_batch
                if stop > n_rows:
                    raise RuntimeError("augmented selection exceeds source cache")

                local_case = frame.case.to_numpy(np.int16, copy=False)
                local_raw = frame[RAW_FEATURES].to_numpy(np.float32, copy=False)
                local_label = (
                    frame.delta_et1.to_numpy(np.float64, copy=False) / shear
                ).astype(np.float32)
                if not np.array_equal(local_case, source_case[cursor:stop]):
                    raise RuntimeError(
                        f"case row-order mismatch at selected rows {cursor}:{stop}"
                    )
                if not np.array_equal(local_raw, source_raw[cursor:stop]):
                    raise RuntimeError(
                        f"raw-feature row-order mismatch at rows {cursor}:{stop}"
                    )
                if not np.array_equal(local_label, source_label[cursor:stop]):
                    raise RuntimeError(
                        f"label row-order mismatch at rows {cursor}:{stop}"
                    )

                values = frame[list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS)].to_numpy(
                    np.float32, copy=False
                )
                if not np.isfinite(values).all() or np.any(values < 0.0):
                    raise RuntimeError("blendness features are non-finite or negative")
                output_features[cursor:stop] = values
                values64 = values.astype(np.float64)
                feature_sum += values64.sum(axis=0)
                feature_square_sum += np.square(values64).sum(axis=0)
                feature_min = np.minimum(feature_min, values64.min(axis=0))
                feature_max = np.maximum(feature_max, values64.max(axis=0))
                case_counts += np.bincount(local_case, minlength=200)
                cursor = stop
                if (batch_index + 1) % 100 == 0:
                    print(
                        f"batch {batch_index + 1}/{reader.num_record_batches}: "
                        f"aligned={cursor:,}/{n_rows:,}",
                        flush=True,
                    )

        if cursor != n_rows:
            raise RuntimeError(f"filled {cursor:,} rows; expected {n_rows:,}")
        expected_counts = np.asarray([
            int(source_metadata["case_counts"][str(case)])
            for case in range(200)
        ])
        if not np.array_equal(case_counts, expected_counts):
            raise RuntimeError("per-case selected counts differ from source cache")
        output_features.flush()
        del output_features

        mean = feature_sum / n_rows
        variance = np.maximum(feature_square_sum / n_rows - np.square(mean), 0.0)
        augmented_sidecar = catalogue.with_suffix(".json")
        payload = {
            "schema_version": 1,
            "kind": "pair-excluded blendness features in V2.2 OOF row order",
            "n_rows": n_rows,
            "shape": [n_rows, 3],
            "dtype": "float32",
            "features": list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS),
            "feature_definition": (
                "log10(1 + absolute intrinsic flux in simulation-count units "
                "after excluding the primary and the designated secondary, in "
                "0--1, 1--3, and 3--10 arcsec shells"
            ),
            "feature_summary": {
                "mean": mean.tolist(),
                "std": np.sqrt(variance).tolist(),
                "min": feature_min.tolist(),
                "max": feature_max.tolist(),
            },
            "array": "x_blendness.npy",
            "source_cache": str(source_cache),
            "source_metadata": str(source_metadata_path),
            "source_metadata_sha256": sha256(source_metadata_path),
            "augmented_catalogue": str(catalogue),
            "augmented_catalogue_size_bytes": int(catalogue.stat().st_size),
            "augmented_sidecar": str(augmented_sidecar),
            "augmented_sidecar_sha256": (
                sha256(augmented_sidecar) if augmented_sidecar.exists() else None
            ),
            "row_alignment_checks": [
                "case exact",
                "seven raw pair features exact",
                "physical response label exact",
                "per-case selected row counts exact",
            ],
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        }
        strict_json(temporary / "metadata.json", payload)
        os.replace(temporary, output)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        print("V22_OTHER3ABS_CROSSFIT_CACHE_DONE", flush=True)
    except Exception:
        print(f"failed cache remains for inspection at {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
