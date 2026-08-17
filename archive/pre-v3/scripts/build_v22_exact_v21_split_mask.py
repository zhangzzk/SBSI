#!/usr/bin/env python3
"""Build the exact float64 V2.1 mask on the frozen V2.2 fitting rows.

The OOF cache intentionally stores raw features as float32.  That is exact for
model inference but can move a handful of rows across the strict V2.1 size/SN
boundary relative to the older calibration plot, which cut the original
float64 Feather columns.  This script streams the original catalogue with the
same V2.2 regression selection, verifies row-for-row alignment to the cache,
and writes a boolean mask for cases 40--199 using the original doubles.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

from scripts.plot_positive_weighted_pair_residual_validation import sha256
from scripts.prepare_v22_oof_learning_cache import RAW_FEATURES, selected_batch
from sbs_shear.domain import in_domain, metadata as v21_domain_metadata, sn_true


FIT_CASE_MIN = 40
FIT_CASE_MAX = 199
MASK_NAME = "fit_v21_primary_domain.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--old-all-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cache = Path(args.cache).resolve()
    reference_path = Path(args.old_all_reference).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exact-mask directory {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary exact-mask directory exists: {temporary}")
    temporary.mkdir()

    try:
        metadata_path = cache / "metadata.json"
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        with reference_path.open(encoding="utf-8") as handle:
            old_reference = json.load(handle)
        if metadata["fit_case_window"] != [FIT_CASE_MIN, FIT_CASE_MAX]:
            raise RuntimeError("cache fitting cases are not 40--199")
        if old_reference.get("case_window") != [FIT_CASE_MIN, FIT_CASE_MAX]:
            raise RuntimeError("old all-row reference cases are not 40--199")
        if old_reference.get("v21_primary_domain") is not True:
            raise RuntimeError("old all-row reference did not apply V2.1")

        source = Path(metadata["source_catalogue"])
        if not source.samefile(Path(old_reference["catalogue"])):
            raise RuntimeError("cache and old plot use different source catalogues")
        with Path(metadata["source_metadata"]).open(encoding="utf-8") as handle:
            source_metadata = json.load(handle)
        cuts = source_metadata["tasks"]["regression"]["cuts"]

        arrays = metadata["arrays"]
        case_all = np.load(cache / arrays["case"], mmap_mode="r")
        input_all = np.load(cache / arrays["input_index"], mmap_mode="r")
        raw_all = np.load(cache / arrays["x_raw"], mmap_mode="r")
        official_all = np.load(cache / arrays["official_train"], mmap_mode="r")
        start = int(np.searchsorted(case_all, FIT_CASE_MIN, side="left"))
        stop = int(np.searchsorted(case_all, FIT_CASE_MAX, side="right"))
        n_fit = stop - start
        expected_fit = int(metadata["n_fit_rows"])
        if n_fit != expected_fit:
            raise RuntimeError("cache fitting-window length disagrees with metadata")

        exact_mask = np.lib.format.open_memmap(
            temporary / MASK_NAME,
            mode="w+",
            dtype=np.bool_,
            shape=(n_fit,),
        )
        cursor = 0
        exact_count = 0
        float32_count = 0
        false_negative = 0
        false_positive = 0
        boundary_rows: list[dict[str, object]] = []

        with ipc.open_file(source) as reader:
            for batch_index in range(reader.num_record_batches):
                frame = selected_batch(reader, batch_index, cuts)
                if frame.empty:
                    continue
                frame = frame.loc[frame.case.between(FIT_CASE_MIN, FIT_CASE_MAX)]
                if frame.empty:
                    continue
                n_batch = len(frame)
                local_start = start + cursor
                local_stop = local_start + n_batch
                if local_stop > stop:
                    raise RuntimeError("source selection exceeds cached fitting window")

                source_case = frame.case.to_numpy(np.int16)
                source_input = frame.input_index.to_numpy(np.int64)
                if not np.array_equal(source_case, case_all[local_start:local_stop]):
                    raise RuntimeError(f"case alignment failed at source batch {batch_index}")
                if not np.array_equal(source_input, input_all[local_start:local_stop]):
                    raise RuntimeError(f"input-index alignment failed at source batch {batch_index}")
                source_raw32 = frame[RAW_FEATURES].to_numpy(np.float32)
                cached_raw = np.asarray(raw_all[local_start:local_stop])
                if not np.array_equal(source_raw32, cached_raw):
                    raise RuntimeError(f"raw-feature alignment failed at source batch {batch_index}")

                mag64 = frame.r_input_p.to_numpy(np.float64)
                re64 = frame.Re_input_p.to_numpy(np.float64)
                exact = in_domain(mag64, re64)
                mag32 = cached_raw[:, RAW_FEATURES.index("r_input_p")]
                re32 = cached_raw[:, RAW_FEATURES.index("Re_input_p")]
                approximate = in_domain(mag32, re32)
                mismatch = exact != approximate
                if mismatch.any():
                    local_mismatch = np.flatnonzero(mismatch)
                    for local in local_mismatch:
                        boundary_rows.append({
                            "case": int(source_case[local]),
                            "input_index": int(source_input[local]),
                            "r_input_p_float64": float(mag64[local]),
                            "Re_input_p_float64": float(re64[local]),
                            "sn_true_float64": float(sn_true(mag64[local], re64[local])),
                            "r_input_p_float32": float(mag32[local]),
                            "Re_input_p_float32": float(re32[local]),
                            "sn_true_float32": float(sn_true(mag32[local], re32[local])),
                            "exact_float64_keep": bool(exact[local]),
                            "cached_float32_keep": bool(approximate[local]),
                            "official_split": (
                                "training"
                                if bool(official_all[local_start + local])
                                else "validation"
                            ),
                        })
                    false_negative += int(np.count_nonzero(exact & ~approximate))
                    false_positive += int(np.count_nonzero(~exact & approximate))

                exact_mask[cursor:cursor + n_batch] = exact
                exact_count += int(exact.sum())
                float32_count += int(approximate.sum())
                cursor += n_batch
                if (batch_index + 1) % 250 == 0:
                    print(
                        f"batch {batch_index + 1}/{reader.num_record_batches}: "
                        f"aligned fitting rows={cursor:,}",
                        flush=True,
                    )

        if cursor != n_fit:
            raise RuntimeError(f"aligned {cursor:,} fitting rows; expected {n_fit:,}")
        exact_mask.flush()
        old_count = int(old_reference["n_pairs"])
        if exact_count != old_count:
            raise RuntimeError(
                f"exact mask retains {exact_count:,}; old plot has {old_count:,}"
            )
        if exact_count - float32_count != false_negative - false_positive:
            raise RuntimeError("float64/float32 mismatch accounting does not close")

        official_fit = np.asarray(official_all[start:stop], dtype=bool)
        training_count = int(np.count_nonzero(exact_mask & official_fit))
        validation_count = int(np.count_nonzero(exact_mask & ~official_fit))
        if training_count + validation_count != exact_count:
            raise RuntimeError("exact training/validation partition does not close")

        payload = {
            "schema_version": 1,
            "source_catalogue": str(source),
            "source_catalogue_size_bytes": int(source.stat().st_size),
            "cache": str(cache),
            "cache_metadata": str(metadata_path),
            "cache_metadata_sha256": sha256(metadata_path),
            "old_all_reference": str(reference_path),
            "old_all_reference_sha256": sha256(reference_path),
            "fit_case_window": [FIT_CASE_MIN, FIT_CASE_MAX],
            "n_fit_rows_before_domain": int(n_fit),
            "n_exact_v21_rows": int(exact_count),
            "n_exact_training_rows": training_count,
            "n_exact_validation_rows": validation_count,
            "n_cached_float32_v21_rows": int(float32_count),
            "float64_minus_float32_count": int(exact_count - float32_count),
            "float32_false_negative_count": int(false_negative),
            "float32_false_positive_count": int(false_positive),
            "boundary_mismatch_rows": boundary_rows,
            "mask": MASK_NAME,
            "mask_alignment": (
                "row-for-row with cache cases 40--199 after the exact frozen V2.2 regression cuts"
            ),
            "domain": v21_domain_metadata(),
            "primary_only": True,
            "secondary_domain_cut": False,
            "truth_precision": "original Feather float64",
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        }
        with (temporary / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
        print("EXACT_V21_MASK_DONE", flush=True)
    except Exception:
        print(f"failed exact-mask build remains at {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
