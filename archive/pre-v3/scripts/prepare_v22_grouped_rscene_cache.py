#!/usr/bin/env python3
"""Attach frozen full-scene coordinates to every cached V2.2 pair row.

No response label is used.  The scene sum and multiplicity are reconstructed
from the complete deployed pair list in the immutable V2.2 cache.  Quantile
edges are fixed from cases 40--159, before candidate models see cases 160--199
or external cases 0--39.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v22_grouped_rscene_common import (
    CASE_MAX,
    CONDITIONAL_FEATURES,
    DEFAULT_FORCED_EDGES,
    DEFAULT_N_SCENE_BINS,
    EXTERNAL_DEVELOPMENT_CASE_MAX,
    EXTERNAL_FINAL_CASE_MAX,
    EXTERNAL_FINAL_CASE_MIN,
    FIT_CASE_MIN,
    INTERNAL_VALIDATION_CASE_MIN,
    SOURCE_TAG,
    TRAIN_CASE_MAX,
    assign_scene_bins,
    case_offsets,
    load_source_metadata,
    make_scene_edges,
    mmap_array,
    sha256,
    strict_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-scene-bins", type=int, default=DEFAULT_N_SCENE_BINS)
    parser.add_argument("--forced-edge", type=float, action="append")
    parser.add_argument("--bin-chunk", type=int, default=5_000_000)
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    scene_audit_path = Path(args.scene_audit).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()

    try:
        metadata = load_source_metadata(source_cache)
        n_rows = int(metadata["n_rows"])
        offsets = case_offsets(metadata)
        case_all = mmap_array(source_cache, metadata, "case")
        primary_all = mmap_array(source_cache, metadata, "input_index")
        prediction_all = mmap_array(source_cache, metadata, "v22_prediction")
        official_train = mmap_array(source_cache, metadata, "official_train")

        audit = pd.read_feather(
            scene_audit_path,
            columns=["case", "input_index", "n_pairs", "scene_prediction"],
        )
        if len(audit) != 4_506_407:
            raise RuntimeError(f"scene-audit row count drifted: {len(audit):,}")
        audit_case = audit.case.to_numpy(np.int64, copy=False)
        audit_primary = audit.input_index.to_numpy(np.int64, copy=False)
        audit_count = audit.n_pairs.to_numpy(np.int64, copy=False)
        audit_prediction = audit.scene_prediction.to_numpy(np.float64, copy=False)
        audit_backwards = (audit_case[1:] < audit_case[:-1]) | (
            (audit_case[1:] == audit_case[:-1])
            & (audit_primary[1:] <= audit_primary[:-1])
        )
        if audit_backwards.any() or set(np.unique(audit_case)) != set(range(40, 200)):
            raise RuntimeError("scene-audit keys are not unique ordered cases 40--199")

        scene_prediction = np.lib.format.open_memmap(
            temporary / "scene_prediction.npy", mode="w+", dtype=np.float32,
            shape=(n_rows,),
        )
        scene_n_pairs = np.lib.format.open_memmap(
            temporary / "scene_n_pairs.npy", mode="w+", dtype=np.uint8,
            shape=(n_rows,),
        )

        edge_population: list[np.ndarray] = []
        case_scene_counts = np.zeros(CASE_MAX + 1, dtype=np.int64)
        case_pair_multiplicity_mean = np.zeros(CASE_MAX + 1, dtype=np.float64)
        audit_max_abs = 0.0
        audit_max_count_error = 0
        for case in range(CASE_MAX + 1):
            start, stop = int(offsets[case]), int(offsets[case + 1])
            local_case = np.asarray(case_all[start:stop], dtype=np.int64)
            if len(local_case) == 0 or not np.all(local_case == case):
                raise RuntimeError(f"source rows for case {case} are not contiguous")
            primary = np.asarray(primary_all[start:stop], dtype=np.int64)
            if np.any(primary[1:] < primary[:-1]):
                raise RuntimeError(f"case {case}: primary rows are not ordered")
            changes = np.empty(len(primary), dtype=bool)
            changes[0] = True
            changes[1:] = primary[1:] != primary[:-1]
            starts = np.flatnonzero(changes)
            counts = np.diff(np.r_[starts, len(primary)]).astype(np.int64)
            if np.any(counts <= 0) or int(counts.max()) > 20:
                raise RuntimeError(f"case {case}: invalid deployed multiplicity")
            scene_primary = primary[starts]
            if len(np.unique(scene_primary)) != len(scene_primary):
                raise RuntimeError(f"case {case}: repeated non-contiguous primary")
            pair_prediction = np.asarray(
                prediction_all[start:stop], dtype=np.float64
            )
            sums = np.add.reduceat(pair_prediction, starts)
            stored_sums = sums.astype(np.float32)
            scene_prediction[start:stop] = np.repeat(stored_sums, counts)
            scene_n_pairs[start:stop] = np.repeat(counts.astype(np.uint8), counts)
            case_scene_counts[case] = len(starts)
            case_pair_multiplicity_mean[case] = float(counts.mean())
            if FIT_CASE_MIN <= case <= TRAIN_CASE_MAX:
                edge_population.append(stored_sums)

            if case >= FIT_CASE_MIN:
                audit_start = int(np.searchsorted(audit_case, case, side="left"))
                audit_stop = int(np.searchsorted(audit_case, case, side="right"))
                if not np.array_equal(
                    scene_primary, audit_primary[audit_start:audit_stop]
                ):
                    raise RuntimeError(f"case {case}: scene-audit primary keys differ")
                count_error = int(np.max(np.abs(
                    counts - audit_count[audit_start:audit_stop]
                )))
                prediction_error = float(np.max(np.abs(
                    sums - audit_prediction[audit_start:audit_stop]
                )))
                audit_max_count_error = max(audit_max_count_error, count_error)
                audit_max_abs = max(audit_max_abs, prediction_error)
                if count_error != 0 or prediction_error > 2.0e-6:
                    raise RuntimeError(
                        f"case {case}: scene replay differs from audit "
                        f"count={count_error} prediction={prediction_error:.3e}"
                    )
            if case % 20 == 0 or case == CASE_MAX:
                print(
                    f"case {case}: pairs={stop-start:,} scenes={len(starts):,} "
                    f"mean_n={counts.mean():.3f}", flush=True,
                )

        scene_prediction.flush()
        scene_n_pairs.flush()
        if int(case_scene_counts[FIT_CASE_MIN:].sum()) != len(audit):
            raise RuntimeError("replayed fitting-scene count differs from audit")
        edge_values = np.concatenate(edge_population).astype(np.float32, copy=False)
        forced_edges = tuple(args.forced_edge or DEFAULT_FORCED_EDGES)
        edges = make_scene_edges(
            edge_values, args.n_scene_bins, forced_edges=forced_edges
        )
        del edge_population, edge_values

        scene_bin = np.lib.format.open_memmap(
            temporary / "scene_bin.npy", mode="w+", dtype=np.uint8,
            shape=(n_rows,),
        )
        for start in range(0, n_rows, args.bin_chunk):
            stop = min(start + args.bin_chunk, n_rows)
            scene_bin[start:stop] = assign_scene_bins(
                scene_prediction[start:stop], edges
            )
        scene_bin.flush()

        case_values = np.asarray(case_all, dtype=np.int16)
        official_values = np.asarray(official_train, dtype=bool)
        role_masks = {
            "external_development_c0_19": case_values <= EXTERNAL_DEVELOPMENT_CASE_MAX,
            "external_final_c20_39": (
                (case_values >= EXTERNAL_FINAL_CASE_MIN)
                & (case_values <= EXTERNAL_FINAL_CASE_MAX)
            ),
            "candidate_train_official_validation_c40_159": (
                (case_values >= FIT_CASE_MIN) & (case_values <= TRAIN_CASE_MAX)
                & ~official_values
            ),
            "candidate_internal_validation_c160_199": (
                (case_values >= INTERNAL_VALIDATION_CASE_MIN)
                & (case_values <= CASE_MAX) & ~official_values
            ),
            "final_train_official_validation_c40_199": (
                (case_values >= FIT_CASE_MIN) & ~official_values
            ),
        }
        role_rows = {name: int(mask.sum()) for name, mask in role_masks.items()}
        expected_final = int(metadata["official_random_row_split"]["validation_rows"])
        if role_rows["final_train_official_validation_c40_199"] != expected_final:
            raise RuntimeError("official-validation role count drifted")

        payload = {
            "schema_version": 1,
            "source_tag": SOURCE_TAG,
            "source_case_window": [0, CASE_MAX],
            "source_cache": str(source_cache),
            "source_metadata": str(source_cache / "metadata.json"),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "scene_audit": str(scene_audit_path),
            "scene_audit_sha256": sha256(scene_audit_path),
            "n_rows": n_rows,
            "n_scenes": int(case_scene_counts.sum()),
            "case_scene_counts": {
                str(case): int(case_scene_counts[case])
                for case in range(CASE_MAX + 1)
            },
            "case_mean_n_pairs": {
                str(case): float(case_pair_multiplicity_mean[case])
                for case in range(CASE_MAX + 1)
            },
            "feature_names": CONDITIONAL_FEATURES,
            "scene_coordinate": (
                "sum of frozen V2.2 predictions over the complete supported "
                "pair list for (case,input_index)"
            ),
            "scene_bin_definition": {
                "fit_cases": [FIT_CASE_MIN, TRAIN_CASE_MAX],
                "n_quantile_bins": int(args.n_scene_bins),
                "forced_edges": list(forced_edges),
                "edges": edges,
                "edge_convention": "lower-inclusive, upper-exclusive",
                "labels_used": False,
            },
            "role_rows": role_rows,
            "audit": {
                "fit_scene_rows": int(case_scene_counts[FIT_CASE_MIN:].sum()),
                "max_abs_scene_prediction_difference": audit_max_abs,
                "max_pair_count_difference": audit_max_count_error,
            },
            "arrays": {
                "scene_prediction": "scene_prediction.npy",
                "scene_n_pairs": "scene_n_pairs.npy",
                "scene_bin": "scene_bin.npy",
            },
            "response_labels_read": False,
            "constgold_opened": False,
            "anchor_truth_opened": False,
        }
        strict_json(temporary / "metadata.json", payload)
        os.replace(temporary, output)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print("V22_GROUPED_RSCENE_CACHE_DONE", flush=True)
    except Exception:
        print(f"failed cache remains for inspection at {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
