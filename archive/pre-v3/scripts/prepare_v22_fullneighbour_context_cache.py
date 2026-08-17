#!/usr/bin/env python3
"""Cache all-neighbour BlendEMU features for half-shear labelled primaries.

The half-shear response catalogue contains only the sheared half of the
neighbours.  Those rows remain the supervised examples.  This companion cache
enumerates the deployed neighbour set around the same primaries from the full
rendered input catalogue, so a scene context can sum predictions over both the
sheared and unsheared populations.

Pair finding exactly follows deployed V2.2 inference: true input positions,
the stored regression cuts, r_max=10 arcsec, and k=20 (including the primary
query before the zero-distance self pair is discarded).  Each case is written
atomically and a partial cache can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from blendemu.inference import BlendingPredictor  # noqa: E402
from blendemu import nz_utils  # noqa: E402
from scripts.prepare_v22_matched_decomposition import pair_id_columns  # noqa: E402
from scripts.tune_sequential_rblend_optuna import BASE_FEATURES  # noqa: E402
from scripts.v22_grouped_rscene_common import (  # noqa: E402
    case_offsets,
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)


COND = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"
TAG = "lsst_r_extnbr_v22"
TILE = "tile180.0_-0.5"
FIELD_COLUMNS = [
    "index_input",
    "RA_input",
    "DEC_input",
    "Re_input",
    "r_input",
    "sersic_n_input",
]
ARRAYS = {
    "x_scaled": "case{case:03d}_x_scaled.npy",
    "x_aux": "case{case:03d}_x_aux.npy",
    "pair_scene": "case{case:03d}_pair_scene.npy",
    "scene_primary": "case{case:03d}_scene_primary.npy",
    "scene_count": "case{case:03d}_scene_count.npy",
}


def input_path(simulation_root: Path, case: int) -> Path:
    return (
        simulation_root
        / f"case{case}_0.2"
        / "real0"
        / "catalogues"
        / "input"
        / f"gals_info_{TILE}.feather"
    )


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def case_paths(output: Path, case: int) -> dict[str, Path]:
    return {
        name: output / pattern.format(case=case)
        for name, pattern in ARRAYS.items()
    }


def existing_case_record(output: Path, case: int) -> dict[str, Any] | None:
    paths = case_paths(output, case)
    present = {name: path.exists() for name, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise RuntimeError(f"case {case}: incomplete existing cache files {present}")
    x_scaled = np.load(paths["x_scaled"], mmap_mode="r")
    x_aux = np.load(paths["x_aux"], mmap_mode="r")
    pair_scene = np.load(paths["pair_scene"], mmap_mode="r")
    scene_primary = np.load(paths["scene_primary"], mmap_mode="r")
    scene_count = np.load(paths["scene_count"], mmap_mode="r")
    n_pair = len(pair_scene)
    n_scene = len(scene_primary)
    if x_scaled.shape != (n_pair, len(BASE_FEATURES)):
        raise RuntimeError(f"case {case}: bad cached x_scaled shape")
    if x_aux.shape != (n_pair, 2) or scene_count.shape != (n_scene,):
        raise RuntimeError(f"case {case}: bad cached auxiliary shape")
    if int(scene_count.sum()) != n_pair:
        raise RuntimeError(f"case {case}: cached scene counts do not close")
    if n_pair and (pair_scene.min() < 0 or pair_scene.max() >= n_scene):
        raise RuntimeError(f"case {case}: invalid cached pair-scene index")
    return {
        "case": case,
        "n_scenes": n_scene,
        "n_pairs": n_pair,
        "n_zero_pair_scenes": int(np.count_nonzero(scene_count == 0)),
        "mean_pairs_per_scene": float(scene_count.mean()),
        "max_pairs_per_scene": int(scene_count.max(initial=0)),
    }


def build_case(
    *,
    case: int,
    output: Path,
    simulation_root: Path,
    source_primary: np.ndarray,
    predictor: BlendingPredictor,
    cuts: list[list[float]],
    rmax: float,
    kmax: int,
) -> dict[str, Any]:
    paths = case_paths(output, case)
    field_raw = pd.read_feather(
        input_path(simulation_root, case), columns=FIELD_COLUMNS
    )
    field = field_raw.rename(
        columns={column: column.replace("_input", "") for column in field_raw}
    )
    if field["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate renderer input ID")
    scene_primary = np.unique(np.asarray(source_primary, dtype=np.int64))
    field_ids = field["index"].to_numpy(np.int64, copy=False)
    if not np.isin(scene_primary, field_ids).all():
        missing = scene_primary[~np.isin(scene_primary, field_ids)][:5]
        raise RuntimeError(f"case {case}: labelled primaries missing: {missing.tolist()}")
    anchors = field[field["index"].isin(scene_primary)].copy()
    pairs = nz_utils.icat2reg(
        anchors,
        field,
        predictor.bst_reg,
        predictor.conditions,
        cuts=cuts,
        r_max=rmax,
        k=kmax,
    )
    pcol, scol = pair_id_columns(pairs)
    pairs = pairs.sort_values([pcol, scol], kind="stable").reset_index(drop=True)
    primary = pairs[pcol].to_numpy(np.int64, copy=False)
    secondary = pairs[scol].to_numpy(np.int64, copy=False)
    if np.any(primary == secondary):
        raise RuntimeError(f"case {case}: zero-distance self pair survived")
    if pairs.duplicated([pcol, scol]).any():
        raise RuntimeError(f"case {case}: duplicate deployed pair")
    pair_scene = np.searchsorted(scene_primary, primary).astype(np.int32)
    if len(primary) and not np.array_equal(scene_primary[pair_scene], primary):
        raise RuntimeError(f"case {case}: pair primary escaped labelled scene set")
    scene_count = np.bincount(
        pair_scene, minlength=len(scene_primary)
    ).astype(np.int16)
    if int(scene_count.max(initial=0)) > kmax:
        raise RuntimeError(f"case {case}: pair multiplicity exceeds deployed k")
    x_scaled = pairs[BASE_FEATURES].to_numpy(np.float32)
    raw_size_p = pairs["Re_input_p"].to_numpy(np.float64)
    raw_size_s = pairs["Re_input_s"].to_numpy(np.float64)
    raw_mag_p = pairs["r_input_p"].to_numpy(np.float64)
    raw_mag_s = pairs["r_input_s"].to_numpy(np.float64)
    if np.any(raw_size_p <= 0.0) or np.any(raw_size_s <= 0.0):
        raise RuntimeError(f"case {case}: non-positive supported size")
    x_aux = np.column_stack((
        -0.4 * (raw_mag_s - raw_mag_p),
        np.log10(raw_size_s / raw_size_p),
    )).astype(np.float32)
    if not np.isfinite(x_scaled).all() or not np.isfinite(x_aux).all():
        raise RuntimeError(f"case {case}: non-finite cached feature")
    for name, value in {
        "x_scaled": x_scaled,
        "x_aux": x_aux,
        "pair_scene": pair_scene,
        "scene_primary": scene_primary,
        "scene_count": scene_count,
    }.items():
        atomic_npy(paths[name], value)
    return existing_case_record(output, case)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--simulation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-min", type=int, default=0)
    parser.add_argument("--case-max", type=int, default=199)
    args = parser.parse_args()
    if args.case_min != 0 or args.case_max != 199:
        raise ValueError("the fixed training experiment requires all cases 0--199")

    source_cache = Path(args.source_cache).resolve()
    simulation_root = Path(args.simulation_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        print(f"Full-neighbour cache already complete: {metadata_path}", flush=True)
        return

    source_meta = load_source_metadata(source_cache)
    source_primary = mmap_array(source_cache, source_meta, "input_index")
    offsets = case_offsets(source_meta)
    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=TAG, conditions=COND, device="cpu", load_self=False
    )
    cuts, rmax, kmax = predictor._select("regression")
    if cuts is None or float(rmax) != 10.0 or int(kmax) != 20:
        raise RuntimeError(
            f"expected deployed regression cuts/r_max=10/k=20, got {cuts}/{rmax}/{kmax}"
        )

    records: list[dict[str, Any]] = []
    for case in range(args.case_min, args.case_max + 1):
        record = existing_case_record(output, case)
        if record is None:
            start, stop = int(offsets[case]), int(offsets[case + 1])
            record = build_case(
                case=case,
                output=output,
                simulation_root=simulation_root,
                source_primary=np.asarray(source_primary[start:stop]),
                predictor=predictor,
                cuts=cuts,
                rmax=float(rmax),
                kmax=int(kmax),
            )
        records.append(record)
        print(
            f"CASE_DONE case={case} scenes={record['n_scenes']:,} "
            f"pairs={record['n_pairs']:,} mean={record['mean_pairs_per_scene']:.3f} "
            f"zero={record['n_zero_pair_scenes']:,}",
            flush=True,
        )

    n_scenes = int(sum(record["n_scenes"] for record in records))
    n_pairs = int(sum(record["n_pairs"] for record in records))
    zero_scenes = int(sum(record["n_zero_pair_scenes"] for record in records))
    payload = {
        "schema_version": 1,
        "kind": "all-deployed-neighbour features for half-shear labelled primaries",
        "source_tag": TAG,
        "source_cache": str(source_cache),
        "source_metadata_sha256": sha256(source_cache / "metadata.json"),
        "simulation_root": str(simulation_root),
        "renderer_input_leg": "case{case}_0.2/real0 exact input catalogue",
        "case_window": [args.case_min, args.case_max],
        "neighbour_definition": {
            "population": "all rendered objects (both half-shear populations)",
            "primary_position": "true renderer input RA/DEC",
            "cuts": cuts,
            "r_max_arcsec": float(rmax),
            "k": int(kmax),
            "self_pair": "discarded by strict positive-distance query",
        },
        "base_features": BASE_FEATURES,
        "aux_features": [
            "log10_flux_s_over_flux_p",
            "log10_Re_s_over_Re_p",
        ],
        "arrays": ARRAYS,
        "n_scenes": n_scenes,
        "n_pairs": n_pairs,
        "n_zero_pair_scenes": zero_scenes,
        "mean_pairs_per_scene": float(n_pairs / n_scenes),
        "case_records": records,
        "labels_cached": False,
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    strict_json(metadata_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("V22_FULLNEIGHBOUR_CONTEXT_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
