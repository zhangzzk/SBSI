#!/usr/bin/env python3
"""Prepare coherent usable-event labels on the full simulation parent.

The classifier population is the complete target-role half of each half-shear
scene.  Target-role membership is an experiment boundary (nonzero applied
g=0.05 shear), not an analysis selection.  No truth magnitude/size cut and no
measured MAG_AUTO/FLUX_RADIUS cut is applied.  Per-leg U requires a unique
crossmatch plus a finite, non-sentinel shape row compatible with the physical
flow support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.fixed_g0_domain import FLOW_FEATURES, packed_keys
from sbsi.shear_map import apply_shear_to_ellipticity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def catalogue_root(raw_root: Path, case: int, shear: str) -> Path:
    return raw_root / f"case{case}_{shear}" / "real0" / "catalogues"


def read_truth(raw_root: Path, case: int, shear: str, tile: str) -> pd.DataFrame:
    return pd.read_feather(
        catalogue_root(raw_root, case, shear)
        / "input"
        / f"gals_info_{tile}.feather"
    )


def usable_labels(
    raw_root: Path,
    case: int,
    shear: str,
    tile: str,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    root = catalogue_root(raw_root, case, shear)
    cross = pd.read_feather(
        root / "CrossMatch" / f"{tile}_rot0_matched.feather",
        columns=["id_detec", "id_input"],
    )
    shape = pd.read_feather(
        root
        / "Shapes"
        / f"shape_catalogue_detect_position_secondaries_{tile}.feather",
        columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"],
    )
    if cross["id_detec"].duplicated().any() or cross["id_input"].duplicated().any():
        raise RuntimeError(f"non-unique crossmatch: case{case:03d} shear={shear}")
    if shape["NUMBER"].duplicated().any():
        raise RuntimeError(f"non-unique shape row: case{case:03d} shear={shear}")
    joined = cross.merge(
        shape,
        left_on="id_detec",
        right_on="NUMBER",
        how="inner",
        validate="one_to_one",
    )
    parent_position = pd.Index(parent_ids).get_indexer(joined["id_input"])
    in_parent = parent_position >= 0
    values = joined[["NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"]].to_numpy(float)
    valid = in_parent & np.isfinite(values).all(axis=1)
    valid &= values[:, 0] != -1.0
    valid &= ~((values[:, 0] == 0.0) & (values[:, 1] == 0.0))
    valid &= np.square(values[:, 0]) + np.square(values[:, 1]) < 1.0
    valid &= values[:, 3] > 0.0
    detected = np.zeros(len(parent_ids), dtype=np.float32)
    detected_position = parent_position[in_parent]
    if len(np.unique(detected_position)) != len(detected_position):
        raise RuntimeError(f"duplicate parent detection: case{case:03d} shear={shear}")
    detected[detected_position] = 1.0
    usable = np.zeros(len(parent_ids), dtype=np.float32)
    usable[parent_position[valid]] = 1.0
    return detected, usable, {
        "parent_rows": int(len(parent_ids)),
        "cross_rows": int(len(cross)),
        "shape_rows": int(len(shape)),
        "cross_shape_rows": int(len(joined)),
        "parent_detected": int(detected.sum()),
        "parent_usable": int(usable.sum()),
        "cross_without_shape": int(len(cross) - len(joined)),
        "shape_without_cross": int(len(shape) - len(joined)),
        "detections_outside_parent": int((~in_parent).sum()),
    }


def base_context(truth: pd.DataFrame, target_rows: np.ndarray) -> np.ndarray:
    e1 = truth["e1_input_rot0"].to_numpy(float)[target_rows]
    e2 = truth["e2_input_rot0"].to_numpy(float)[target_rows]
    g1 = truth["gamma1_input"].to_numpy(float)[target_rows]
    g2 = truth["gamma2_input"].to_numpy(float)[target_rows]
    folded1, folded2 = apply_shear_to_ellipticity(e1, e2, g1, g2)
    q = truth["axis_ratio_input"].to_numpy(float)[target_rows]
    circularized = truth["Re_input"].to_numpy(float)[target_rows] * np.sqrt(q)
    return np.column_stack(
        (
            folded1,
            folded2,
            truth["sersic_n_input"].to_numpy(float)[target_rows],
            truth["r_input"].to_numpy(float)[target_rows],
            circularized,
        )
    ).astype(np.float32)


def prepare(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full-parent classifier preparation must run under Slurm")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"preserving existing classifier preparation: {output_root}")
    output_root.mkdir(parents=True)
    case_dir = output_root / "cases"
    case_dir.mkdir()
    cases = list(range(args.case_start, args.case_stop))
    started = time.monotonic()

    partial = {}
    case_records = {}
    parent_keys_by_case = {}
    for case in cases:
        zero = read_truth(args.raw_simulation_root, case, "0.0", args.tile_name)
        shear = read_truth(args.raw_simulation_root, case, "0.05", args.tile_name)
        ids0 = zero["index_input"].to_numpy(np.int64)
        idsg = shear["index_input"].to_numpy(np.int64)
        np.testing.assert_array_equal(ids0, idsg)
        amplitude = np.hypot(
            shear["gamma1_input"].to_numpy(float),
            shear["gamma2_input"].to_numpy(float),
        )
        if not np.all(np.isclose(amplitude, 0.0) | np.isclose(amplitude, 0.05)):
            raise RuntimeError(f"unexpected half-shear role amplitudes in case{case:03d}")
        target_rows = np.flatnonzero(np.isclose(amplitude, 0.05))
        parent_ids = idsg[target_rows]
        if len(parent_ids) == 0 or len(np.unique(parent_ids)) != len(parent_ids):
            raise RuntimeError(f"invalid target-role parent in case{case:03d}")
        # g=0 and g=0.05 target-role halves are aligned by the g=0.05 role flag.
        x0 = base_context(zero, target_rows)
        x1 = base_context(shear, target_rows)
        d0, u0, label0 = usable_labels(
            args.raw_simulation_root, case, "0.0", args.tile_name, parent_ids
        )
        d1, u1, label1 = usable_labels(
            args.raw_simulation_root, case, "0.05", args.tile_name, parent_ids
        )
        partial[case] = {
            "ids": parent_ids,
            "x0": x0,
            "x1": x1,
            "d0": d0,
            "d1": d1,
            "u0": u0,
            "u1": u1,
        }
        parent_keys_by_case[case] = np.sort(parent_ids)
        case_records[str(case)] = {
            "parent_definition": "g=0.05 target role (|gamma_primary|=0.05)",
            "parent_rows": int(len(parent_ids)),
            "g0": label0,
            "g05": label1,
        }
        print(
            f"CLASSIFIER_LABELS case={case} parent={len(parent_ids):,} "
            f"U0={int(u0.sum()):,} U05={int(u1.sum()):,}",
            flush=True,
        )

    crowd_parts = {case: [] for case in cases}
    crowd_columns = ["case", "input_index", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]
    with ipc.open_file(args.crowd_catalogue) as reader:
        missing = sorted(set(crowd_columns) - set(reader.schema.names))
        if missing:
            raise KeyError(f"crowd catalogue lacks {missing}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(crowd_columns).to_pandas()
            in_window = frame["case"].isin(cases).to_numpy()
            if in_window.any():
                frame = frame.loc[in_window]
                for case, group in frame.groupby("case", sort=False):
                    case = int(case)
                    keep = np.isin(
                        group["input_index"].to_numpy(np.int64),
                        parent_keys_by_case[case],
                        assume_unique=False,
                    )
                    if keep.any():
                        crowd_parts[case].append(group.loc[keep].copy())
            if (batch_index + 1) % 100 == 0:
                print(
                    f"CLASSIFIER_CROWD_PROGRESS batches={batch_index + 1}/{reader.num_record_batches}",
                    flush=True,
                )

    total_parent = 0
    total_u0 = 0
    total_u1 = 0
    for case in cases:
        data = partial.pop(case)
        if not crowd_parts[case]:
            raise RuntimeError(f"no crowd rows found for case{case:03d}")
        crowd = pd.concat(crowd_parts[case], ignore_index=True)
        if crowd.duplicated(["case", "input_index"]).any():
            raise RuntimeError(f"duplicate crowd key for case{case:03d}")
        aligned = pd.DataFrame({"input_index": data["ids"]}).merge(
            crowd.drop(columns="case"),
            on="input_index",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        crowd_values = aligned[["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]].to_numpy(
            np.float32
        )
        if not np.isfinite(crowd_values).all():
            raise RuntimeError(f"missing/nonfinite parent crowd values for case{case:03d}")
        x0 = np.column_stack((data["x0"], crowd_values)).astype(np.float32)
        x1 = np.column_stack((data["x1"], crowd_values)).astype(np.float32)
        if x0.shape[1] != len(FLOW_FEATURES) or not np.isfinite(x0).all() or not np.isfinite(x1).all():
            raise RuntimeError(f"invalid classifier context for case{case:03d}")
        path = case_dir / f"case{case:03d}.npz"
        temporary = case_dir / f"case{case:03d}.tmp.npz"
        np.savez(
            temporary,
            ids=data["ids"],
            x0=x0,
            x1=x1,
            d0=data["d0"],
            d1=data["d1"],
            u0=data["u0"],
            u1=data["u1"],
        )
        os.replace(temporary, path)
        case_records[str(case)]["prepared_sha256"] = sha256(path)
        total_parent += len(data["ids"])
        total_u0 += int(data["u0"].sum())
        total_u1 += int(data["u1"].sum())

    manifest = {
        "format_version": 1,
        "population": {
            "event": "per-leg usable shape measurement U",
            "parent": "complete half-shear target-role simulation parent",
            "target_role": "|gamma_primary|=0.05 in the g=0.05 leg",
            "truth_analysis_cut": None,
            "measured_selection_cut": None,
            "inherited_generator_boundaries_only": True,
            "validity": (
                "unique crossmatch plus finite non-sentinel strictly-interior ngmix shape, "
                "finite MAG_AUTO/FLUX_RADIUS, and positive FLUX_RADIUS"
            ),
        },
        "features": list(FLOW_FEATURES),
        "cases": cases,
        "case_records": case_records,
        "counts": {
            "parent_objects": total_parent,
            "leg_rows": 2 * total_parent,
            "g0_usable": total_u0,
            "g05_usable": total_u1,
        },
        "paths": {
            "raw_simulation_root": str(args.raw_simulation_root.resolve()),
            "crowd_catalogue": str(args.crowd_catalogue.resolve()),
        },
        "crowd_catalogue_sha256": sha256(args.crowd_catalogue),
        "script_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        f"FULL_PARENT_CLASSIFIER_PREPARED parent={total_parent:,} "
        f"U0={total_u0:,} U05={total_u1:,}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-simulation-root", type=Path, required=True)
    parser.add_argument("--crowd-catalogue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-start", type=int, default=40)
    parser.add_argument("--case-stop", type=int, default=120)
    parser.add_argument("--tile-name", default="tile180.0_-0.5")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
