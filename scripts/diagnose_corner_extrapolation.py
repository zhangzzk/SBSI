#!/usr/bin/env python3
"""Test the blending emulator against measured labels near the rejected corner.

`scripts/diagnose_label_rejection` establishes that BlendEMU's label pipeline
removes detections with a neighbour inside 3 arcsec brighter than five times,
so the emulator has neither training data nor a measured reference in that
corner of the pair feature space, while inference sums it there regardless.

The rejection fires on *measured* FLUX_AUTO at the detected position.  A pair
whose true configuration lies inside the corner therefore survives whenever the
blend redistributed enough measured flux for the ratio to fall below five.
Those survivors carry labels, so the emulator can be tested inside the corner
after all -- on a biased subsample of it, which is stated rather than hidden.

The diagnostic reports the measured and modelled per-pair response cell by cell
on the same grid, so the emulator's accuracy can be read as a function of
distance from the rejection boundary instead of assumed on either side of it.

It reports nothing about `m`: it is a component audit of one additive term.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.models import ModelPaths, load_emulator
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS
from scripts.build_fixed_g0_response_profile_predictions import (
    BLEND_RESPONSE_COLUMNS,
    response_anchor_for_case,
    valid_blend_response_pairs,
)
from scripts.diagnose_label_rejection import (
    DELTA_MAG_EDGES,
    DISTANCE_EDGES,
    REJECTION_DELTA_MAG,
    REJECTION_RADIUS_ARCSEC,
)


FIELDS = ("n", "model", "measured")

#  The label catalogue and the ConstGold anchor differ mainly in how bright
#  their PRIMARIES are, so the emulator's accuracy is resolved along that axis
#  too.  The bands are wide on purpose: this asks whether accuracy drifts with
#  primary brightness, not where exactly it turns over.
PRIMARY_MAG_EDGES = (-np.inf, 23.0, 24.0, 25.0, 26.0, np.inf)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--response-catalogue", type=Path, required=True)
    parser.add_argument(
        "--measurement-model",
        type=Path,
        required=True,
        help="flow checkpoint required by ModelPaths; unused by R_blend",
    )
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--response-shear", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def cell_shape() -> tuple[int, int]:
    return len(DISTANCE_EDGES) - 1, len(DELTA_MAG_EDGES) - 1


def primary_bands() -> int:
    return len(PRIMARY_MAG_EDGES) - 1


def primary_band_names() -> list[str]:
    names = []
    for low, high in zip(PRIMARY_MAG_EDGES[:-1], PRIMARY_MAG_EDGES[1:]):
        lo = "-inf" if low == -np.inf else f"{low:g}"
        hi = "inf" if high == np.inf else f"{high:g}"
        names.append(f"r_p[{lo},{hi})")
    return names


def primary_band_index(magnitude: np.ndarray) -> np.ndarray:
    """Half-open band of the PRIMARY's true r magnitude, clipped at both ends."""
    inner = np.array(PRIMARY_MAG_EDGES[1:-1])
    return np.clip(
        np.searchsorted(inner, np.asarray(magnitude, dtype=np.float64), side="right"),
        0,
        primary_bands() - 1,
    )


def cell_indices(distance: np.ndarray, delta_mag: np.ndarray):
    """Flat grid index per pair, plus the mask of pairs inside the footprint.

    Separations outside the grid are dropped rather than folded into an edge
    bin, so this shares a footprint exactly with the support audit.
    """
    distance = np.asarray(distance, dtype=np.float64)
    delta_mag = np.asarray(delta_mag, dtype=np.float64)
    rows, columns = cell_shape()
    inside = (
        np.isfinite(distance)
        & np.isfinite(delta_mag)
        & (distance >= DISTANCE_EDGES[0])
        & (distance <= DISTANCE_EDGES[-1])
    )
    # side="right" reproduces histogram2d's half-open bins, so a value sitting
    # exactly on an edge lands in the same cell as in the support audit.
    row = np.clip(
        np.searchsorted(np.array(DISTANCE_EDGES[1:-1]), distance[inside], side="right"),
        0,
        rows - 1,
    )
    column = np.clip(
        np.searchsorted(np.array(DELTA_MAG_EDGES[1:-1]), delta_mag[inside], side="right"),
        0,
        columns - 1,
    )
    return row * columns + column, inside


def accumulate(totals: dict[str, np.ndarray], frame: pd.DataFrame) -> None:
    """Add one batch of labelled pairs into a case's grid sums, in place.

    The sums are resolved by primary-magnitude band as well as by grid cell;
    summing the band axis away recovers exactly the two-dimensional grid this
    diagnostic reported before that axis existed.
    """
    flat, inside = cell_indices(frame["distance"], frame["delta_mag"])
    if not inside.any():
        return
    rows, columns = cell_shape()
    band = primary_band_index(frame["primary_mag"].to_numpy(np.float64)[inside])
    flat = band * (rows * columns) + flat
    np.add.at(totals["n"], flat, 1.0)
    np.add.at(totals["model"], flat, frame["model"].to_numpy(np.float64)[inside])
    np.add.at(totals["measured"], flat, frame["measured"].to_numpy(np.float64)[inside])


def empty_totals() -> dict[str, np.ndarray]:
    rows, columns = cell_shape()
    size = primary_bands() * rows * columns
    return {field: np.zeros(size, dtype=np.float64) for field in FIELDS}


def corner_mask_flat() -> np.ndarray:
    """Flat mask of the grid cells the rejection geometry covers."""
    rows, columns = cell_shape()
    distance_low = np.array(DISTANCE_EDGES[:-1])[:, None] * np.ones((1, columns))
    delta_low = np.ones((rows, 1)) * np.array(DELTA_MAG_EDGES[:-1])[None, :]
    inside = (distance_low < REJECTION_RADIUS_ARCSEC) & (
        delta_low >= REJECTION_DELTA_MAG
    )
    return inside.reshape(-1)


def scan_cases(
    predictor,
    response_catalogue: Path,
    cases: tuple[int, ...],
    anchor_by_case: dict[int, np.ndarray],
    response_shear: float,
):
    """Predict and accumulate the labelled pair grid, one pass over the file."""
    allowed = np.asarray(sorted(cases), dtype=np.int64)
    totals = {case: empty_totals() for case in cases}
    counts = {
        case: {"anchored_pair_rows": 0, "invalid_pair_rows_dropped": 0, "valid_pair_rows": 0}
        for case in cases
    }
    previous_max = None
    with ipc.open_file(response_catalogue) as reader:
        missing = sorted(set(BLEND_RESPONSE_COLUMNS) - set(reader.schema.names))
        if missing:
            raise KeyError(f"response catalogue lacks {missing}")
        case_column = reader.schema.get_field_index("case")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            case_values = batch.column(case_column).to_numpy(zero_copy_only=False)
            batch_min, batch_max = int(case_values.min()), int(case_values.max())
            if previous_max is not None and batch_min < previous_max:
                raise RuntimeError("response catalogue is not ordered by case")
            previous_max = batch_max
            if batch_max < allowed.min():
                continue
            if batch_min > allowed.max():
                break
            frame = pa.Table.from_batches([batch]).select(
                BLEND_RESPONSE_COLUMNS
            ).to_pandas()
            frame = frame.loc[np.isin(frame["case"].to_numpy(np.int64), allowed)]
            if frame.empty:
                continue
            anchored = np.zeros(len(frame), dtype=bool)
            case_column_values = frame["case"].to_numpy(np.int64)
            for case in np.unique(case_column_values):
                local = case_column_values == case
                anchored[local] = np.isin(
                    frame.loc[local, "input_index"].to_numpy(np.int64),
                    anchor_by_case[int(case)],
                )
            frame = frame.loc[anchored]
            if frame.empty:
                continue
            valid = valid_blend_response_pairs(frame)
            case_column_values = frame["case"].to_numpy(np.int64)
            for case in np.unique(case_column_values):
                case = int(case)
                local = case_column_values == case
                counts[case]["anchored_pair_rows"] += int(local.sum())
                counts[case]["invalid_pair_rows_dropped"] += int((local & ~valid).sum())
                counts[case]["valid_pair_rows"] += int((local & valid).sum())
            frame = frame.loc[valid]
            if frame.empty:
                continue
            predicted = predictor.predict_on_pairs(
                frame, task="response", rescaled=False, warn_extrapolation=False
            )
            model = predicted["response"].to_numpy(np.float64)
            if len(model) != len(frame) or not np.isfinite(model).all():
                raise RuntimeError("response emulator returned invalid pair predictions")
            block = pd.DataFrame(
                {
                    "case": frame["case"].to_numpy(np.int64),
                    "distance": frame["distance"].to_numpy(np.float64),
                    "delta_mag": frame["r_input_p"].to_numpy(np.float64)
                    - frame["r_input_s"].to_numpy(np.float64),
                    "model": model,
                    "primary_mag": frame["r_input_p"].to_numpy(np.float64),
                    "measured": frame["delta_et1"].to_numpy(np.float64) / response_shear,
                }
            )
            for case, part in block.groupby("case", sort=False):
                accumulate(totals[int(case)], part)
            if (batch_index + 1) % 200 == 0:
                print(
                    f"CORNER_EXTRAPOLATION_SCAN batch={batch_index + 1}/"
                    f"{reader.num_record_batches}",
                    flush=True,
                )
    for case in cases:
        if totals[case]["n"].sum() == 0:
            raise RuntimeError(f"case {case} contributed no labelled pairs")
    return totals, counts


def stack_cases(totals: dict[int, dict[str, np.ndarray]]):
    """Return the case order and a (case, primary band, cell, field) array."""
    cases = tuple(sorted(totals))
    rows, columns = cell_shape()
    stacked = np.stack(
        [
            np.stack([totals[case][field] for field in FIELDS], axis=-1)
            for case in cases
        ],
        axis=0,
    )
    return cases, stacked.reshape(
        len(cases), primary_bands(), rows * columns, len(FIELDS)
    )


def collapse(stacked: np.ndarray) -> np.ndarray:
    """Sum the primary-magnitude axis away, giving the original (case, cell, field)."""
    return stacked.sum(axis=1)


def pooled(stacked: np.ndarray) -> dict[str, np.ndarray]:
    """Pool cases by summing, then form the per-pair means and their ratio.

    Pooling by sum rather than by averaging case means weights each case by the
    pairs it actually contributed, which is what the inference-time sum does.
    """
    n = stacked[..., FIELDS.index("n")].sum(axis=0)
    model = stacked[..., FIELDS.index("model")].sum(axis=0)
    measured = stacked[..., FIELDS.index("measured")].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "n": n,
            "mean_model": np.where(n > 0, model / np.maximum(n, 1.0), np.nan),
            "mean_measured": np.where(n > 0, measured / np.maximum(n, 1.0), np.nan),
            "model_over_measured": np.where(
                measured != 0.0, model / np.where(measured == 0.0, 1.0, measured), np.nan
            ),
            "response_share": model / model.sum() if model.sum() != 0 else np.full_like(model, np.nan),
        }


def bootstrap(stacked: np.ndarray, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Case bootstrap of the per-cell means and of the model/measured ratio."""
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, stacked.shape[0], size=(n_boot, stacked.shape[0]))
    replicates = stacked[draws].sum(axis=1)
    n = replicates[..., FIELDS.index("n")]
    model = replicates[..., FIELDS.index("model")]
    measured = replicates[..., FIELDS.index("measured")]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_model = np.where(n > 0, model / np.maximum(n, 1.0), np.nan)
        mean_measured = np.where(n > 0, measured / np.maximum(n, 1.0), np.nan)
        ratio = np.where(measured != 0.0, model / np.where(measured == 0.0, 1.0, measured), np.nan)
    return {
        "mean_model": np.nanstd(mean_model, axis=0, ddof=1),
        "mean_measured": np.nanstd(mean_measured, axis=0, ddof=1),
        "model_over_measured": np.nanstd(ratio, axis=0, ddof=1),
    }


def region_report(stacked: np.ndarray, mask: np.ndarray, n_boot: int, seed: int) -> dict:
    """Aggregate a set of cells and bootstrap the aggregate the same way."""
    region = stacked[:, mask, :].sum(axis=1)
    n = region[:, FIELDS.index("n")].sum()
    model = region[:, FIELDS.index("model")].sum()
    measured = region[:, FIELDS.index("measured")].sum()
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, stacked.shape[0], size=(n_boot, stacked.shape[0]))
    replicates = region[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = replicates[:, FIELDS.index("model")] / replicates[:, FIELDS.index("measured")]
        mean_model = replicates[:, FIELDS.index("model")] / replicates[:, FIELDS.index("n")]
    if n == 0:
        return {"pairs": 0, "status": "no_labelled_pairs"}
    return {
        "pairs": int(n),
        "mean_model": float(model / n),
        "mean_measured": float(measured / n),
        "mean_model_standard_error": float(np.nanstd(mean_model, ddof=1)),
        "model_over_measured": float(model / measured) if measured != 0 else None,
        "model_over_measured_standard_error": float(np.nanstd(ratio, ddof=1)),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("corner extrapolation diagnosis must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if len(cases) != len(args.case):
        raise ValueError("cases must be unique")
    if not np.isfinite(args.response_shear) or args.response_shear <= 0:
        raise ValueError("response shear must be positive and finite")
    for path in (
        args.response_catalogue,
        args.measurement_model,
        args.emulator_model,
        args.emulator_metadata,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(args.emulator_metadata.read_text())
    regression = metadata["tasks"]["regression"]
    if regression.get("cuts") is not None:
        raise ValueError("this diagnosis requires emulator cuts=null")
    if regression.get("truth_analysis_cuts_applied") is not False:
        raise ValueError("this diagnosis requires no truth analysis cut")

    paths = ModelPaths(
        flow_checkpoints=(args.measurement_model,),
        emulator_model=args.emulator_model,
        emulator_metadata=args.emulator_metadata,
    )
    predictor = load_emulator(paths, conditions=CONDITIONS, device=args.device)
    anchor_path = args.domain_root / "response_anchor.feather"
    anchor_by_case = {case: response_anchor_for_case(anchor_path, case) for case in cases}

    totals, counts = scan_cases(
        predictor, args.response_catalogue, cases, anchor_by_case, args.response_shear
    )
    case_order, resolved = stack_cases(totals)
    stacked = collapse(resolved)
    rows, columns = cell_shape()
    grids = pooled(stacked)
    errors = bootstrap(stacked, args.n_boot, args.bootstrap_seed)

    corner = corner_mask_flat()
    distance_low = np.repeat(np.array(DISTANCE_EDGES[:-1]), columns)
    delta_low = np.tile(np.array(DELTA_MAG_EDGES[:-1]), rows)
    regions = {
        # the corner the pipeline removes, reachable only through pairs whose
        # measured flux ratio fell below the threshold their true one exceeds
        "rejected_corner": corner,
        # the two strips that share a boundary with it
        "adjacent_in_magnitude": (distance_low < REJECTION_RADIUS_ARCSEC)
        & (delta_low >= 1.0)
        & (delta_low < REJECTION_DELTA_MAG),
        "adjacent_in_distance": (distance_low >= REJECTION_RADIUS_ARCSEC)
        & (distance_low < 4.0)
        & (delta_low >= REJECTION_DELTA_MAG),
        "close_and_fainter_neighbour": (distance_low < REJECTION_RADIUS_ARCSEC)
        & (delta_low < 0.0),
        "everything": np.ones(rows * columns, dtype=bool),
    }

    result = {
        "format_version": 1,
        "purpose": (
            "accuracy of the blending emulator against measured labels as a "
            "function of distance from the label pipeline's rejection boundary"
        ),
        "cases": list(case_order),
        "response_shear": args.response_shear,
        "grid": {
            "distance_edges_arcsec": list(DISTANCE_EDGES),
            "delta_magnitude_edges": list(DELTA_MAG_EDGES),
            "delta_magnitude_sign": "true r of primary minus true r of neighbour; "
            "positive means the neighbour is brighter",
            "shape": [rows, columns],
            **{name: value.reshape(rows, columns) for name, value in grids.items()},
            **{
                f"{name}_standard_error": value.reshape(rows, columns)
                for name, value in errors.items()
            },
        },
        "regions": {
            name: region_report(stacked, mask, args.n_boot, args.bootstrap_seed)
            for name, mask in regions.items()
        },
        "primary_magnitude_edges": list(PRIMARY_MAG_EDGES),
        "regions_by_primary_magnitude": {
            name: {
                band: region_report(
                    resolved[:, index], mask, args.n_boot, args.bootstrap_seed
                )
                for index, band in enumerate(primary_band_names())
            }
            for name, mask in regions.items()
        },
        "case_counts": {str(case): counts[case] for case in case_order},
        "inputs": {
            "response_catalogue": str(args.response_catalogue),
            "response_catalogue_sha256": file_sha256(args.response_catalogue),
            "emulator_model": str(args.emulator_model),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "emulator_metadata": str(args.emulator_metadata),
            "emulator_metadata_sha256": file_sha256(args.emulator_metadata),
        },
        "limitations": [
            "Pairs inside the rejected corner are present only because the "
            "pipeline's measured flux ratio fell below five where the true "
            "ratio exceeds it; they are a biased subsample of that corner and "
            "are plausibly its least blended members.",
            "A single-pair comparison is far noisier than a per-primary sum; "
            "read the region aggregates, not individual cells, for accuracy.",
            "The measured label is one realisation of a noisy shape difference; "
            "its mean is unbiased but its scatter does not shrink with the model.",
            "Case bootstrap over the supplied cases only; one emulator, one seed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, output)
    print(f"CORNER_EXTRAPOLATION_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
