#!/usr/bin/env python3
"""Measure the blending response that BlendEMU's label pipeline cannot reach.

`blendemu.response.retrieve_response` drops every detection that has a
neighbouring detection within 3 arcsec brighter than it by more than a factor
of five, independently in each of the two shear legs.  The surviving pair rows
are the response catalogue, and that same catalogue -- restricted to the same
anchor -- is both the emulator's training set and the only measured reference
`R_blend` has ever been compared against.  Inference, by contrast, sums the
emulator over the complete truth scene.

This diagnostic quantifies the consequence.  It asks, for the same primaries
and the same emulator, (a) whether the anchor primaries that own no catalogue
row are the ones the rejection would remove, and (b) how much of the
inference-time `R_blend` sum is carried by pairs in the region of feature space
the catalogue cannot populate.

It reports nothing about `m`: it is a support audit of one additive term.
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
from scipy.spatial import cKDTree

from sbsi.models import ModelPaths, load_emulator
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS, prepare_truth
from scripts.build_fixed_g0_response_profile_predictions import (
    BLEND_RESPONSE_COLUMNS,
    response_anchor_for_case,
    valid_blend_response_pairs,
)


# BlendEMU's rejection, verbatim from `utils.remove_detection_w_bright_neighbour`
# as called by `response.retrieve_response`: a detection is dropped when some
# neighbour within 3 arcsec exceeds it in FLUX_AUTO by more than a factor 5.
REJECTION_RADIUS_ARCSEC = 3.0
REJECTION_FLUX_RATIO = 5.0
REJECTION_NEIGHBOURS = 30
# A flux ratio of five is a magnitude difference of 2.5 log10 5.
REJECTION_DELTA_MAG = 2.5 * math.log10(REJECTION_FLUX_RATIO)

# Pair-level grid.  The separation edges bracket the 3 arcsec rejection radius
# and the magnitude edges bracket the factor-five threshold, so the rejected
# corner is one exact block of the grid rather than an interpolation of it.
DISTANCE_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
DELTA_MAG_EDGES = (-np.inf, -2.0, -1.0, 0.0, 1.0, REJECTION_DELTA_MAG, 3.0, np.inf)

PRIMARY_CLASSES = ("labelled", "unlabelled")
BRIGHT_CLASSES = ("bright_neighbour", "no_bright_neighbour")


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
    parser.add_argument(
        "--input-pattern", required=True, help="truth format string containing {case}"
    )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def row_lookup(identity: np.ndarray) -> np.ndarray:
    """Map a source identity onto its row in the truth table.

    The truth identities are dense but not guaranteed to start at zero, so a
    lookup array is both simpler and faster than a searchsorted on every pair.
    """
    identity = np.asarray(identity, dtype=np.int64)
    if identity.min() < 0:
        raise ValueError("truth identities must be non-negative")
    lookup = np.full(int(identity.max()) + 1, -1, dtype=np.int64)
    lookup[identity] = np.arange(len(identity), dtype=np.int64)
    return lookup


def bright_neighbour_audit(truth: pd.DataFrame, anchor_ids: np.ndarray) -> pd.DataFrame:
    """Apply BlendEMU's rejection geometry to the truth scene.

    The pipeline rejects on measured FLUX_AUTO at the detected position; this
    uses true flux at the true position, so it is a proxy and not the rule
    itself.  A blend biases measured flux upward for both members, which moves
    a pair toward equal measured brightness, so the proxy should if anything
    under-count rejections rather than invent them.
    """
    for column in ("index", "RA", "DEC", "r"):
        if column not in truth:
            raise KeyError(f"truth table lacks {column!r}")
    positions = np.column_stack(
        (truth["RA"].to_numpy(np.float64), truth["DEC"].to_numpy(np.float64))
    )
    magnitude = truth["r"].to_numpy(np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(magnitude).all():
        raise RuntimeError("truth scene carries non-finite position or magnitude")
    lookup = row_lookup(truth["index"].to_numpy(np.int64))
    anchor_rows = lookup[np.asarray(anchor_ids, dtype=np.int64)]
    if (anchor_rows < 0).any():
        raise RuntimeError("anchor contains identities absent from the truth scene")

    tree = cKDTree(positions)
    radius = REJECTION_RADIUS_ARCSEC / 3600.0
    distance, neighbour = tree.query(
        positions[anchor_rows],
        k=REJECTION_NEIGHBOURS + 1,
        distance_upper_bound=radius,
        workers=-1,
    )
    # cKDTree pads short neighbour lists with an out-of-range row and infinite
    # distance; the first column is the primary matching itself.
    present = np.isfinite(distance) & (neighbour < len(positions))
    present[:, 0] = False
    neighbour_rows = np.where(present, neighbour, 0)
    delta_mag = magnitude[anchor_rows][:, None] - magnitude[neighbour_rows]
    delta_mag = np.where(present, delta_mag, -np.inf)
    brightest = delta_mag.max(axis=1)

    saturated = int((present.sum(axis=1) >= REJECTION_NEIGHBOURS).sum())
    if saturated:
        # k is BlendEMU's own value, so saturation reproduces the pipeline
        # rather than truncating relative to it; record it regardless.
        print(
            f"LABEL_REJECTION_NOTE primaries_at_neighbour_cap={saturated}", flush=True
        )
    separation = np.where(present, distance, np.inf).min(axis=1) * 3600.0
    return pd.DataFrame(
        {
            "input_index": np.asarray(anchor_ids, dtype=np.int64),
            "true_r_magnitude": magnitude[anchor_rows],
            "neighbours_within_radius": present.sum(axis=1).astype(np.int64),
            "nearest_neighbour_arcsec": separation,
            "brightest_neighbour_delta_mag": brightest,
            "bright_neighbour": brightest > REJECTION_DELTA_MAG,
        }
    )


def aperture_pairs(predictor, primaries, truth, anchor_ids) -> pd.DataFrame:
    """Return the inference-time pair table with both members' magnitudes."""
    predicted = predictor.predict_response(primaries, truth)
    for column in ("index_input_p", "index_input_s", "response", "distance"):
        if column not in predicted:
            raise RuntimeError(f"response prediction lacks {column!r}")
    response = predicted["response"].to_numpy(np.float64)
    if not np.isfinite(response).all():
        raise RuntimeError("non-finite full-scene blending response")
    primary = predicted["index_input_p"].to_numpy(np.int64)
    secondary = predicted["index_input_s"].to_numpy(np.int64)
    unexpected = np.setdiff1d(np.unique(primary), anchor_ids, assume_unique=False)
    if len(unexpected):
        raise RuntimeError(f"{len(unexpected)} non-anchor primaries in aperture pairs")
    lookup = row_lookup(truth["index"].to_numpy(np.int64))
    magnitude = truth["r"].to_numpy(np.float64)
    return pd.DataFrame(
        {
            "input_index": primary,
            "index_input_s": secondary,
            "response": response,
            "distance": predicted["distance"].to_numpy(np.float64),
            "delta_mag": magnitude[lookup[primary]] - magnitude[lookup[secondary]],
        }
    )


def grid_histogram(distance: np.ndarray, delta_mag: np.ndarray, weight=None):
    """Bin pairs on the separation / magnitude-difference grid.

    Separations beyond the last edge are dropped rather than folded into the
    final bin, so the inference aperture and the catalogue are compared on
    exactly the same footprint.
    """
    distance = np.asarray(distance, dtype=np.float64)
    delta_mag = np.asarray(delta_mag, dtype=np.float64)
    inside = (distance >= DISTANCE_EDGES[0]) & (distance <= DISTANCE_EDGES[-1])
    counts, _, _ = np.histogram2d(
        distance[inside],
        np.clip(delta_mag[inside], -1e30, 1e30),
        bins=[np.array(DISTANCE_EDGES), np.array(DELTA_MAG_EDGES)],
        weights=None if weight is None else np.asarray(weight, np.float64)[inside],
    )
    return counts


def rejected_corner_mask(distance: np.ndarray, delta_mag: np.ndarray) -> np.ndarray:
    """Pairs the label pipeline's rejection geometry removes."""
    return (np.asarray(distance, np.float64) <= REJECTION_RADIUS_ARCSEC) & (
        np.asarray(delta_mag, np.float64) > REJECTION_DELTA_MAG
    )


def catalogue_scan(response_catalogue: Path, cases: tuple[int, ...]):
    """Collect, per case, the primaries the catalogue carries and its pair grid.

    Both the raw primaries and the ones owning at least one valid pair are
    returned, so a primary absent from the catalogue can be told apart from one
    whose every pair failed the two-leg shape rule.
    """
    allowed = np.asarray(sorted(cases), dtype=np.int64)
    present = {case: [] for case in cases}
    valid_primaries = {case: [] for case in cases}
    grids = {case: np.zeros((len(DISTANCE_EDGES) - 1, len(DELTA_MAG_EDGES) - 1)) for case in cases}
    counts = {
        case: {"case_window_pair_rows": 0, "invalid_pair_rows_dropped": 0, "valid_pair_rows": 0}
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
            valid = valid_blend_response_pairs(frame)
            case_column_values = frame["case"].to_numpy(np.int64)
            for case in np.unique(case_column_values):
                case = int(case)
                local = case_column_values == case
                counts[case]["case_window_pair_rows"] += int(local.sum())
                counts[case]["invalid_pair_rows_dropped"] += int((local & ~valid).sum())
                counts[case]["valid_pair_rows"] += int((local & valid).sum())
                present[case].append(
                    np.unique(frame.loc[local, "input_index"].to_numpy(np.int64))
                )
                keep = local & valid
                if not keep.any():
                    continue
                block = frame.loc[keep]
                valid_primaries[case].append(
                    np.unique(block["input_index"].to_numpy(np.int64))
                )
                grids[case] += grid_histogram(
                    block["distance"].to_numpy(np.float64),
                    block["r_input_p"].to_numpy(np.float64)
                    - block["r_input_s"].to_numpy(np.float64),
                )
            if (batch_index + 1) % 200 == 0:
                print(
                    f"LABEL_REJECTION_SCAN batch={batch_index + 1}/"
                    f"{reader.num_record_batches}",
                    flush=True,
                )
    scanned = {}
    for case in cases:
        if not present[case]:
            raise RuntimeError(f"case {case} has no response-catalogue rows")
        scanned[case] = {
            "present": np.unique(np.concatenate(present[case])),
            "valid": (
                np.unique(np.concatenate(valid_primaries[case]))
                if valid_primaries[case]
                else np.zeros(0, dtype=np.int64)
            ),
            "grid": grids[case],
            "counts": counts[case],
        }
    return scanned


def case_summary(primaries: pd.DataFrame) -> dict:
    """Mean response per primary class, and per class crossed with the proxy."""
    summary = {}
    blocks = {"all_anchor_primaries": primaries}
    for name in PRIMARY_CLASSES:
        blocks[name] = primaries.loc[primaries["class"] == name]
    for name in BRIGHT_CLASSES:
        blocks[name] = primaries.loc[primaries["bright_neighbour"] == (name == "bright_neighbour")]
    for primary_class in PRIMARY_CLASSES:
        for bright_class in BRIGHT_CLASSES:
            blocks[f"{primary_class}__{bright_class}"] = primaries.loc[
                (primaries["class"] == primary_class)
                & (primaries["bright_neighbour"] == (bright_class == "bright_neighbour"))
            ]
    for name, block in blocks.items():
        summary[name] = {
            "n": int(len(block)),
            "mean_R_blend": float(block["R_blend"].mean()) if len(block) else float("nan"),
            "mean_R_blend_rejected_corner": (
                float(block["R_blend_rejected_corner"].mean()) if len(block) else float("nan")
            ),
            "mean_pairs": float(block["pairs"].mean()) if len(block) else float("nan"),
            "mean_true_r_magnitude": (
                float(block["true_r_magnitude"].mean()) if len(block) else float("nan")
            ),
            # The nearest-neighbour separation is infinite for a primary with
            # no neighbour inside the rejection radius, which is most of them,
            # so the class is summarised by counts rather than by a separation.
            "mean_neighbours_within_radius": (
                float(block["neighbours_within_radius"].mean())
                if len(block)
                else float("nan")
            ),
            "fraction_bright_neighbour": (
                float(block["bright_neighbour"].mean()) if len(block) else float("nan")
            ),
        }
    return summary


def bootstrap_classes(per_case: dict[int, dict], n_boot: int, seed: int) -> dict:
    """Case bootstrap of every class mean, reported only where every case has members."""
    cases = sorted(per_case)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(cases), size=(n_boot, len(cases)))
    names = sorted({name for case in cases for name in per_case[case]})
    quantities = (
        "mean_R_blend",
        "mean_R_blend_rejected_corner",
        "mean_pairs",
        "mean_true_r_magnitude",
        "mean_neighbours_within_radius",
        "fraction_bright_neighbour",
    )
    summary = {}
    for name in names:
        counts = np.array(
            [per_case[case].get(name, {"n": 0})["n"] for case in cases], dtype=np.int64
        )
        entry = {"objects": int(counts.sum()), "cases": len(cases)}
        series = {
            quantity: np.array(
                [per_case[case].get(name, {}).get(quantity, np.nan) for case in cases],
                dtype=np.float64,
            )
            for quantity in quantities
        }
        if not np.isfinite(np.concatenate(list(series.values()))).all():
            entry["status"] = "not_evaluable_on_every_case"
            entry["cases_without_members"] = [
                int(case) for case in cases if per_case[case].get(name, {"n": 0})["n"] == 0
            ]
            summary[name] = entry
            continue
        entry["status"] = "complete"
        fractions = counts / np.array(
            [per_case[case]["all_anchor_primaries"]["n"] for case in cases], dtype=np.float64
        )
        entry["fraction_of_anchor"] = {
            "mean": float(fractions.mean()),
            "standard_error": float(fractions[draws].mean(axis=1).std(ddof=1)),
        }
        for quantity, values in series.items():
            entry[quantity] = {
                "mean": float(values.mean()),
                "standard_error": float(values[draws].mean(axis=1).std(ddof=1)),
            }
        summary[name] = entry
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("label-rejection diagnosis must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if len(cases) != len(args.case):
        raise ValueError("cases must be unique")
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
    scanned = catalogue_scan(args.response_catalogue, cases)

    per_case = {}
    case_reports = {}
    aperture_grid = np.zeros((len(DISTANCE_EDGES) - 1, len(DELTA_MAG_EDGES) - 1))
    aperture_response_grid = np.zeros_like(aperture_grid)
    catalogue_grid = np.zeros_like(aperture_grid)
    for case in cases:
        print(f"LABEL_REJECTION_START case={case}", flush=True)
        anchor_ids = anchor_by_case[case]
        truth = prepare_truth(Path(args.input_pattern.format(case=case)))
        aligned = truth.set_index("index", drop=False).reindex(anchor_ids)
        if aligned["index"].isna().any():
            raise RuntimeError(f"case {case} anchor contains identities absent from truth")
        primaries_truth = aligned.reset_index(drop=True)

        pairs = aperture_pairs(predictor, primaries_truth, truth, anchor_ids)
        corner = rejected_corner_mask(pairs["distance"], pairs["delta_mag"])
        aperture_grid += grid_histogram(pairs["distance"], pairs["delta_mag"])
        aperture_response_grid += grid_histogram(
            pairs["distance"], pairs["delta_mag"], weight=pairs["response"]
        )
        catalogue_grid += scanned[case]["grid"]

        aggregated = pairs.groupby("input_index", sort=True)["response"].agg(
            ["sum", "size"]
        )
        table = pd.DataFrame(
            {
                "input_index": aggregated.index.to_numpy(np.int64),
                "R_blend": aggregated["sum"].to_numpy(np.float64),
                "pairs": aggregated["size"].to_numpy(np.int64),
            }
        )
        corner_sum = (
            pairs.loc[corner]
            .groupby("input_index", sort=True)["response"]
            .sum()
            .reindex(table["input_index"], fill_value=0.0)
            .to_numpy(np.float64)
        )
        table["R_blend_rejected_corner"] = corner_sum

        audit = bright_neighbour_audit(truth, anchor_ids)
        table = table.merge(audit, on="input_index", how="right", validate="one_to_one")
        # An anchor primary with no aperture pair at all still belongs to the
        # cohort and contributes zero blending response; keep it rather than
        # letting the merge silently shrink the denominator.
        table[["R_blend", "R_blend_rejected_corner"]] = table[
            ["R_blend", "R_blend_rejected_corner"]
        ].fillna(0.0)
        table["pairs"] = table["pairs"].fillna(0).astype(np.int64)

        labelled = np.isin(table["input_index"].to_numpy(np.int64), scanned[case]["valid"])
        in_catalogue = np.isin(
            table["input_index"].to_numpy(np.int64), scanned[case]["present"]
        )
        table["class"] = np.where(labelled, "labelled", "unlabelled")
        per_case[case] = case_summary(table)
        case_reports[str(case)] = {
            "anchor_rows": int(len(anchor_ids)),
            "aperture_pair_rows": int(len(pairs)),
            "catalogue": scanned[case]["counts"],
            "primaries_absent_from_catalogue": int((~in_catalogue).sum()),
            "primaries_in_catalogue_without_valid_pair": int(
                (in_catalogue & ~labelled).sum()
            ),
            "unlabelled_primaries": int((~labelled).sum()),
            "unlabelled_with_bright_neighbour": int(
                (~labelled & table["bright_neighbour"].to_numpy(bool)).sum()
            ),
            "labelled_with_bright_neighbour": int(
                (labelled & table["bright_neighbour"].to_numpy(bool)).sum()
            ),
        }
        print(
            f"LABEL_REJECTION_DONE case={case} "
            f"unlabelled={case_reports[str(case)]['unlabelled_primaries']} "
            f"absent={case_reports[str(case)]['primaries_absent_from_catalogue']} "
            f"unlabelled_bright={case_reports[str(case)]['unlabelled_with_bright_neighbour']} "
            f"R_blend={table['R_blend'].mean():.6f} "
            f"corner={table['R_blend_rejected_corner'].mean():.6f}",
            flush=True,
        )

    corner_rows = slice(0, int(np.searchsorted(np.array(DISTANCE_EDGES), REJECTION_RADIUS_ARCSEC)))
    corner_columns = slice(
        int(np.searchsorted(np.array(DELTA_MAG_EDGES), REJECTION_DELTA_MAG)), None
    )
    result = {
        "format_version": 1,
        "purpose": (
            "support audit: the inference-time R_blend carried by pairs the "
            "BlendEMU label pipeline's bright-neighbour rejection removes"
        ),
        "cases": list(cases),
        "rejection_rule": {
            "source": "blendemu.utils.remove_detection_w_bright_neighbour via "
            "blendemu.response.retrieve_response",
            "radius_arcsec": REJECTION_RADIUS_ARCSEC,
            "flux_ratio_max": REJECTION_FLUX_RATIO,
            "delta_magnitude": REJECTION_DELTA_MAG,
            "neighbours": REJECTION_NEIGHBOURS,
            "applied": "independently in each of the two response shear legs",
            "proxy": "true flux at true position; the pipeline uses measured "
            "FLUX_AUTO at the detected position",
        },
        "grid": {
            "distance_edges_arcsec": list(DISTANCE_EDGES),
            "delta_magnitude_edges": list(DELTA_MAG_EDGES),
            "delta_magnitude_sign": "true r of primary minus true r of neighbour; "
            "positive means the neighbour is brighter",
            "aperture_pair_counts": aperture_grid,
            "aperture_response_sum": aperture_response_grid,
            "catalogue_valid_pair_counts": catalogue_grid,
        },
        "totals": {
            "aperture_pairs": float(aperture_grid.sum()),
            "aperture_response": float(aperture_response_grid.sum()),
            "aperture_pairs_rejected_corner": float(
                aperture_grid[corner_rows, corner_columns].sum()
            ),
            "aperture_response_rejected_corner": float(
                aperture_response_grid[corner_rows, corner_columns].sum()
            ),
            "catalogue_pairs": float(catalogue_grid.sum()),
            "catalogue_pairs_rejected_corner": float(
                catalogue_grid[corner_rows, corner_columns].sum()
            ),
        },
        "inputs": {
            "response_catalogue": str(args.response_catalogue),
            "response_catalogue_sha256": file_sha256(args.response_catalogue),
            "emulator_model": str(args.emulator_model),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "emulator_metadata": str(args.emulator_metadata),
            "emulator_metadata_sha256": file_sha256(args.emulator_metadata),
            "measurement_model_unused_for_R_blend": str(args.measurement_model),
        },
        "limitations": [
            "The rejection proxy uses true flux and true position; the pipeline "
            "rejects on measured FLUX_AUTO at the detected position, so class "
            "membership is approximate at the threshold.",
            "A primary can also be unlabelled because it was undetected in a "
            "response leg; this diagnostic does not separate that route.",
            "The catalogue neighbour search runs over the sheared half alone "
            "while the inference aperture searches the whole scene, so the two "
            "pair grids share a footprint but not a selection.",
            "Case bootstrap over the supplied cases only; one emulator, one seed.",
        ],
        "case_reports": case_reports,
        "summary": bootstrap_classes(per_case, args.n_boot, args.bootstrap_seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, output)
    print(f"LABEL_REJECTION_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
