#!/usr/bin/env python3
"""Ask where the inference-time R_blend comes from, and whether labels cover it.

At inference ``R_blend`` for a primary is the emulator summed over every truth
neighbour inside the trained aperture, detected or not.  A label, by contrast,
exists only for a pair whose secondary was itself detected and measured, so the
faint end of the neighbour population contributes to the sum but can never
appear in training.  The aperture audit already shows the sum runs over about
16.2 pairs per primary while only about 8.5 carry a label.

This reports the summed response and the pair count in bands of neighbour
magnitude and of magnitude difference, for the ConstGold scenes, beside the same
bands taken over the response catalogue the emulator was trained on.  It
measures how much of the global blending prediction sits outside the labelled
support; it changes nothing and recommends nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pyarrow_dataset

from sbsi.models import ModelPaths, load_emulator

from scripts.diagnose_corner_extrapolation import (
    PRIMARY_MAG_EDGES as CORNER_PRIMARY_MAG_EDGES,
)
from scripts.build_constgold_fixed_g0_blend_lookup import (
    CONDITIONS,
    file_sha256,
    load_anchor,
    prepare_truth,
    verify_anchor_truth,
)

#  delta_mag = r of primary minus r of neighbour; positive means the neighbour
#  is the brighter of the two.  1.7474 is the flux ratio of five the label
#  pipeline's bright-neighbour guard rejects on.
DELTA_MAG_EDGES = (-np.inf, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                   1.747425010840047, 3.0, np.inf)
NEIGHBOUR_MAG_EDGES = (-np.inf, 22.0, 24.0, 25.0, 25.8, 26.5, 27.5, np.inf)
DISTANCE_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, np.inf)
#  the label pipeline's bright-neighbour guard refuses a pair whose neighbour is
#  more than five times brighter within three arcseconds.  These four cells say
#  how much of the inference-time sum sits in that refused configuration.
GUARD_RATIO_DELTA_MAG = 1.747425010840047
GUARD_RADIUS_ARCSEC = 3.0
GUARD_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0)
GUARD_NAMES = ("faint_or_far", "bright_and_far", "faint_and_near", "bright_and_near")
#  the same primary-magnitude bands the corner diagnostic resolves its accuracy
#  on, imported rather than restated so the two tables can be multiplied
PRIMARY_MAG_EDGES = tuple(CORNER_PRIMARY_MAG_EDGES)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--response-catalogue", type=Path, required=True,
                        help="the labelled catalogue the emulator was trained on")
    parser.add_argument("--catalogue-cases", type=int, default=20,
                        help="how many response-catalogue cases to read for the "
                             "label-side distribution")
    parser.add_argument("--response-shear", type=float, default=0.2,
                        help="delta_et1 is divided by this to match the emulator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def banded(values: np.ndarray, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), np.asarray(edges[1:-1]),
                       right=False)


def guard_cell(delta_mag, distance) -> np.ndarray:
    """0-3 index over (neighbour brighter than the guard ratio) x (inside 3")."""
    bright = np.asarray(delta_mag, dtype=float) > GUARD_RATIO_DELTA_MAG
    near = np.asarray(distance, dtype=float) < GUARD_RADIUS_ARCSEC
    return (bright.astype(np.int64) + 2 * near.astype(np.int64)).astype(float) + 0.5


def band_names(edges) -> list[str]:
    if tuple(edges) == GUARD_EDGES:
        return list(GUARD_NAMES)
    names = []
    for low, high in zip(edges[:-1], edges[1:]):
        low_text = "-inf" if low == -np.inf else f"{low:g}"
        high_text = "inf" if high == np.inf else f"{high:g}"
        names.append(f"[{low_text},{high_text})")
    return names


def accumulate(table: dict, key: str, edges, values, response) -> None:
    """Add one case's pairs into the running band totals for ``key``."""
    index = banded(values, edges)
    counts = np.bincount(index, minlength=len(edges) - 1).astype(np.float64)
    sums = np.bincount(index, weights=response, minlength=len(edges) - 1)
    absolute = np.bincount(index, weights=np.abs(response), minlength=len(edges) - 1)
    entry = table.setdefault(key, {"pairs": np.zeros(len(edges) - 1),
                                   "response": np.zeros(len(edges) - 1),
                                   "absolute_response": np.zeros(len(edges) - 1)})
    entry["pairs"] += counts
    entry["response"] += sums
    entry["absolute_response"] += absolute


def report(table: dict, edges) -> list[dict]:
    pairs = table["pairs"]
    response = table["response"]
    absolute = table["absolute_response"]
    pair_total = pairs.sum()
    response_total = response.sum()
    absolute_total = absolute.sum()
    rows = []
    for name, n, r, a in zip(band_names(edges), pairs, response, absolute):
        rows.append({
            "band": name,
            "pairs": float(n),
            "pair_share": float(n / pair_total) if pair_total else None,
            "summed_response": float(r),
            "response_share": float(r / response_total) if response_total else None,
            "summed_absolute_response": float(a),
            "absolute_share": float(a / absolute_total) if absolute_total else None,
        })
    return rows


def label_side_bands(path: Path, cases: int, response_shear: float) -> dict:
    """The same bands over the labelled catalogue, from its first ``cases`` cases."""
    dataset = pyarrow_dataset.dataset(path, format="feather")
    columns = ["case", "r_input_p", "r_input_s", "distance", "delta_et1"]
    tables = {}
    rows = 0
    for batch in dataset.to_batches(columns=columns):
        frame = batch.to_pandas()
        #  the catalogue is written case by case, so once a whole batch sits
        #  above the window every later batch does too
        if rows and int(frame["case"].min()) >= cases:
            break
        frame = frame.loc[frame["case"] < cases]
        frame = frame.loc[np.isfinite(frame["delta_et1"].to_numpy(np.float64))]
        if frame.empty:
            continue
        rows += len(frame)
        response = frame["delta_et1"].to_numpy(np.float64) / response_shear
        primary = frame["r_input_p"].to_numpy(np.float64)
        secondary = frame["r_input_s"].to_numpy(np.float64)
        distance = frame["distance"].to_numpy(np.float64)
        accumulate(tables, "delta_magnitude", DELTA_MAG_EDGES, primary - secondary,
                   response)
        accumulate(tables, "neighbour_magnitude", NEIGHBOUR_MAG_EDGES, secondary,
                   response)
        accumulate(tables, "distance_arcsec", DISTANCE_EDGES, distance, response)
        accumulate(tables, "guard_region", GUARD_EDGES,
                   guard_cell(primary - secondary, distance), response)
        accumulate(tables, "primary_magnitude", PRIMARY_MAG_EDGES, primary, response)
    if not rows:
        raise RuntimeError(f"no labelled rows below case {cases} in {path}")
    return {"rows": rows, "bands": tables}


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))

    paths = ModelPaths(
        flow_checkpoints=(args.measurement_model,),
        emulator_metadata=args.emulator_metadata,
        emulator_model=args.emulator_model,
    )
    paths.validate(verify_hashes=False)
    predictor = load_emulator(paths, conditions=CONDITIONS, device=args.device)

    scene = {}
    per_case = {}
    for case in cases:
        anchor_ids, anchor_context = load_anchor(Path(args.anchor_pattern.format(case=case)))
        truth = prepare_truth(Path(args.input_pattern.format(case=case)))
        primaries = verify_anchor_truth(truth, anchor_ids, anchor_context)
        pairs = predictor.predict_response(primaries, truth)
        missing = {"r_input_p", "r_input_s", "distance", "response"} - set(pairs)
        if missing:
            raise RuntimeError(f"pair frame lacks {sorted(missing)}")
        response = pairs["response"].to_numpy(np.float64)
        if not np.isfinite(response).all():
            raise RuntimeError(f"non-finite predicted response in case {case}")
        primary_mag = pairs["r_input_p"].to_numpy(np.float64)
        secondary_mag = pairs["r_input_s"].to_numpy(np.float64)
        accumulate(scene, "delta_magnitude", DELTA_MAG_EDGES,
                   primary_mag - secondary_mag, response)
        accumulate(scene, "neighbour_magnitude", NEIGHBOUR_MAG_EDGES, secondary_mag,
                   response)
        distance = pairs["distance"].to_numpy(np.float64)
        accumulate(scene, "distance_arcsec", DISTANCE_EDGES, distance, response)
        accumulate(scene, "guard_region", GUARD_EDGES,
                   guard_cell(primary_mag - secondary_mag, distance), response)
        accumulate(scene, "primary_magnitude", PRIMARY_MAG_EDGES, primary_mag, response)
        per_case[str(case)] = {
            "anchor_primaries": int(len(anchor_ids)),
            "truth_rows": int(len(truth)),
            "pair_rows": int(len(pairs)),
            "pairs_per_primary": float(len(pairs) / len(anchor_ids)),
            "summed_response_per_primary": float(response.sum() / len(anchor_ids)),
        }
        print(f"NEIGHBOUR_SUPPORT_CASE_DONE case={case} pairs={len(pairs)} "
              f"R_blend_mean={response.sum() / len(anchor_ids):.6f}", flush=True)

    labels = label_side_bands(args.response_catalogue, args.catalogue_cases,
                              args.response_shear)

    edges_by_key = {"delta_magnitude": DELTA_MAG_EDGES,
                    "neighbour_magnitude": NEIGHBOUR_MAG_EDGES,
                    "distance_arcsec": DISTANCE_EDGES,
                    "guard_region": GUARD_EDGES,
                    "primary_magnitude": PRIMARY_MAG_EDGES}
    result = {
        "format_version": 1,
        "purpose": ("where the inference-time R_blend sum comes from, in bands of "
                    "neighbour magnitude and magnitude difference, beside the same "
                    "bands over the labelled training catalogue"),
        "delta_magnitude_sign": ("r of primary minus r of neighbour; positive means "
                                 "the neighbour is brighter"),
        "cases": list(cases),
        "inputs": {
            "emulator_model": str(args.emulator_model),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "response_catalogue": str(args.response_catalogue),
            "catalogue_cases_read": int(args.catalogue_cases),
            "response_shear": float(args.response_shear),
        },
        "per_case": per_case,
        "scene_bands": {key: report(scene[key], edges_by_key[key]) for key in scene},
        "label_bands": {key: report(labels["bands"][key], edges_by_key[key])
                        for key in labels["bands"]},
        "label_rows": labels["rows"],
        "limitations": [
            "The scene side is the emulator's own prediction, not a measurement; "
            "it says where the prediction comes from, not whether it is right.",
            "The label side is a pair count weighted by a noisy measured response, "
            "so its response shares are noisier than the scene side's.",
            "Neighbour magnitude is a truth magnitude, not a detection flag; it is "
            "a proxy for detectability, not the pipeline's actual detection.",
            "The scene side runs over ConstGold scenes and fixed-g0 anchor "
            "primaries; the label side runs over half-shear cases and the whole "
            "catalogue population. Band SHAPES are comparable, absolute per-pair "
            "means are not, because the two primary populations differ.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"NEIGHBOUR_SUPPORT_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
