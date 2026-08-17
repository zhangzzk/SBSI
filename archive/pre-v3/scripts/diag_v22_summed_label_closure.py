"""Close V2.2 predictions against its own held-out pair labels after summing.

The response catalogue gives each primary one row per supported secondary.  A
row's label is the primary shape change projected onto that secondary's random
shear direction.  Cross-neighbour projections average to zero, so summing the
labels over a primary is the training-side analogue of deployed ``R_blend``.

Only cases below ``heldout_min`` are reported; V2.2 was trained on cases >=40.
No coherent-anchor or constgold quantity is read by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from sbs_shear.domain import in_domain  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
NEED = ["case", "input_index", "delta_et1", "delta_et2", *FEATURES]
POSITION_COLUMNS = [
    "RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s",
]


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def input_frame_distance(frame: pd.DataFrame) -> np.ndarray:
    """Great-circle input-to-input pair separation in arcseconds."""
    ra_p = np.deg2rad(frame["RA_input_p"].to_numpy(float))
    dec_p = np.deg2rad(frame["DEC_input_p"].to_numpy(float))
    ra_s = np.deg2rad(frame["RA_input_s"].to_numpy(float))
    dec_s = np.deg2rad(frame["DEC_input_s"].to_numpy(float))
    delta_ra = ra_s - ra_p
    delta_dec = dec_s - dec_p
    haversine = (
        np.sin(delta_dec / 2.0) ** 2
        + np.cos(dec_p) * np.cos(dec_s) * np.sin(delta_ra / 2.0) ** 2
    )
    radians = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return np.rad2deg(radians) * 3600.0


def summarize(case_table: pd.DataFrame) -> dict:
    label = case_table["label_sum_mean"].to_numpy(float)
    prediction = case_table["prediction_sum_mean"].to_numpy(float)
    null = case_table["null_sum_mean"].to_numpy(float)
    gap = prediction - label
    return {
        "n_cases": int(len(case_table)),
        "n_primaries": int(case_table["n_primaries"].sum()),
        "n_pairs": int(case_table["n_pairs"].sum()),
        "mean_pairs_per_primary": float(
            case_table["n_pairs"].sum() / case_table["n_primaries"].sum()
        ),
        "label_sum_mean": float(label.mean()),
        "label_case_sem": sem(label),
        "prediction_sum_mean": float(prediction.mean()),
        "prediction_case_sem": sem(prediction),
        "prediction_minus_label": float(gap.mean()),
        "gap_case_sem": sem(gap),
        "prediction_over_label_minus_one": float(prediction.mean() / label.mean() - 1.0),
        "null_sum_mean": float(null.mean()),
        "null_case_sem": sem(null),
    }


def summarize_component(case_table: pd.DataFrame, prefix: str) -> dict:
    """Summarize one additive pair subset using all eligible primaries.

    The component columns are already sums divided by the total eligible-primary
    count in each case.  Keeping that denominator makes disjoint components add
    exactly to the all-pair result.
    """
    label = case_table[f"{prefix}_label_sum_mean"].to_numpy(float)
    prediction = case_table[f"{prefix}_prediction_sum_mean"].to_numpy(float)
    null = case_table[f"{prefix}_null_sum_mean"].to_numpy(float)
    gap = prediction - label
    return {
        "n_cases": int(len(case_table)),
        "n_pairs": int(case_table[f"{prefix}_n_pairs"].sum()),
        "mean_pairs_per_primary": float(
            case_table[f"{prefix}_n_pairs"].sum()
            / case_table["n_primaries"].sum()
        ),
        "label_sum_mean": float(label.mean()),
        "label_case_sem": sem(label),
        "prediction_sum_mean": float(prediction.mean()),
        "prediction_case_sem": sem(prediction),
        "label_minus_prediction": float((-gap).mean()),
        "prediction_minus_label": float(gap.mean()),
        "gap_case_sem": sem(gap),
        "null_sum_mean": float(null.mean()),
        "null_case_sem": sem(null),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--heldout-min", type=int, default=40)
    ap.add_argument(
        "--case-min", type=int, default=0,
        help="first case to summarize (default preserves the original held-out window)",
    )
    ap.add_argument(
        "--case-max", type=int, default=None,
        help="last case to summarize; defaults to heldout-min minus one",
    )
    ap.add_argument("--shear", type=float, default=0.2)
    ap.add_argument(
        "--split-distance", type=float, nargs=2, metavar=("LO", "HI"),
        help=(
            "also report additive sums for LO <= stored distance < HI and "
            "for every supported pair outside that interval"
        ),
    )
    ap.add_argument(
        "--split-coordinate", choices=("stored", "input"), default="stored",
        help=(
            "distance coordinate used only for the optional split; pair support "
            "always follows V2.2's stored training coordinate"
        ),
    )
    ap.add_argument(
        "--v21-domain", action="store_true",
        help="intersect primaries with the anchor's intrinsic Re/SNR domain",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING existing {args.output}")

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu"
    )
    cuts, _, _ = predictor._select("regression")
    if len(cuts) != 5:
        raise RuntimeError(f"unexpected regression cuts: {cuts}")

    case_max = args.heldout_min - 1 if args.case_max is None else args.case_max
    if args.case_min < 0 or case_max < args.case_min:
        raise ValueError("invalid case window")
    if args.split_distance is not None:
        split_lo, split_hi = args.split_distance
        if not (0.0 <= split_lo < split_hi):
            raise ValueError("split-distance requires 0 <= LO < HI")
    else:
        split_lo = split_hi = None
    accum = {}
    for case in range(args.case_min, case_max + 1):
        row = {
            "primary": set(), "pairs": 0, "label": 0.0,
            "prediction": 0.0, "null": 0.0,
        }
        if args.split_distance is not None:
            for prefix in ("inside", "outside"):
                row.update({
                    f"{prefix}_pairs": 0,
                    f"{prefix}_label": 0.0,
                    f"{prefix}_prediction": 0.0,
                    f"{prefix}_null": 0.0,
                })
        accum[case] = row
    previous_min = -1
    n_batches = 0
    read_columns = [*NEED]
    if args.split_distance is not None and args.split_coordinate == "input":
        read_columns.extend(POSITION_COLUMNS)
    with ipc.open_file(args.catalogue) as reader:
        missing = set(read_columns) - set(reader.schema.names)
        if missing:
            raise KeyError(f"catalogue missing columns: {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                read_columns
            ).to_pandas()
            if len(frame) == 0:
                continue
            batch_min = int(frame["case"].min())
            if batch_min < previous_min:
                raise RuntimeError("catalogue cases are not monotonically ordered")
            previous_min = batch_min
            frame = frame[frame["case"].between(args.case_min, case_max)]
            if len(frame) == 0:
                if batch_min > case_max:
                    break
                continue
            n_batches += 1
            finite_label = np.isfinite(frame[["delta_et1", "delta_et2"]].to_numpy(float)).all(axis=1)
            frame = frame.loc[finite_label]
            # Count every eligible held-out primary before secondary cuts.  A
            # primary with no supported pair has a deployed sum of exactly zero.
            primary_ok = (
                (frame["r_input_p"] > cuts[1][0]) & (frame["r_input_p"] < cuts[1][1])
                & (frame["Re_input_p"] > cuts[3][0]) & (frame["Re_input_p"] < cuts[3][1])
            )
            if args.v21_domain:
                primary_ok &= in_domain(
                    frame["r_input_p"].to_numpy(float),
                    frame["Re_input_p"].to_numpy(float),
                )
            eligible = frame.loc[primary_ok]
            for case, values in eligible.groupby("case", sort=False):
                accum[int(case)]["primary"].update(values["input_index"].to_numpy(np.int64).tolist())

            pair_ok = primary_ok & (
                (frame["r_input_s"] > cuts[0][0]) & (frame["r_input_s"] < cuts[0][1])
                & (frame["Re_input_s"] > cuts[2][0]) & (frame["Re_input_s"] < cuts[2][1])
                & (frame["distance"] > cuts[4][0]) & (frame["distance"] < cuts[4][1])
            )
            pairs = frame.loc[pair_ok].copy()
            if len(pairs) == 0:
                continue
            prediction = predictor.predict_on_pairs(
                pairs[FEATURES], task="response"
            )["response"].to_numpy(float)
            label = pairs["delta_et1"].to_numpy(float) / args.shear
            null = pairs["delta_et2"].to_numpy(float) / args.shear
            if args.split_distance is not None:
                split_values = (
                    pairs["distance"].to_numpy(float)
                    if args.split_coordinate == "stored"
                    else input_frame_distance(pairs)
                )
                inside = (
                    (split_values >= split_lo) & (split_values < split_hi)
                )
            for case, positions in pairs.groupby("case", sort=False).indices.items():
                positions = np.asarray(positions, dtype=int)
                row = accum[int(case)]
                row["pairs"] += int(len(positions))
                row["label"] += float(label[positions].sum())
                row["prediction"] += float(prediction[positions].sum())
                row["null"] += float(null[positions].sum())
                if args.split_distance is not None:
                    for prefix, mask in (
                        ("inside", inside[positions]),
                        ("outside", ~inside[positions]),
                    ):
                        selected = positions[mask]
                        row[f"{prefix}_pairs"] += int(len(selected))
                        row[f"{prefix}_label"] += float(label[selected].sum())
                        row[f"{prefix}_prediction"] += float(
                            prediction[selected].sum()
                        )
                        row[f"{prefix}_null"] += float(null[selected].sum())
            if n_batches % 100 == 0:
                print(f"processed {n_batches} held-out batches", flush=True)

    rows = []
    for case, values in accum.items():
        n_primary = len(values["primary"])
        if n_primary == 0:
            raise RuntimeError(f"held-out case {case} has no eligible primaries")
        output_row = {
            "case": case,
            "n_primaries": n_primary,
            "n_pairs": values["pairs"],
            "label_sum_mean": values["label"] / n_primary,
            "prediction_sum_mean": values["prediction"] / n_primary,
            "null_sum_mean": values["null"] / n_primary,
        }
        if args.split_distance is not None:
            for prefix in ("inside", "outside"):
                output_row.update({
                    f"{prefix}_n_pairs": values[f"{prefix}_pairs"],
                    f"{prefix}_label_sum_mean": (
                        values[f"{prefix}_label"] / n_primary
                    ),
                    f"{prefix}_prediction_sum_mean": (
                        values[f"{prefix}_prediction"] / n_primary
                    ),
                    f"{prefix}_null_sum_mean": (
                        values[f"{prefix}_null"] / n_primary
                    ),
                })
        rows.append(output_row)
    case_table = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    result = {
        "tag": args.tag,
        "catalogue": args.catalogue,
        "case_window": [args.case_min, case_max],
        "heldout_cases": [args.case_min, case_max],
        "shear": args.shear,
        "v21_domain": bool(args.v21_domain),
        "regression_cuts": cuts,
        "summary": summarize(case_table),
        "case_table": case_table.to_dict(orient="records"),
    }
    if args.split_distance is not None:
        result["distance_split"] = {
            "coordinate": (
                (
                    "stored response-catalogue distance used by V2.2 training"
                    if args.split_coordinate == "stored"
                    else "input-to-input great-circle distance matching inference"
                )
                + "; lower-inclusive and upper-exclusive"
            ),
            "inside_interval_arcsec": [split_lo, split_hi],
            "inside": summarize_component(case_table, "inside"),
            "outside": summarize_component(case_table, "outside"),
        }
        for quantity in (
            "n_pairs", "label_sum_mean", "prediction_sum_mean",
            "prediction_minus_label", "null_sum_mean",
        ):
            all_value = result["summary"][quantity]
            component_value = sum(
                result["distance_split"][name][quantity]
                for name in ("inside", "outside")
            )
            if not np.isclose(all_value, component_value, rtol=0.0, atol=2e-12):
                raise RuntimeError(
                    f"distance components fail to replay {quantity}: "
                    f"{component_value} != {all_value}"
                )
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}\nV22_SUMMED_LABEL_CLOSURE_DONE", flush=True)


if __name__ == "__main__":
    main()
