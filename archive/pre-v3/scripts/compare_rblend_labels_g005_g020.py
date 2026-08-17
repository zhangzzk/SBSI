"""Compare exact matched pair-response labels at g=0.05 and g=0.20.

Both catalogues use the same g=0 reference geometry.  The comparison is made
inside the V2.2 regression domain on a common case window, pair-matched by primary and
secondary input coordinates.  Responses are first averaged within each
rendered case; case means are the independent replicates used for errors.
Constgold, emulator predictions, and coherent-anchor truth are never read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


SHEAR05 = 0.05
SHEAR20 = 0.20
POSITION_ROUND = 7
COLUMNS = [
    "case", "input_index", "RA_input_s", "DEC_input_s", "distance",
    "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
    "delta_et1", "delta_et2",
]
CUTS = {
    "r_input_s": (13.0, 29.0),
    "r_input_p": (18.0, 25.8),
    "Re_input_s": (0.0, 10.0),
    "Re_input_p": (0.5, 1.5),
    "distance": (0.0, 10.0),
}
AXES = {
    "separation": ("distance", np.asarray([0, .25, .5, .75, 1, 1.5, 2, 3, 5, 7, 10.])),
    "primary_mag": ("r_input_p", np.asarray([18, 22, 23, 24, 25, 25.4, 25.8])),
    "primary_size": ("Re_input_p", np.asarray([.5, .6, .75, .9, 1.1, 1.3, 1.5])),
    "secondary_mag": ("r_input_s", np.asarray([13, 18, 22, 24, 26, 28, 29.])),
    "secondary_minus_primary_mag": (
        "dmag", np.asarray([-16, -5, -2, -1, 0, 1, 2, 5, 16.]),
    ),
}


def sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values):
    values = np.asarray(values, float)
    error = sem(values)
    mean = float(values.mean())
    return {
        "mean": mean, "case_sem": error, "n_cases": int(len(values)),
        "normal_95_interval": [mean - 1.96 * error, mean + 1.96 * error],
    }


def jackknife_ratio(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    n = len(numerator)
    estimates = np.asarray([
        np.delete(numerator, index).mean() / np.delete(denominator, index).mean() - 1.0
        for index in range(n)
    ])
    estimate = float(numerator.mean() / denominator.mean() - 1.0)
    error = float(np.sqrt((n - 1) / n * np.square(estimates - estimates.mean()).sum()))
    return {
        "value": estimate, "delete_one_case_jackknife_sem": error,
        "normal_95_interval": [estimate - 1.96 * error, estimate + 1.96 * error],
    }


def iter_cases(path, max_case=99):
    """Yield monotonically ordered case frames without loading a full catalogue."""
    with ipc.open_file(path) as reader:
        missing = set(COLUMNS) - set(reader.schema.names)
        if missing:
            raise KeyError(f"{path} lacks {sorted(missing)}")
        current = None
        chunks = []
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(COLUMNS).to_pandas()
            for case in sorted(frame["case"].astype(int).unique()):
                if case > max_case:
                    if current is not None and chunks:
                        yield current, pd.concat(chunks, ignore_index=True)
                    return
                part = frame[frame["case"].to_numpy(int) == case]
                if current is None:
                    current = case
                if case < current:
                    raise RuntimeError(f"{path}: cases are not monotonic")
                if case != current:
                    yield current, pd.concat(chunks, ignore_index=True)
                    current = case
                    chunks = []
                chunks.append(part)
        if current is not None and chunks and current <= max_case:
            yield current, pd.concat(chunks, ignore_index=True)


def iter_catalogues(paths, max_case):
    """Yield one monotonic case sequence from non-overlapping catalogue shards."""
    previous = -1
    for path in paths:
        for case, frame in iter_cases(path, max_case=max_case):
            if case <= previous:
                raise RuntimeError(
                    f"catalogue shards overlap or are out of order: case {case} after {previous}"
                )
            previous = case
            yield case, frame


def select(frame):
    mask = np.isfinite(frame[["delta_et1", "delta_et2"]].to_numpy(float)).all(axis=1)
    for column, (lo, hi) in CUTS.items():
        values = frame[column].to_numpy(float)
        mask &= (values > lo) & (values < hi)
    output = frame.loc[mask].copy()
    output["_ra_s"] = np.round(output["RA_input_s"].to_numpy(float), POSITION_ROUND)
    output["_dec_s"] = np.round(output["DEC_input_s"].to_numpy(float), POSITION_ROUND)
    output["dmag"] = output["r_input_s"] - output["r_input_p"]
    return output


def unique_pairs(frame, label):
    keys = ["input_index", "_ra_s", "_dec_s"]
    duplicate = frame.duplicated(keys, keep=False)
    duplicate_rows = int(duplicate.sum())
    if duplicate_rows:
        grouped = frame.loc[duplicate].groupby(keys, sort=False)[
            ["delta_et1", "delta_et2", "distance"]
        ].nunique(dropna=False)
        if (grouped.to_numpy() > 1).any():
            raise RuntimeError(f"{label}: non-identical duplicate pair keys")
        frame = frame.drop_duplicates(keys)
    return frame, duplicate_rows


def match_case(case, frame05, frame20):
    frame05, duplicate05 = unique_pairs(select(frame05), "g005")
    frame20, duplicate20 = unique_pairs(select(frame20), "g020")
    keys = ["input_index", "_ra_s", "_dec_s"]
    shared = ["distance", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "dmag"]
    left = frame05[keys + shared + ["delta_et1", "delta_et2"]]
    right = frame20[keys + ["distance", "delta_et1", "delta_et2"]]
    matched = left.merge(
        right, on=keys, how="inner", validate="one_to_one", suffixes=("_05", "_20"),
    )
    if len(matched) < 1000:
        raise RuntimeError(f"case {case}: only {len(matched)} exact pairs")
    distance_error = np.abs(
        matched["distance_05"].to_numpy(float) - matched["distance_20"].to_numpy(float)
    )
    if distance_error.max() > 1.0e-9:
        raise RuntimeError(f"case {case}: geometry differs by {distance_error.max():.3e}")
    matched = matched.rename(columns={"distance_05": "distance"})
    matched["R05"] = matched["delta_et1_05"].to_numpy(float) / SHEAR05
    matched["R20"] = matched["delta_et1_20"].to_numpy(float) / SHEAR20
    matched["N05"] = matched["delta_et2_05"].to_numpy(float) / SHEAR05
    matched["N20"] = matched["delta_et2_20"].to_numpy(float) / SHEAR20
    return matched, {
        "case": case, "selected_g005": len(frame05), "selected_g020": len(frame20),
        "matched": len(matched), "duplicate_rows_g005": duplicate05,
        "duplicate_rows_g020": duplicate20,
    }


def binned_case_rows(case, matched):
    rows = []
    for axis, (column, edges) in AXES.items():
        values = matched[column].to_numpy(float)
        for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (values >= lo) & (values < hi)
            if not mask.any():
                continue
            rows.append({
                "case": case, "axis": axis, "bin": index,
                "lo": float(lo), "hi": float(hi), "n_pairs": int(mask.sum()),
                "R05": float(matched.loc[mask, "R05"].mean()),
                "R20": float(matched.loc[mask, "R20"].mean()),
                "R05_minus_R20": float((matched.loc[mask, "R05"] - matched.loc[mask, "R20"]).mean()),
            })
    return rows


def aggregate_bins(case_bins):
    output = []
    for (axis, index), group in case_bins.groupby(["axis", "bin"], sort=False):
        output.append({
            "axis": axis, "bin": int(index), "lo": float(group["lo"].iloc[0]),
            "hi": float(group["hi"].iloc[0]), "n_pairs": int(group["n_pairs"].sum()),
            "R05": stat(group["R05"]), "R20": stat(group["R20"]),
            "R05_minus_R20": stat(group["R05_minus_R20"]),
        })
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g005", required=True, nargs="+")
    ap.add_argument("--g020", required=True, nargs="+")
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-csv", required=True)
    ap.add_argument("--bin-csv", required=True)
    args = ap.parse_args()
    for path in (args.output, args.case_csv, args.bin_csv):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    if args.max_case < 0:
        raise ValueError("max-case must be non-negative")
    cases05 = iter_catalogues(args.g005, args.max_case)
    cases20 = iter_catalogues(args.g020, args.max_case)
    case_rows = []
    bin_rows = []
    audits = []
    for expected, ((case05, frame05), (case20, frame20)) in enumerate(zip(cases05, cases20)):
        if case05 != expected or case20 != expected:
            raise RuntimeError(
                f"expected case {expected}, got g005={case05}, g020={case20}"
            )
        matched, audit = match_case(expected, frame05, frame20)
        row = {
            "case": expected, "n_pairs": len(matched),
            "R05": float(matched["R05"].mean()),
            "R20": float(matched["R20"].mean()),
            "R05_minus_R20": float((matched["R05"] - matched["R20"]).mean()),
            "N05": float(matched["N05"].mean()),
            "N20": float(matched["N20"].mean()),
            "N05_minus_N20": float((matched["N05"] - matched["N20"]).mean()),
        }
        case_rows.append(row)
        bin_rows.extend(binned_case_rows(expected, matched))
        audits.append(audit)
        print(
            f"case {expected:3d}: matched={len(matched):,} "
            f"R05={row['R05']:+.6f} R20={row['R20']:+.6f} "
            f"diff={row['R05_minus_R20']:+.6f}", flush=True,
        )
    cases = pd.DataFrame(case_rows)
    n_expected = args.max_case + 1
    if len(cases) != n_expected or not np.array_equal(cases["case"], np.arange(n_expected)):
        raise RuntimeError(
            f"expected cases 0--{args.max_case}, got {cases['case'].tolist()}"
        )
    case_bins = pd.DataFrame(bin_rows)
    bins = aggregate_bins(case_bins)
    difference = cases["R05_minus_R20"].to_numpy(float)
    null_difference = cases["N05_minus_N20"].to_numpy(float)
    result = {
        "design": (
            f"exact pair match, V2.2 regression cuts, cases 0--{args.max_case}; "
            "response averaged within case before uncertainty calculation"
        ),
        "uncertainty_unit": "rendered simulation case",
        "catalogues": {"g005": args.g005, "g020": args.g020},
        "n_cases": n_expected, "n_matched_pairs": int(cases["n_pairs"].sum()),
        "mean_pairs_per_case": float(cases["n_pairs"].mean()),
        "R05": stat(cases["R05"]), "R20": stat(cases["R20"]),
        "R05_minus_R20": stat(difference),
        "R20_over_R05_minus_one": jackknife_ratio(cases["R20"], cases["R05"]),
        "N05": stat(cases["N05"]), "N20": stat(cases["N20"]),
        "N05_minus_N20": stat(null_difference),
        "case_correlation_R05_R20": float(cases[["R05", "R20"]].corr().iloc[0, 1]),
        "conditional": bins,
        "audit": {
            "selected_g005": int(sum(row["selected_g005"] for row in audits)),
            "selected_g020": int(sum(row["selected_g020"] for row in audits)),
            "duplicate_rows_g005": int(sum(row["duplicate_rows_g005"] for row in audits)),
            "duplicate_rows_g020": int(sum(row["duplicate_rows_g020"] for row in audits)),
            "constgold_opened": False,
        },
    }
    result["interpretive_threshold"] = {
        "needed_positive_per_pair_shift_approx": 0.0009,
        "definition": (
            "approximately 0.0075 missing coherent response divided by 8.4 pairs per primary; "
            "diagnostic scale only, not fitted or applied"
        ),
        "R05_minus_R20_exceeds_needed_scale": bool(difference.mean() > 0.0009),
    }
    cases.to_csv(args.case_csv, index=False)
    pd.json_normalize(bins).to_csv(args.bin_csv, index=False)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "n_cases", "n_matched_pairs", "R05", "R20", "R05_minus_R20",
        "R20_over_R05_minus_one", "N05_minus_N20", "case_correlation_R05_R20",
        "interpretive_threshold",
    )}, indent=2, sort_keys=True), flush=True)
    print("RBLEND_LABEL_AMPLITUDE_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
