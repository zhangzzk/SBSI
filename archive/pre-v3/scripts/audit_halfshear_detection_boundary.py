"""Test detection-instability structure on held-out half-shear pair labels.

Cases 0--39 were not used to train V2.2.  Pair labels and predictions are
summed per primary, including eligible primaries whose supported-pair sum is
zero.  Existing g=0 and g=0.2 catalogues then provide centroid/match stability
metrics for the same primaries.  Cases 0--19 define all bin edges and within-
cell medians; cases 20--39 are the untouched validation half.

This reads already-rendered simulation catalogues only and never opens
constgold.  The primary endpoint is the within-cell high-minus-low contrast in
V2.2-minus-label residual for centroid shift.  Other stability coordinates are
reported as secondary, Holm-adjusted endpoints.
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
from scipy import stats


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from scripts.audit_anchorblend_detection_boundary import (  # noqa: E402
    load_leg, pair_legs, holm_adjust,
)


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
NEED = [
    "case", "input_index", "RA_input_p", "DEC_input_p",
    "delta_et1", "delta_et2", *FEATURES,
]
METRICS = [
    "centroid_shift_arcsec", "match_distance_change_pix",
    "flux_fractional_change", "isoarea_fractional_change", "abs_dmag_max",
]


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()), "case_sem": sem(values),
        "n_cases": int(len(values)), "t": float(test.statistic),
        "p_two_sided": float(test.pvalue),
    }


def quantile_edges(values: np.ndarray, count: int) -> np.ndarray:
    boundary = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(boundary) != count + 1:
        raise RuntimeError(f"collapsed quantile edges: wanted {count + 1}, got {len(boundary)}")
    boundary[0], boundary[-1] = -np.inf, np.inf
    return boundary


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def build_primary_table(args: argparse.Namespace, predictor: BlendingPredictor) -> pd.DataFrame:
    cuts, _, _ = predictor._select("regression")
    pieces: list[pd.DataFrame] = []
    with ipc.open_file(args.catalogue) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(NEED).to_pandas()
            if len(frame) == 0:
                continue
            frame = frame[frame.case.to_numpy(int) < args.n_cases]
            if len(frame) == 0:
                if int(reader.get_batch(batch_index).column(reader.schema.get_field_index("case"))[0].as_py()) >= args.n_cases:
                    break
                continue
            finite = np.isfinite(frame[["delta_et1", "delta_et2"]].to_numpy(float)).all(axis=1)
            frame = frame.loc[finite].copy()
            primary_ok = (
                (frame.r_input_p > cuts[1][0]) & (frame.r_input_p < cuts[1][1])
                & (frame.Re_input_p > cuts[3][0]) & (frame.Re_input_p < cuts[3][1])
            )
            eligible = frame.loc[primary_ok, [
                "case", "input_index", "RA_input_p", "DEC_input_p",
                "r_input_p", "Re_input_p",
            ]].drop_duplicates(["case", "input_index"])
            pair_ok = primary_ok & (
                (frame.r_input_s > cuts[0][0]) & (frame.r_input_s < cuts[0][1])
                & (frame.Re_input_s > cuts[2][0]) & (frame.Re_input_s < cuts[2][1])
                & (frame.distance > cuts[4][0]) & (frame.distance < cuts[4][1])
            )
            pairs = frame.loc[pair_ok]
            if len(pairs):
                pred = predictor.predict_on_pairs(pairs[FEATURES], task="response")["response"].to_numpy(float)
                scored = pairs[["case", "input_index"]].copy()
                scored["label"] = pairs.delta_et1.to_numpy(float) / args.shear
                scored["null"] = pairs.delta_et2.to_numpy(float) / args.shear
                scored["prediction"] = pred
                sums = scored.groupby(["case", "input_index"], as_index=False).agg(
                    label=("label", "sum"), null=("null", "sum"),
                    prediction=("prediction", "sum"), n_pairs=("label", "size"),
                )
                eligible = eligible.merge(sums, on=["case", "input_index"], how="left", validate="one_to_one")
            else:
                eligible[["label", "null", "prediction", "n_pairs"]] = np.nan
            eligible[["label", "null", "prediction", "n_pairs"]] = eligible[
                ["label", "null", "prediction", "n_pairs"]
            ].fillna(0.0)
            pieces.append(eligible)
            if len(pieces) % 100 == 0:
                print(f"scored {len(pieces)} response batches", flush=True)
    table = pd.concat(pieces, ignore_index=True)
    # Batches may split a primary's secondary rows; consolidate exactly once.
    identity = ["case", "input_index"]
    invariant = table.groupby(identity)[
        ["RA_input_p", "DEC_input_p", "r_input_p", "Re_input_p"]
    ].nunique()
    if (invariant > 1).any().any():
        raise RuntimeError("primary properties vary across response batches")
    table = table.groupby(identity, as_index=False).agg(
        RA_input_p=("RA_input_p", "first"), DEC_input_p=("DEC_input_p", "first"),
        r_input_p=("r_input_p", "first"), Re_input_p=("Re_input_p", "first"),
        label=("label", "sum"), null=("null", "sum"),
        prediction=("prediction", "sum"), n_pairs=("n_pairs", "sum"),
    )
    table["gap"] = table.prediction - table.label
    return table


def matched_contrast(frame: pd.DataFrame, metric: str, threshold: pd.Series) -> tuple[np.ndarray, list[int]]:
    work = frame.copy()
    work["threshold"] = work.cell.map(threshold)
    work = work[np.isfinite(work.threshold)].copy()
    work["high"] = work[metric] > work.threshold
    contrasts, used_cells = [], []
    for _, case in work.groupby("case", sort=True):
        difference, weight = [], []
        for _, cell in case.groupby("cell", sort=False):
            high = cell.loc[cell.high, "gap"].to_numpy(float)
            low = cell.loc[~cell.high, "gap"].to_numpy(float)
            if len(high) < 3 or len(low) < 3:
                continue
            difference.append(float(high.mean() - low.mean()))
            weight.append(float(min(len(high), len(low))))
        if difference:
            contrasts.append(float(np.average(difference, weights=weight)))
            used_cells.append(len(difference))
    return np.asarray(contrasts), used_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu",
    )
    primary = build_primary_table(args, predictor)
    context = []
    counts = []
    for case_id, target in primary.groupby("case", sort=True):
        truth = target.rename(columns={"RA_input_p": "RA", "DEC_input_p": "DEC"})[
            ["input_index", "RA", "DEC"]
        ]
        sheared, sheared_count = load_leg(args.base, int(case_id), args.shear, truth, "")
        zero, zero_count = load_leg(args.base, int(case_id), 0.0, truth, "")
        context.append(pair_legs(sheared, zero, int(case_id)))
        counts.append({"case": int(case_id), "sheared": sheared_count, "zero": zero_count})
        print(f"case {case_id}: eligible={len(target)} common={len(context[-1])}", flush=True)
    context = pd.concat(context, ignore_index=True)
    table = primary.merge(context, on=["case", "input_index"], how="inner", validate="one_to_one")
    if table.case.nunique() != args.n_cases:
        raise RuntimeError("missing half-shear cases after detection join")
    table.to_feather(args.table_output)

    development = table[table.case <= args.development_max].copy()
    validation = table[table.case > args.development_max].copy()
    boundaries = {
        "prediction": quantile_edges(development.prediction, 8),
        "r_input_p": quantile_edges(development.r_input_p, 4),
        "Re_input_p": quantile_edges(development.Re_input_p, 4),
    }
    for part in (development, validation):
        bins = [assign(part[name], boundary) for name, boundary in boundaries.items()]
        part["cell"] = bins[0] * 16 + bins[1] * 4 + bins[2]

    endpoints = {}
    raw_p = {}
    for metric in METRICS:
        threshold = development.groupby("cell", sort=True)[metric].median()
        dev, dev_cells = matched_contrast(development, metric, threshold)
        val, val_cells = matched_contrast(validation, metric, threshold)
        endpoints[metric] = {
            "development": stat(dev), "validation": stat(val),
            "development_median_cells_per_case": float(np.median(dev_cells)),
            "validation_median_cells_per_case": float(np.median(val_cells)),
            "sign_definition": "high instability minus low instability V2.2-minus-label gap",
        }
        raw_p[metric] = endpoints[metric]["validation"]["p_two_sided"]
    adjusted = holm_adjust(raw_p)
    for metric in endpoints:
        endpoints[metric]["validation_p_holm"] = float(adjusted[metric])

    def global_summary(part: pd.DataFrame) -> dict:
        case = part.groupby("case", sort=True)[["label", "prediction", "gap", "null"]].mean()
        return {
            name: {"mean": float(case[name].mean()), "case_sem": sem(case[name])}
            for name in case.columns
        } | {"n_cases": int(len(case)), "n_primaries": int(len(part))}

    payload = {
        "design": "held-out cases 0--19 development, 20--39 validation; 8 prediction x 4 mag x 4 size matched cells",
        "tag": args.tag, "shear": args.shear,
        "development": global_summary(development),
        "validation": global_summary(validation),
        "matched_endpoints": endpoints,
        "boundaries": {
            name: [None if not np.isfinite(value) else float(value) for value in boundary]
            for name, boundary in boundaries.items()
        },
        "catalogue_counts": counts,
        "primary_endpoint": "centroid_shift_arcsec",
        "correction_fitted": False, "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"development": payload["development"], "validation": payload["validation"],
                      "matched_endpoints": endpoints}, indent=2, sort_keys=True))
    print("HALFSHEAR_DETECTION_BOUNDARY_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
