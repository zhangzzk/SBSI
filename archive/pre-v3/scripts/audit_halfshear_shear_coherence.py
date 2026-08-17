"""Test whether simultaneous-neighbour shear coherence drives V2.2 residuals.

V2.2 is pairwise, whereas several neighbours are sheared simultaneously in a
training image.  Cases 0--19 define property cells and coherence medians;
cases 20--39 test whether primaries whose active neighbour shear directions
are unusually coherent have a different summed-label residual.  Exact
supported-pair count is included in the matching cells.

This uses held-out half-shear simulations only.  No coherent-anchor response
or constgold quantity is used to define the test.
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


NEED = [
    "case", "input_index", "shear_angle", "polarization_angle",
    "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "distance",
    "sersic_n_input_p", "sersic_n_input_s", "delta_et1", "delta_et2",
]
CUTS = [[13, 29], [18, 25.8], [0.0, 10.0], [0.5, 1.5], [0, 10]]
FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)


def edges(values: np.ndarray, count: int) -> np.ndarray:
    boundary = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(boundary) != count + 1:
        raise RuntimeError(f"collapsed edges: wanted {count + 1}, got {len(boundary)}")
    boundary[0], boundary[-1] = -np.inf, np.inf
    return boundary


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)), "t": float(test.statistic),
        "p_two_sided": float(test.pvalue),
    }


def build_coherence(path: str, n_cases: int, predictor: BlendingPredictor) -> pd.DataFrame:
    parts = []
    with ipc.open_file(path) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(NEED).to_pandas()
            frame = frame[frame.case.to_numpy(int) < n_cases]
            if len(frame) == 0:
                continue
            finite = np.isfinite(frame[["delta_et1", "delta_et2", "shear_angle"]]).all(axis=1)
            pair_ok = finite & (
                (frame.r_input_s > CUTS[0][0]) & (frame.r_input_s < CUTS[0][1])
                & (frame.r_input_p > CUTS[1][0]) & (frame.r_input_p < CUTS[1][1])
                & (frame.Re_input_s > CUTS[2][0]) & (frame.Re_input_s < CUTS[2][1])
                & (frame.Re_input_p > CUTS[3][0]) & (frame.Re_input_p < CUTS[3][1])
                & (frame.distance > CUTS[4][0]) & (frame.distance < CUTS[4][1])
            )
            selected = frame.loc[pair_ok, [
                "case", "input_index", "shear_angle", "polarization_angle",
            ]].copy()
            if len(selected) == 0:
                continue
            shear_phase = np.deg2rad(2.0 * selected.shear_angle.to_numpy(float))
            relative_phase = np.deg2rad(
                2.0 * (selected.shear_angle.to_numpy(float)
                       - selected.polarization_angle.to_numpy(float))
            )
            selected["cos_shear"] = np.cos(shear_phase)
            selected["sin_shear"] = np.sin(shear_phase)
            selected["cos_relative"] = np.cos(relative_phase)
            selected["sin_relative"] = np.sin(relative_phase)
            prediction = predictor.predict_on_pairs(
                frame.loc[pair_ok, FEATURES], task="response",
            )["response"].to_numpy(float)
            selected["pair_prediction"] = prediction
            selected["prediction_cos"] = prediction * selected.cos_shear
            selected["prediction_sin"] = prediction * selected.sin_shear
            parts.append(selected.groupby(["case", "input_index"], as_index=False).agg(
                n_pairs=("shear_angle", "size"),
                cos_shear=("cos_shear", "sum"), sin_shear=("sin_shear", "sum"),
                cos_relative=("cos_relative", "sum"), sin_relative=("sin_relative", "sum"),
                desired_prediction=("pair_prediction", "sum"),
                prediction_cos=("prediction_cos", "sum"),
                prediction_sin=("prediction_sin", "sum"),
            ))
    table = pd.concat(parts, ignore_index=True).groupby(
        ["case", "input_index"], as_index=False,
    ).sum()
    table["shear_coherence"] = np.hypot(table.cos_shear, table.sin_shear) / table.n_pairs
    table["relative_angle_concentration"] = (
        np.hypot(table.cos_relative, table.sin_relative) / table.n_pairs
    )
    # The observed summed label is Delta-e dot sum(u_i)/g.  A pairwise model
    # predicts Delta-e/g = sum(R_i u_i), hence its prediction in that *same*
    # label coordinate is [sum(R_i u_i)] dot sum(u_i).  Comparing sum(R_i)
    # directly to the conditional label sum would manufacture coherence cross-
    # terms even for a perfect model.
    table["projected_label_prediction"] = (
        table.prediction_cos * table.cos_shear
        + table.prediction_sin * table.sin_shear
    )
    return table


def matched_contrast(frame: pd.DataFrame, metric: str, thresholds: pd.Series,
                     target: str) -> tuple[np.ndarray, list[int]]:
    work = frame.copy()
    keys = pd.MultiIndex.from_frame(work[["property_cell", "n_pairs"]])
    work["threshold"] = thresholds.reindex(keys).to_numpy(float)
    work = work[np.isfinite(work.threshold)].copy()
    work["high"] = work[metric] > work.threshold
    contrasts, cells_used = [], []
    for _, case in work.groupby("case", sort=True):
        differences, weights = [], []
        for _, cell in case.groupby(["property_cell", "n_pairs"], sort=False):
            high = cell.loc[cell.high, target].to_numpy(float)
            low = cell.loc[~cell.high, target].to_numpy(float)
            if len(high) < 3 or len(low) < 3:
                continue
            differences.append(float(high.mean() - low.mean()))
            weights.append(float(min(len(high), len(low))))
        if differences:
            contrasts.append(float(np.average(differences, weights=weights)))
            cells_used.append(len(differences))
    return np.asarray(contrasts), cells_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--primary-table", required=True)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    primary = pd.read_feather(args.primary_table)
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    coherence = build_coherence(args.catalogue, args.n_cases, predictor)
    table = primary.merge(
        coherence, on=["case", "input_index"], suffixes=("", "_coherence"),
        how="inner", validate="one_to_one",
    )
    if not np.array_equal(
        table.n_pairs.to_numpy(int), table.n_pairs_coherence.to_numpy(int),
    ):
        raise RuntimeError("supported-pair counts differ from the held-out closure table")
    table = table.drop(columns="n_pairs_coherence")
    replay = float(np.max(np.abs(table.prediction - table.desired_prediction)))
    if replay > 2e-12:
        raise RuntimeError(f"pair prediction replay differs by {replay:.3e}")
    table["projected_gap"] = table.projected_label_prediction - table.label
    table.to_feather(args.table_output)
    development = table[table.case <= args.development_max].copy()
    validation = table[table.case > args.development_max].copy()
    boundaries = {
        "prediction": edges(development.prediction, 8),
        "r_input_p": edges(development.r_input_p, 4),
        "Re_input_p": edges(development.Re_input_p, 4),
    }
    for frame in (development, validation):
        b0 = assign(frame.prediction, boundaries["prediction"])
        b1 = assign(frame.r_input_p, boundaries["r_input_p"])
        b2 = assign(frame.Re_input_p, boundaries["Re_input_p"])
        frame["property_cell"] = b0 * 16 + b1 * 4 + b2

    results = {}
    for metric in ("shear_coherence", "relative_angle_concentration"):
        threshold = development.groupby(
            ["property_cell", "n_pairs"], sort=True,
        )[metric].median()
        dev, dev_cells = matched_contrast(
            development, metric, threshold, "projected_gap",
        )
        val, val_cells = matched_contrast(
            validation, metric, threshold, "projected_gap",
        )
        results[metric] = {
            "development": stat(dev), "validation": stat(val),
            "development_median_cells_per_case": float(np.median(dev_cells)),
            "validation_median_cells_per_case": float(np.median(val_cells)),
            "sign_definition": "high concentration minus low concentration cross-talk-corrected projected-prediction-minus-label gap",
        }

    # Transparent validation profile: coherence quartiles frozen within exact
    # property+pair-count cells, then averaged with cases as uncertainty units.
    q_edges = development.groupby(["property_cell", "n_pairs"])["shear_coherence"].quantile(
        [0.25, 0.5, 0.75]
    ).unstack()
    profile_rows = []
    for row in validation.itertuples():
        key = (row.property_cell, row.n_pairs)
        if key not in q_edges.index:
            continue
        boundary = q_edges.loc[key].to_numpy(float)
        quartile = int(np.searchsorted(boundary, row.shear_coherence, side="right"))
        profile_rows.append((row.case, quartile, row.projected_gap))
    profile = pd.DataFrame(profile_rows, columns=["case", "quartile", "projected_gap"])
    profile_result = {}
    for quartile, group in profile.groupby("quartile"):
        by_case = group.groupby("case").projected_gap.mean().to_numpy(float)
        profile_result[str(int(quartile))] = {
            "n_rows": int(len(group)), "projected_gap": stat(by_case),
        }

    global_summary = {}
    for name, frame in (("development", development), ("validation", validation)):
        case = frame.groupby("case")[["gap", "projected_gap"]].mean()
        global_summary[name] = {
            "naive_desired_prediction_minus_label": stat(case.gap.to_numpy(float)),
            "projected_prediction_minus_label": stat(case.projected_gap.to_numpy(float)),
        }

    payload = {
        "design": "c0--19 thresholds; c20--39 validation; match 8 prediction x 4 mag x 4 size x exact supported-pair count",
        "primary_endpoint": "shear_coherence",
        "shear_coherence_definition": "magnitude of mean exp(2i*neighbour shear angle)",
        "label_coordinate": "projected prediction [sum R_i u_i] dot [sum u_i] compared to observed sum of projected labels",
        "naive_desired_prediction_minus_label_is_conditionally_confounded": True,
        "prediction_replay_max_abs": replay,
        "global_summary": global_summary,
        "matched_endpoints": results,
        "validation_coherence_quartiles": profile_result,
        "n_rows": int(len(table)), "n_cases": int(table.case.nunique()),
        "constgold_opened": False, "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_SHEAR_COHERENCE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
