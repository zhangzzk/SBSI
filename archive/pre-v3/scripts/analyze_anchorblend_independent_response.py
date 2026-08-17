"""Compare independent-neighbour projected labels with coherent anchor truth."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summary(frame: pd.DataFrame, columns: list[str]) -> dict:
    case = frame.groupby("case", sort=True)[columns].mean()
    return {
        "n_rows": int(len(frame)), "n_cases": int(len(case)),
        **{column: stat(case[column].to_numpy(float)) for column in columns},
    }


def vector_fit(frame: pd.DataFrame, g: float) -> dict:
    """Case-level OLS of measured response vector on predicted response vector."""
    work = frame.copy()
    work["y1"] = (work.measured_e1_plus - work.measured_e1_minus) / (2.0 * g)
    work["y2"] = (work.measured_e2_plus - work.measured_e2_minus) / (2.0 * g)
    work["power"] = work.prediction_u1**2 + work.prediction_u2**2
    work["dot"] = work.prediction_u1 * work.y1 + work.prediction_u2 * work.y2
    work["cross"] = -work.prediction_u2 * work.y1 + work.prediction_u1 * work.y2
    case = work.groupby("case", sort=True)[["power", "dot", "cross"]].sum()
    if (case.power <= 0).any():
        raise RuntimeError("non-positive case model-vector power")
    slope = case["dot"] / case["power"]
    null = case["cross"] / case["power"]
    return {
        "slope_measured_on_predicted": stat(slope.to_numpy(float)),
        "slope_minus_one": stat((slope - 1.0).to_numpy(float)),
        "orthogonal_slope_null": stat(null.to_numpy(float)),
        "pooled_slope": float(case["dot"].sum() / case["power"].sum()),
        "pooled_orthogonal_slope": float(case["cross"].sum() / case["power"].sum()),
        "n_cases": int(len(case)),
    }


def normalize_input_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the legacy case-244 reset-index schema without changing IDs."""
    if "index" not in frame:
        return frame
    result = frame.copy()
    fallback = result["index"].notna()
    if "input_index" not in result:
        result["input_index"] = result["index"]
    else:
        missing = result["input_index"].isna()
        if np.any(missing & ~fallback):
            raise RuntimeError("missing input_index has no legacy index fallback")
        result.loc[missing, "input_index"] = result.loc[missing, "index"]
    result = result.drop(columns="index")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", required=True)
    parser.add_argument("--coherent", required=True)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument(
        "--allow-prediction-difference", action="store_true",
        help="permit an intentionally changed pair enumeration relative to coherent input",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    independent = normalize_input_index(pd.read_feather(args.independent))
    coherent = pd.read_feather(args.coherent)[
        KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    ]
    if independent.duplicated(KEY).any() or coherent.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor keys")
    common = independent.merge(coherent, on=KEY, how="inner", validate="one_to_one")
    common["coherent_gap"] = (
        common.R_blend_lsst_r_extnbr_v22 - common.R_blend_truth
    )
    common["prediction_replay"] = (
        common.prediction - common.R_blend_lsst_r_extnbr_v22
    )
    common["coherent_minus_random_label"] = (
        common.R_blend_truth - common.label_sum
    )
    replay = float(np.max(np.abs(common.prediction_replay)))
    if replay > 2e-7 and not args.allow_prediction_difference:
        raise RuntimeError(f"V2.2 prediction replay differs by {replay:.3e}")
    columns = [
        "prediction", "label_sum", "projected_prediction", "desired_gap",
        "projected_gap", "null_sum",
    ]
    common_columns = [
        *columns, "R_blend_truth", "coherent_gap",
        "coherent_minus_random_label", "prediction_replay",
    ]
    development = independent.case <= args.development_max
    common_development = common.case <= args.development_max
    payload = {
        "design": "same latent anchor scenes/noise; independent per-neighbour +/-0.05 directions inside 10 arcsec versus coherent neighbour shear",
        "independent": {
            "development": summary(independent.loc[development], columns),
            "validation": summary(independent.loc[~development], columns),
            "all": summary(independent, columns),
            "vector_fit_development": vector_fit(independent.loc[development], args.g),
            "vector_fit_validation": vector_fit(independent.loc[~development], args.g),
            "vector_fit_all": vector_fit(independent, args.g),
        },
        "exact_common_with_coherent": {
            "development": summary(common.loc[common_development], common_columns),
            "validation": summary(common.loc[~common_development], common_columns),
            "all": summary(common, common_columns),
        },
        "n_independent_rows": int(len(independent)),
        "n_coherent_rows": int(len(coherent)),
        "n_common_rows": int(len(common)),
        "common_fraction_of_independent": float(len(common) / len(independent)),
        "common_fraction_of_coherent": float(len(common) / len(coherent)),
        "prediction_replay_max_abs": replay,
        "prediction_replay_required": not args.allow_prediction_difference,
        "interpretation": {
            "desired_gap": "V2.2 coherent-response estimand minus random-direction projected-label sum; cross terms cancel only in expectation",
            "projected_gap": "V2.2 prediction transformed into the exact random-label coordinate minus label; tests model-label reproduction without coherence cross-term",
            "coherent_gap": "V2.2 summed response minus direct coherent neighbour response",
            "vector_fit": "case-level OLS of measured independent-scene response vector on sum_i(R_i u_i); slope one is closure and the orthogonal slope is a null",
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_INDEPENDENT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
