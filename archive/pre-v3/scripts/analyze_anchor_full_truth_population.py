"""Compare selected and full intrinsic anchor populations at truth position."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402


TILE = "tile180.0_-0.5"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    case = frame.groupby("case", sort=True)[["truth", "prediction", "gap", "null"]].mean()
    return {
        "n_rows": int(len(frame)), "n_cases": int(len(case)),
        **{name: stat(case[name].to_numpy(float)) for name in case},
    }


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--missing-dir", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    selected_paths = sorted(glob.glob(os.path.join(args.selected_dir, "case*.feather")))
    missing_paths = sorted(glob.glob(os.path.join(args.missing_dir, "case*.feather")))
    expected = args.case_max - args.case_min + 1
    if len(selected_paths) != expected or len(missing_paths) != expected:
        raise RuntimeError(
            f"expected {expected}+{expected} case files, got {len(selected_paths)}+{len(missing_paths)}"
        )
    selected_raw = pd.concat([pd.read_feather(path) for path in selected_paths], ignore_index=True)
    missing_raw = pd.concat([pd.read_feather(path) for path in missing_paths], ignore_index=True)
    shape_columns = [
        "g1_truthpos_plus", "g2_truthpos_plus", "g1_truthpos_minus", "g2_truthpos_minus",
    ]
    selected = selected_raw[["case", "input_index", *shape_columns]].copy()
    selected["selected"] = True
    missing = missing_raw[["case", "input_index", *shape_columns]].copy()
    missing["selected"] = False
    measured = pd.concat([selected, missing], ignore_index=True)
    if measured.duplicated(["case", "input_index"]).any():
        raise RuntimeError("selected and missing truth-position keys overlap")

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu",
    )
    parts = []
    selected_response = pd.read_feather(args.response, columns=["case", "input_index"])
    for case in range(args.case_min, args.case_max + 1):
        manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
        ids = manifest["index"].to_numpy(np.int64)
        case_measured = measured.loc[measured.case == case].set_index(
            "input_index", verify_integrity=True,
        )
        if set(case_measured.index) != set(ids):
            raise RuntimeError(f"case {case}: measured IDs do not partition manifest")
        frame = input_frame(args.base, case, args.g)
        pairs = predictor.predict_response(frame, frame)
        primary = next(
            column for column in pairs if column.startswith("index") and column.endswith("_p")
        )
        prediction = pairs.groupby(primary, sort=False).response.sum().reindex(
            ids, fill_value=0.0,
        ).to_numpy(float)
        case_table = case_measured.loc[ids].reset_index()
        case_table["prediction"] = prediction
        parts.append(case_table)
        n_stored = int((selected_response.case == case).sum())
        print(
            f"case {case}: manifest={len(ids)} selected={n_stored} "
            f"prediction_mean={prediction.mean():+.6f}", flush=True,
        )
    table = pd.concat(parts, ignore_index=True)
    finite = np.isfinite(table[shape_columns].to_numpy(float)).all(axis=1)
    finite_selected = table.selected.to_numpy(bool) & finite
    finite_full = finite
    scale = 2.0 * args.g
    table["truth"] = (table.g1_truthpos_plus - table.g1_truthpos_minus) / scale
    table["null"] = (table.g2_truthpos_plus - table.g2_truthpos_minus) / scale
    table["gap"] = table.prediction - table.truth
    table.to_feather(args.table_output)

    development = table.case <= args.development_max
    regions = {
        "development_selected": development & finite_selected,
        "development_full": development & finite_full,
        "validation_selected": ~development & finite_selected,
        "validation_full": ~development & finite_full,
        "all_selected": finite_selected,
        "all_full": finite_full,
    }
    summaries = {name: summarize(table.loc[mask]) for name, mask in regions.items()}
    contrasts = {}
    for name in ("development", "validation", "all"):
        selected_case = table.loc[regions[f"{name}_selected"]].groupby("case")[
            ["truth", "prediction", "gap"]
        ].mean()
        full_case = table.loc[regions[f"{name}_full"]].groupby("case")[
            ["truth", "prediction", "gap"]
        ].mean()
        difference = selected_case - full_case
        contrasts[name] = {
            f"selected_minus_full_{column}": stat(difference[column].to_numpy(float))
            for column in difference
        }
    payload = {
        "design": "same rendered pixels, fixed truth-position ngmix; full intrinsic manifest versus stored both-leg detection/match/bright-cut population",
        "summaries": summaries, "selected_minus_full": contrasts,
        "n_manifest": int(len(table)), "n_selected": int(table.selected.sum()),
        "n_finite_full": int(finite_full.sum()),
        "n_finite_selected": int(finite_selected.sum()),
        "full_finite_fraction": float(finite_full.mean()),
        "selection_carrier_gate": {
            "validation_full_reduces_absolute_gap": bool(
                abs(summaries["validation_full"]["gap"]["mean"])
                < abs(summaries["validation_selected"]["gap"]["mean"])
            ),
            "all_full_reduces_absolute_gap": bool(
                abs(summaries["all_full"]["gap"]["mean"])
                < abs(summaries["all_selected"]["gap"]["mean"])
            ),
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_FULL_TRUTH_POPULATION_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
