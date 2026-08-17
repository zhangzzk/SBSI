"""Test whether restricting the V2.1 anchor prediction from 10 to 7 arcsec closes its deficit.

The coherent anchor truth is read from the completed neighbour-only simulations.  For the exact
same retained anchors, the V2.1 emulator is evaluated once with its native 10-arcsec/k=20 pair
list; that pair table is then summed both in full and after a distance < 7 arcsec restriction.
The native sum must reproduce the value stored by ``build_anchorblend_response.py``.

This isolates the aperture question without fitting anything and without reading constgold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402

COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def read_disjoint(paths: list[str]) -> pd.DataFrame:
    frames = []
    seen: set[int] = set()
    for path in paths:
        frame = pd.read_feather(path)
        cases = set(frame["case"].astype(int).unique())
        overlap = seen & cases
        if overlap:
            raise RuntimeError(f"input response files overlap in cases: {sorted(overlap)[:5]}")
        seen.update(cases)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def field_path(base: str, case: int, sign: float) -> str:
    return os.path.join(
        base, f"case{case}_{float(sign)}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )


def summed_predictions(predictor, frame: pd.DataFrame, anchors: np.ndarray,
                       radius: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = predictor.predict_response(frame, frame)
    primary = next(c for c in pairs.columns if c.startswith("index") and c.endswith("_p"))
    if "distance" not in pairs.columns:
        raise KeyError(f"prediction pair table has no distance column: {pairs.columns.tolist()}")
    full = pairs.groupby(primary, sort=False)["response"].sum()
    inner = pairs.loc[pairs["distance"].to_numpy(float) < radius].groupby(
        primary, sort=False,
    )["response"].sum()
    return (
        full.reindex(anchors, fill_value=0.0).to_numpy(float),
        inner.reindex(anchors, fill_value=0.0).to_numpy(float),
    )


def ratio_summary(case_table: pd.DataFrame, pred: str, draws: int, seed: int) -> dict:
    truth = case_table["truth"].to_numpy(float)
    prediction = case_table[pred].to_numpy(float)
    ratio = float(prediction.mean() / truth.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for i in range(draws):
        take = rng.integers(0, len(case_table), len(case_table))
        boot[i] = prediction[take].mean() / truth[take].mean()
    return {
        "truth_mean": float(truth.mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_over_truth_minus_one": ratio - 1.0,
        "prediction_over_truth_minus_one_se": float(boot.std(ddof=1)),
        "truth_over_prediction_scale": 1.0 / ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", nargs="+", required=True)
    ap.add_argument("--base-old", required=True, help="simulation base for cases below --split-case")
    ap.add_argument("--base-new", required=True, help="simulation base for cases at/above --split-case")
    ap.add_argument("--split-case", type=int, default=100)
    ap.add_argument("--sign", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=7.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_v21")
    ap.add_argument("--bootstrap", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=25876)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    data = read_disjoint(args.response)
    stored = f"R_blend_{args.tag}"
    required = {"case", "input_index", "R_blend_truth", stored}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"response inputs missing {sorted(missing)}")

    predictor = BlendingPredictor.load(
        "/home/z/Zekang.Zhang/blendemu/models", tag=args.tag,
        conditions=COND, device="cpu",
    )
    _, native_radius, native_k = predictor._select("regression")
    if native_radius != 10 or native_k != 20:
        raise RuntimeError(f"expected native 10 arcsec/k=20, got {native_radius}/{native_k}")
    if not 0 < args.radius < native_radius:
        raise ValueError(f"restricted radius must be between 0 and {native_radius}")

    case_rows = []
    maximum_replay_error = 0.0
    for case, rows in data.groupby("case", sort=True):
        case = int(case)
        anchors = rows["input_index"].to_numpy(np.int64)
        base = args.base_old if case < args.split_case else args.base_new
        path = field_path(base, case, args.sign)
        field = pd.read_feather(path)
        field = field.rename(columns={c: c.replace("_input", "") for c in field.columns})
        pred10, pred7 = summed_predictions(predictor, field, anchors, args.radius)
        expected = rows[stored].to_numpy(float)
        replay_error = float(np.max(np.abs(pred10 - expected)))
        maximum_replay_error = max(maximum_replay_error, replay_error)
        if replay_error > 1e-10:
            raise RuntimeError(f"case {case}: native prediction replay differs by {replay_error}")
        case_rows.append({
            "case": case,
            "n": len(rows),
            "truth": float(rows["R_blend_truth"].mean()),
            "pred10": float(pred10.mean()),
            "pred7": float(pred7.mean()),
        })
        print(
            f"case {case}: n={len(rows):,} truth={case_rows[-1]['truth']:+.6f} "
            f"pred10={case_rows[-1]['pred10']:+.6f} pred7={case_rows[-1]['pred7']:+.6f}",
            flush=True,
        )

    cases = pd.DataFrame(case_rows)
    ten = ratio_summary(cases, "pred10", args.bootstrap, args.seed)
    seven = ratio_summary(cases, "pred7", args.bootstrap, args.seed + 1)
    shell = cases["pred10"].to_numpy(float) - cases["pred7"].to_numpy(float)
    result = {
        "tag": args.tag,
        "n_cases": len(cases),
        "n_anchors": int(cases["n"].sum()),
        "native_radius_arcsec": native_radius,
        "restricted_radius_arcsec": args.radius,
        "k": native_k,
        "native_replay_max_abs_error": maximum_replay_error,
        "native_10_arcsec": ten,
        "restricted_7_arcsec": seven,
        "prediction_7_over_10": float(cases["pred7"].mean() / cases["pred10"].mean()),
        "mean_7_to_10_shell": float(shell.mean()),
        "case_table": case_rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "case_table"}, indent=2))
    if os.path.exists(args.output_json):
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("ANCHORBLEND_APERTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
