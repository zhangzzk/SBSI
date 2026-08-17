"""Fit a half-shear conditional-mean response target for a V2.7 scene arm.

The noisy per-object SNC response is never used directly as a training label.
A fixed-capacity HGB regressor estimates its conditional mean on half-shear
cases 0--99.  Case-grouped out-of-fold diagnostics are stored before the final
model is refit on all allowed cases.  Constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection


SHELL_FEATURES = [
    "logflux_abs_shell_0_0p5", "logflux_abs_shell_0p5_1",
    "logflux_abs_shell_1_2", "logflux_abs_shell_2_3",
    "logflux_abs_shell_3_5", "logflux_abs_shell_5_10",
]
PURITY_FEATURES = ["true_blendedness_eq17"]
KEYMUL = 1_000_003


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=240,
        max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0,
        early_stopping=False, random_state=7301,
    )


def load_snc(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pf.read_table(path, columns=[
        "case", "input_index", "ngmix0_g1", "ngmix0_g2",
    ])
    key = (table["case"].to_numpy().astype(np.int64) * KEYMUL
           + table["input_index"].to_numpy().astype(np.int64))
    order = np.argsort(key)
    key = key[order]
    if np.any(key[1:] == key[:-1]):
        raise RuntimeError("SNC lookup has duplicate keys")
    return key, table["ngmix0_g1"].to_numpy()[order], table["ngmix0_g2"].to_numpy()[order]


def match_snc(key: np.ndarray, skey: np.ndarray, s1: np.ndarray,
              s2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match SNC shapes and mark genuinely absent g=0 measurements as NaN."""
    pos = np.clip(np.searchsorted(skey, key), 0, len(skey) - 1)
    match = skey[pos] == key
    g01 = np.full(len(key), np.nan, dtype=float)
    g02 = np.full(len(key), np.nan, dtype=float)
    g01[match] = s1[pos[match]]
    g02[match] = s2[pos[match]]
    return match, g01, g02


def load_labels(args, features: list[str]) -> pd.DataFrame:
    skey, s1, s2 = load_snc(args.snc_lookup)
    need = [
        "case", "input_index", "detected", "r_input_p", "Re_input_p",
        "distance", "neighbored", "gamma1_input_p", "gamma2_input_p",
        "measured_ngmix_g1", "measured_ngmix_g2", *features,
    ]
    parts = []
    cuts = [list(value) for value in DEFAULT_SELECTION_CUTS]
    cuts[1][1] = args.primary_mag_max
    cuts[3][0] = args.primary_re_min
    with ipc.open_file(pa.memory_map(args.catalogue)) as reader:
        missing = set(need) - set(reader.schema.names)
        if missing:
            raise KeyError(f"target catalogue is missing {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(need).to_pandas()
            frame = frame[frame["case"].to_numpy(int) <= args.max_case]
            if frame.empty:
                continue
            frame = source_select_selection(frame, cuts=cuts)
            frame = frame[frame["detected"].astype(bool)].reset_index(drop=True)
            if len(frame):
                parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    g1 = frame["gamma1_input_p"].to_numpy(float)
    g2 = frame["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    keep = gmag > 1.0e-6
    frame = frame.loc[keep].reset_index(drop=True)
    g1, g2, gmag = g1[keep], g2[keep], gmag[keep]
    key = (frame["case"].to_numpy(np.int64) * KEYMUL
           + frame["input_index"].to_numpy(np.int64))
    match, g01, g02 = match_snc(key, skey, s1, s2)
    snc_match_fraction = float(match.mean())
    print(f"SNC match after selection: {snc_match_fraction:.6%} "
          f"({int(match.sum()):,}/{len(match):,})", flush=True)
    # The certified V2.2 target has 99.66% coverage: its g=0 lookup intentionally
    # contains only ngmix-converged measurements, and unmatched rows are removed
    # by the finite-label cut.  Refuse a gross population/key mismatch, but do not
    # demand measurements that the established SNC definition does not contain.
    if snc_match_fraction < 0.99:
        raise RuntimeError(f"SNC lookup coverage is only {match.mean():.6%}")
    de1 = frame["measured_ngmix_g1"].to_numpy(float) - g01
    de2 = frame["measured_ngmix_g2"].to_numpy(float) - g02
    frame["R_snc"] = (de1 * g1 / gmag + de2 * g2 / gmag) / args.nominal_g
    finite = np.isfinite(frame[[*features, "r_input_p", "Re_input_p", "R_snc"]]).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)

    # All-pairs rows repeat the same target and the same scene-level features.
    # Verify invariance before retaining one statistically independent row.
    key = (frame["case"].to_numpy(np.int64) * KEYMUL
           + frame["input_index"].to_numpy(np.int64))
    order = np.argsort(key, kind="mergesort")
    ordered = frame.iloc[order].reset_index(drop=True)
    okey = key[order]
    duplicate = okey[1:] == okey[:-1]
    invariant_columns = ["r_input_p", "Re_input_p", *features, "R_snc"]
    left = ordered.iloc[1:][invariant_columns].to_numpy(float)
    right = ordered.iloc[:-1][invariant_columns].to_numpy(float)
    if duplicate.any() and not np.allclose(left[duplicate], right[duplicate], rtol=0, atol=1e-7):
        raise RuntimeError("per-target response or scene feature differs across pair rows")
    independent = ordered.loc[np.r_[True, ~duplicate]].reset_index(drop=True)
    independent.attrs["snc_match_fraction"] = snc_match_fraction
    independent.attrs["snc_selected_rows"] = len(match)
    independent.attrs["snc_matched_rows"] = int(match.sum())
    print(f"target rows: {len(frame):,} pair rows -> {len(independent):,} unique targets", flush=True)
    return independent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--arm", choices=["sixshell", "purity"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--primary-mag-max", type=float, default=25.8)
    ap.add_argument("--primary-re-min", type=float, default=0.5)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    scene = SHELL_FEATURES if args.arm == "sixshell" else PURITY_FEATURES
    features = ["r_input_p", "Re_input_p", *scene]
    frame = load_labels(args, scene)
    X = frame[features].to_numpy(float)
    y = frame["R_snc"].to_numpy(float)
    groups = frame["case"].to_numpy(np.int64)
    oof = np.full(len(frame), np.nan)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        model = make_model()
        model.fit(X[train], y[train])
        oof[test] = model.predict(X[test])
        print(f"fold {fold}: train={len(train):,}, test={len(test):,}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF prediction coverage incomplete")
    global_residual = float(oof.mean() / y.mean() - 1.0)
    case_residuals = []
    for case in np.unique(groups):
        use = groups == case
        case_residuals.append(float(oof[use].mean() / y[use].mean() - 1.0))
    case_residuals = np.asarray(case_residuals)
    final = make_model(); final.fit(X, y)
    pred = final.predict(X)
    provenance = {
        "arm": args.arm, "catalogue": os.path.abspath(args.catalogue),
        "snc_lookup": os.path.abspath(args.snc_lookup), "features": features,
        "n_fit_rows": len(frame), "n_cases": int(frame["case"].nunique()),
        "max_case": args.max_case, "primary_mag_max": args.primary_mag_max,
        "primary_re_min": args.primary_re_min, "nominal_g": args.nominal_g,
        "label_mean": float(y.mean()), "label_std": float(y.std()),
        "snc_match_fraction": frame.attrs["snc_match_fraction"],
        "snc_selected_rows": frame.attrs["snc_selected_rows"],
        "snc_matched_rows": frame.attrs["snc_matched_rows"],
        "target_mean": float(pred.mean()), "target_min": float(pred.min()),
        "target_max": float(pred.max()), "oof_global_ratio_minus_one": global_residual,
        "oof_case_ratio_sem": float(case_residuals.std(ddof=1) / np.sqrt(len(case_residuals))),
        "firewall": "half-shear cases only; constgold not read",
    }
    payload = {"model": final, "features": features, "derived": {}, "provenance": provenance}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    joblib.dump(payload, args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"OOF target/label - 1 = {100*global_residual:+.4f}% +/- "
          f"{100*provenance['oof_case_ratio_sem']:.4f}% case SEM")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
