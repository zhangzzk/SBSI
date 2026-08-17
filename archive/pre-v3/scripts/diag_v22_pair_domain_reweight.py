"""Test exact pair-feature covariate shift between half-shear and anchors.

V2.2 closes against its held-out random-direction pair labels but misses the
local-10 anchor response.  This diagnostic asks whether the difference is just
occupancy in the seven exact V2.2 pair inputs.  A balanced domain classifier
estimates anchor/half-shear density ratios; held-out pair residuals are then
reweighted and summed at the anchor mean pair count.  Cases remain the unit of
uncertainty.  No correction is fit to anchor truth and constgold is never read.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils, nz_utils  # noqa: E402
from blendemu.response import angle_between  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402
from sbs_shear.domain import in_domain  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
BASE_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
MORPH_RAW = [
    "axis_ratio_input_p", "axis_ratio_input_s",
    "position_angle_input_p", "position_angle_input_s",
    "redshift_input_p", "redshift_input_s",
    "RA_input_p", "RA_input_s", "DEC_input_p", "DEC_input_s",
]
MORPH_FEATURES = [
    *BASE_FEATURES,
    "axis_ratio_input_p", "axis_ratio_input_s",
    "redshift_input_p", "redshift_input_s",
    "primary_sep_cos2", "primary_sep_sin2",
    "secondary_sep_cos2", "secondary_sep_sin2",
    "primary_secondary_cos2", "primary_secondary_sin2",
    "shear_sep_cos2", "shear_sep_sin2",
    "shear_primary_cos2", "shear_primary_sin2",
    "shear_secondary_cos2", "shear_secondary_sin2",
]
FEATURES = BASE_FEATURES
SOURCE_COLUMNS = BASE_FEATURES
KEY = ["case", "input_index"]
TILE = "tile180.0_-0.5"


def sem(values) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def transformed_features(frame: pd.DataFrame) -> np.ndarray:
    """Numerically stable representation of the requested domain features."""
    x = frame[FEATURES].to_numpy(np.float32, copy=True)
    for feature in ("Re_input_p", "Re_input_s", "sersic_n_input_p",
                    "sersic_n_input_s", "distance"):
        index = FEATURES.index(feature)
        x[:, index] = np.log(np.clip(x[:, index], 1.0e-4, None))
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite transformed pair feature")
    return x


def add_morphology_geometry(frame: pd.DataFrame, *, anchor: bool) -> pd.DataFrame:
    """Add rotation-invariant morphology/shear geometry omitted by V2.2."""
    out = frame.copy()
    deg = np.pi / 180.0
    primary_angle = out.position_angle_input_p.to_numpy(float) * deg
    secondary_angle = out.position_angle_input_s.to_numpy(float) * deg
    if anchor:
        primary_position = out[["RA_input_p", "DEC_input_p"]].to_numpy(float).T
        secondary_position = out[["RA_input_s", "DEC_input_s"]].to_numpy(float).T
        separation_angle = angle_between(primary_position, secondary_position)
        shear_angle = 0.5 * np.arctan2(
            out.gamma2_input_s.to_numpy(float), out.gamma1_input_s.to_numpy(float),
        )
    else:
        separation_angle = out.polarization_angle.to_numpy(float) * deg
        shear_angle = out.shear_angle.to_numpy(float) * deg
    angles = {
        "primary_sep": primary_angle - separation_angle,
        "secondary_sep": secondary_angle - separation_angle,
        "primary_secondary": primary_angle - secondary_angle,
        "shear_sep": shear_angle - separation_angle,
        "shear_primary": shear_angle - primary_angle,
        "shear_secondary": shear_angle - secondary_angle,
    }
    for name, value in angles.items():
        out[f"{name}_cos2"] = np.cos(2.0 * value).astype(np.float32)
        out[f"{name}_sin2"] = np.sin(2.0 * value).astype(np.float32)
    return out


def fit_density_odds(source_x: np.ndarray, target_x: np.ndarray, *, seed: int,
                     max_train_per_domain: int) -> tuple[HistGradientBoostingClassifier, dict]:
    rng = np.random.default_rng(seed)
    ns = min(len(source_x), max_train_per_domain)
    nt = min(len(target_x), max_train_per_domain)
    si = rng.choice(len(source_x), ns, replace=False)
    ti = rng.choice(len(target_x), nt, replace=False)
    x = np.concatenate([source_x[si], target_x[ti]])
    y = np.concatenate([np.zeros(ns, np.int8), np.ones(nt, np.int8)])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, random_state=seed,
    ).fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return model, {
        "n_source_sample": int(ns), "n_target_sample": int(nt),
        "heldout_auc": float(roc_auc_score(y_test, probability)),
    }


def predict_odds(model, x: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(x), np.float64)
    for start in range(0, len(x), chunk):
        p = model.predict_proba(x[start:start + chunk])[:, 1]
        out[start:start + chunk] = p / np.clip(1.0 - p, 1.0e-6, None)
    if not np.isfinite(out).all() or np.any(out < 0):
        raise RuntimeError("invalid density odds")
    return out


def weighted_case_estimate(frame: pd.DataFrame, weights: np.ndarray,
                           target_pairs_per_primary: float) -> dict:
    work = pd.DataFrame({
        "case": frame["case"].to_numpy(np.int16),
        "wr": weights * frame["residual"].to_numpy(float),
        "w": weights,
    })
    grouped = work.groupby("case", sort=True)[["wr", "w"]].sum()
    estimates = grouped.wr.to_numpy(float) / grouped.w.to_numpy(float)
    estimates *= float(target_pairs_per_primary)
    return stat(estimates)


def weight_diagnostics(source_x: np.ndarray, target_x: np.ndarray,
                       weights: np.ndarray) -> dict:
    # NumPy may accumulate float32 means in float32.  With millions of rows and
    # magnitude-scale values that loses enough precision to fake O(0.1) SMDs.
    source_x = np.asarray(source_x, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    total = float(weights.sum())
    ess = total * total / float(np.square(weights).sum())
    source_mean = np.average(source_x, axis=0, weights=weights)
    target_mean = target_x.mean(axis=0)
    source_var = np.average(np.square(source_x - source_mean), axis=0, weights=weights)
    target_var = target_x.var(axis=0)
    smd = (source_mean - target_mean) / np.sqrt(0.5 * (source_var + target_var) + 1e-12)
    return {
        "ess": float(ess), "ess_fraction": float(ess / len(weights)),
        "max_abs_smd": float(np.max(np.abs(smd))),
        "abs_smd_by_feature": {
            feature: float(abs(value)) for feature, value in zip(FEATURES, smd)
        },
        "weight_p50": float(np.quantile(weights, 0.50)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
    }


def load_halfshear_pairs(path: str, predictor: BlendingPredictor, case_max: int,
                         shear: float) -> tuple[pd.DataFrame, dict, pd.Series]:
    cuts, _, _ = predictor._select("regression")
    need = ["case", "input_index", "delta_et1", "delta_et2", *SOURCE_COLUMNS]
    parts = []
    primary_ids = {case: set() for case in range(case_max + 1)}
    residual_sum = {case: 0.0 for case in range(case_max + 1)}
    null_sum = {case: 0.0 for case in range(case_max + 1)}
    pair_count = {case: 0 for case in range(case_max + 1)}
    with ipc.open_file(path) as reader:
        missing = set(need) - set(reader.schema.names)
        if missing:
            raise KeyError(f"half-shear catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(need).to_pandas()
            if len(frame) == 0:
                continue
            if int(frame.case.min()) > case_max:
                break
            frame = frame[frame.case <= case_max]
            finite = np.isfinite(frame[["delta_et1", "delta_et2"]].to_numpy(float)).all(axis=1)
            frame = frame.loc[finite]
            primary_ok = (
                (frame.r_input_p > cuts[1][0]) & (frame.r_input_p < cuts[1][1])
                & (frame.Re_input_p > cuts[3][0]) & (frame.Re_input_p < cuts[3][1])
                & in_domain(frame.r_input_p.to_numpy(float), frame.Re_input_p.to_numpy(float))
            )
            eligible = frame.loc[primary_ok]
            for case, values in eligible.groupby("case", sort=False):
                primary_ids[int(case)].update(values.input_index.to_numpy(np.int64).tolist())
            pair_ok = primary_ok & (
                (frame.r_input_s > cuts[0][0]) & (frame.r_input_s < cuts[0][1])
                & (frame.Re_input_s > cuts[2][0]) & (frame.Re_input_s < cuts[2][1])
                & (frame.distance > cuts[4][0]) & (frame.distance < cuts[4][1])
            )
            pairs = frame.loc[pair_ok, need].copy()
            if len(pairs) == 0:
                continue
            prediction = predictor.predict_on_pairs(
                pairs[BASE_FEATURES], task="response", warn_extrapolation=False,
            ).response.to_numpy(float)
            pairs["residual"] = prediction - pairs.delta_et1.to_numpy(float) / shear
            if FEATURES is MORPH_FEATURES:
                pairs = add_morphology_geometry(pairs, anchor=False)
            pair_null = pairs.delta_et2.to_numpy(float) / shear
            for case, positions in pairs.groupby("case", sort=False).indices.items():
                positions = np.asarray(positions, int)
                case = int(case)
                residual_sum[case] += float(pairs.residual.to_numpy(float)[positions].sum())
                null_sum[case] += float(pair_null[positions].sum())
                pair_count[case] += int(len(positions))
            parts.append(pairs[["case", "residual", *FEATURES]].astype({"case": np.int16}))
            if (batch_index + 1) % 100 == 0:
                print(f"half-shear batches processed: {batch_index + 1}", flush=True)
    pairs = pd.concat(parts, ignore_index=True)
    rows = []
    for case in range(case_max + 1):
        n_primary = len(primary_ids[case])
        if n_primary == 0:
            raise RuntimeError(f"half-shear case {case}: no eligible primary")
        rows.append({
            "case": case, "n_primary": n_primary, "n_pairs": pair_count[case],
            "gap": residual_sum[case] / n_primary,
            "null": null_sum[case] / n_primary,
            "pairs_per_primary": pair_count[case] / n_primary,
        })
    case_table = pd.DataFrame(rows).set_index("case")
    metadata = {
        "n_pairs": int(len(pairs)),
        "n_primaries": int(case_table.n_primary.sum()),
        "pairs_per_primary_case_mean": float(case_table.pairs_per_primary.mean()),
        "gap": stat(case_table.gap), "null": stat(case_table.null),
    }
    return pairs, metadata, case_table


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def load_anchor_pairs(base: str, response10: list[str], response15: list[str],
                      predictor: BlendingPredictor, case_min: int, case_max: int,
                      sign: float) -> tuple[pd.DataFrame, dict]:
    common_parts = []
    for layer, (path10, path15) in enumerate(zip(response10, response15)):
        r10 = pd.read_feather(path10, columns=KEY + ["R_blend_lsst_r_extnbr_v22"])
        r15 = pd.read_feather(path15, columns=KEY)
        common = r10.merge(r15, on=KEY, validate="one_to_one")
        common["layer"] = layer
        common_parts.append(common)
    common = pd.concat(common_parts, ignore_index=True)
    if common.duplicated(KEY).any():
        raise RuntimeError("anchor response keys overlap across layers")

    cuts, r_max, k = predictor._select("regression")
    parts = []
    replay_max = 0.0
    case_counts = []
    for case in range(case_min, case_max + 1):
        observed = common[common.case == case]
        anchor_ids = observed.input_index.to_numpy(np.int64)
        frame = input_frame(base, case, sign)
        primary = frame[frame["index"].isin(anchor_ids)].copy()
        if len(primary) != len(anchor_ids):
            raise RuntimeError(f"anchor case {case}: input ids missing")
        raw = nz_utils.make_reg_features(
            primary, frame, r_max=float(r_max) / 3600.0, k=int(k),
        )
        pair_id = next(column for column in raw if column.startswith("index") and column.endswith("_p"))
        pairs = data_utils.source_select_reg(raw, cuts=cuts).copy()
        pairs["distance"] *= 3600.0
        prediction = predictor.predict_on_pairs(
            pairs[BASE_FEATURES], task="response", warn_extrapolation=False,
        ).response.to_numpy(float)
        pairs["prediction"] = prediction
        replay = pairs.groupby(pair_id, sort=False).prediction.sum().reindex(
            anchor_ids, fill_value=0.0,
        ).to_numpy(float)
        stored = observed.set_index("input_index").loc[
            anchor_ids, "R_blend_lsst_r_extnbr_v22"
        ].to_numpy(float)
        error = float(np.max(np.abs(replay - stored)))
        replay_max = max(replay_max, error)
        if not np.allclose(replay, stored, rtol=1e-7, atol=5e-7):
            raise RuntimeError(f"anchor case {case}: V2.2 replay failed, max abs={error:.3e}")
        if FEATURES is MORPH_FEATURES:
            pairs = add_morphology_geometry(pairs, anchor=True)
        pairs["case"] = np.int16(case)
        parts.append(pairs[["case", *FEATURES]])
        case_counts.append(len(pairs) / len(anchor_ids))
        print(
            f"anchor case {case}: anchors={len(anchor_ids):,} pairs={len(pairs):,} "
            f"pairs/anchor={case_counts[-1]:.3f} replay={error:.2e}", flush=True,
        )
    pairs = pd.concat(parts, ignore_index=True)
    return pairs, {
        "n_pairs": int(len(pairs)), "n_primaries": int(len(common)),
        "pairs_per_primary_case_mean": float(np.mean(case_counts)),
        "prediction_replay_max_abs": float(replay_max),
    }


def capped_results(source: pd.DataFrame, source_x: np.ndarray, target_x: np.ndarray,
                   odds: np.ndarray, target_count: float, caps: list[float | None]) -> dict:
    out = {}
    for cap in caps:
        weights = odds if cap is None else np.minimum(odds, cap)
        name = "raw" if cap is None else f"cap{cap:g}"
        out[name] = {
            "estimated_sum_residual": weighted_case_estimate(source, weights, target_count),
            **weight_diagnostics(source_x, target_x, weights),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--halfshear-catalogue", required=True)
    ap.add_argument("--anchor-base", required=True)
    ap.add_argument("--response10", action="append", required=True)
    ap.add_argument("--response15", action="append", required=True)
    ap.add_argument("--anchor-result", required=True)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--halfshear-case-max", type=int, default=39)
    ap.add_argument("--anchor-case-min", type=int, default=200)
    ap.add_argument("--anchor-case-max", type=int, default=299)
    ap.add_argument("--development-max", type=int, default=19)
    ap.add_argument("--shear", type=float, default=0.2)
    ap.add_argument("--anchor-sign", type=float, default=0.05)
    ap.add_argument("--max-train-per-domain", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument(
        "--feature-set", choices=("v22", "morphology"), default="v22",
        help="domain coordinates: exact V2.2 inputs or V2.2 plus omitted morphology/orientation",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.response10) != len(args.response15):
        raise RuntimeError("response10/15 argument counts differ")
    global FEATURES, SOURCE_COLUMNS
    if args.feature_set == "morphology":
        FEATURES = MORPH_FEATURES
        SOURCE_COLUMNS = list(dict.fromkeys([
            *BASE_FEATURES, *MORPH_RAW, "polarization_angle", "shear_angle",
        ]))
    else:
        FEATURES = BASE_FEATURES
        SOURCE_COLUMNS = BASE_FEATURES

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu",
    )
    half, half_meta, half_case = load_halfshear_pairs(
        args.halfshear_catalogue, predictor, args.halfshear_case_max, args.shear,
    )
    anchor, anchor_meta = load_anchor_pairs(
        args.anchor_base, args.response10, args.response15, predictor,
        args.anchor_case_min, args.anchor_case_max, args.anchor_sign,
    )
    half_x = transformed_features(half)
    anchor_x = transformed_features(anchor)

    model, domain = fit_density_odds(
        half_x, anchor_x, seed=args.seed,
        max_train_per_domain=args.max_train_per_domain,
    )
    odds = predict_odds(model, half_x)
    caps = [3.0, 5.0, 10.0, 20.0, None]
    anchor_reweight = capped_results(
        half, half_x, anchor_x, odds,
        anchor_meta["pairs_per_primary_case_mean"], caps,
    )

    development = half.case.to_numpy(int) <= args.development_max
    validation = ~development
    transfer_model, transfer_domain = fit_density_odds(
        half_x[development], half_x[validation], seed=args.seed + 1,
        max_train_per_domain=args.max_train_per_domain,
    )
    transfer_odds = predict_odds(transfer_model, half_x[development])
    validation_count = float(
        half_case.loc[half_case.index > args.development_max, "pairs_per_primary"].mean()
    )
    transfer_results = capped_results(
        half.loc[development].reset_index(drop=True), half_x[development],
        half_x[validation], transfer_odds, validation_count, caps,
    )
    validation_actual = stat(
        half_case.loc[half_case.index > args.development_max, "gap"].to_numpy(float)
    )

    with open(args.anchor_result, encoding="utf-8") as handle:
        anchor_result = json.load(handle)
    observed = anchor_result["v22_minus_local10"]
    primary = anchor_reweight["cap10"]["estimated_sum_residual"]
    combined_sem = float(np.hypot(primary["case_sem"], observed["case_sem"]))
    transfer_primary = transfer_results["cap10"]["estimated_sum_residual"]
    transfer_combined_sem = float(np.hypot(
        transfer_primary["case_sem"], validation_actual["case_sem"],
    ))
    gates = {
        "cap10_transfer_control_matches_validation_at_2_combined_sem": bool(
            abs(transfer_primary["mean"] - validation_actual["mean"])
            <= 2.0 * transfer_combined_sem
        ),
        "cap10_anchor_reweight_matches_observed_at_2_combined_sem": bool(
            abs(primary["mean"] - observed["mean"]) <= 2.0 * combined_sem
        ),
        "cap10_anchor_ess_fraction_above_20pct": bool(
            anchor_reweight["cap10"]["ess_fraction"] >= 0.20
        ),
        "cap10_anchor_max_abs_smd_below_0p05": bool(
            anchor_reweight["cap10"]["max_abs_smd"] <= 0.05
        ),
    }
    payload = {
        "question": "does occupancy in the requested pair coordinates explain the local10 anchor gap?",
        "tag": args.tag, "feature_set": args.feature_set, "features": FEATURES,
        "halfshear_cases": [0, args.halfshear_case_max],
        "anchor_cases": [args.anchor_case_min, args.anchor_case_max],
        "halfshear": half_meta, "anchor": anchor_meta,
        "anchor_domain_classifier": domain,
        "anchor_reweight": anchor_reweight,
        "observed_anchor_v22_minus_local10": observed,
        "cap10_anchor_difference_reweighted_minus_observed": float(primary["mean"] - observed["mean"]),
        "cap10_anchor_difference_combined_case_sem": combined_sem,
        "transfer_control": {
            "development_cases": [0, args.development_max],
            "validation_cases": [args.development_max + 1, args.halfshear_case_max],
            "domain_classifier": transfer_domain,
            "actual_validation_sum_residual": validation_actual,
            "reweighted_development": transfer_results,
        },
        "gates": gates,
        "ordinary_exact_pair_feature_shift_supported": bool(all(gates.values())),
        "correction_fitted": False,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_PAIR_DOMAIN_REWEIGHT_DONE", flush=True)


if __name__ == "__main__":
    main()
