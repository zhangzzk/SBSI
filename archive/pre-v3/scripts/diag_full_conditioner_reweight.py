"""Reweight half-shear residuals to constgold in the exact deployed conditioner space.

The V2.2/V3.0 checkpoints in this experiment condition on eight quantities:
intrinsic e1/e2, primary Sersic index, true primary magnitude/size, and the
near/far/maximum neighbour-flux summaries.  A case-cross-fitted domain
classifier estimates p_constgold(x) / p_halfshear(x).  We also repeat with the
deployed R_blend appended, since it is a response-regularization axis even
though it is not a conditioner of these checkpoints.

This is diagnostic importance weighting only.  It does not fit a response
correction, alter a score, render an image, or retrain a flow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle


KEYS = ["case", "input_index"]
FLOW_FEATURES = [
    "e1_input_p", "e2_input_p", "sersic_n_input_p", "r_input_p", "Re_input_p",
    "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
]


def case_filtered_table(path: str, columns: list[str], lo: int, hi: int) -> pd.DataFrame:
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        missing = set(columns) - set(reader.schema.names)
        if missing:
            raise KeyError(f"{path} missing {sorted(missing)}")
        for index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(index)]).select(columns)
            table = table.filter(pc.and_(
                pc.greater_equal(table["case"], pa.scalar(lo)),
                pc.less(table["case"], pa.scalar(hi)),
            ))
            if table.num_rows:
                parts.append(table)
    if not parts:
        raise RuntimeError(f"no rows in {path} for cases [{lo},{hi})")
    return pa.concat_tables(parts).to_pandas()


def read_unique(paths: list[str], columns: list[str]) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_feather(path, columns=columns) for path in sorted(paths)],
        ignore_index=True,
    )
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate keys across {len(paths)} files")
    return frame


def attach_score(features: pd.DataFrame, path: str, flow_column: str) -> pd.DataFrame:
    score = pd.read_feather(path, columns=KEYS + ["r_sim_self", flow_column])
    out = features.merge(score, on=KEYS, how="inner", validate="one_to_one", sort=False)
    if len(out) != len(features) or len(out) != len(score):
        raise RuntimeError(f"incomplete half-shear join for {path}: {len(out):,}")
    return out


def sample_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    index = np.flatnonzero(mask)
    if len(index) > limit:
        index = rng.choice(index, size=limit, replace=False)
    return index


def predict_chunked(model, values: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float32)
    for lo in range(0, len(values), chunk):
        hi = min(lo + chunk, len(values))
        out[lo:hi] = model.predict_proba(values[lo:hi])[:, 1].astype(np.float32)
    return out


def crossfit_odds(
    xh: np.ndarray,
    xc: np.ndarray,
    case_h: np.ndarray,
    case_c: np.ndarray,
    folds: int,
    train_per_domain: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    ph = np.full(len(xh), np.nan, dtype=np.float32)
    pcg = np.full(len(xc), np.nan, dtype=np.float32)
    fold_rows = []
    for fold in range(folds):
        rng = np.random.default_rng(seed + fold)
        ih = sample_indices(case_h % folds != fold, train_per_domain, rng)
        ic = sample_indices(case_c % folds != fold, train_per_domain, rng)
        xt = np.concatenate([xh[ih], xc[ic]])
        yt = np.concatenate([np.zeros(len(ih), np.int8), np.ones(len(ic), np.int8)])
        order = rng.permutation(len(yt))
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=seed + fold,
        )
        model.fit(xt[order], yt[order])
        uh = case_h % folds == fold
        uc = case_c % folds == fold
        ph[uh] = predict_chunked(model, xh[uh])
        pcg[uc] = predict_chunked(model, xc[uc])
        fold_rows.append({
            "fold": fold,
            "n_train_half": int(len(ih)),
            "n_train_constgold": int(len(ic)),
            "n_test_half": int(uh.sum()),
            "n_test_constgold": int(uc.sum()),
            "n_iter": int(model.n_iter_),
        })
        print(
            f"  fold {fold}: train={len(ih):,}+{len(ic):,}, "
            f"test={uh.sum():,}+{uc.sum():,}, iterations={model.n_iter_}",
            flush=True,
        )
        del model, xt, yt, order
    if not np.isfinite(ph).all() or not np.isfinite(pcg).all():
        raise RuntimeError("cross-fit probabilities are incomplete")

    rng = np.random.default_rng(seed + 1000)
    ah = sample_indices(np.ones(len(ph), bool), min(300_000, len(ph)), rng)
    ac = sample_indices(np.ones(len(pcg), bool), min(300_000, len(pcg)), rng)
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(ah)), np.ones(len(ac))]),
        np.concatenate([ph[ah], pcg[ac]]),
    )
    # Equal class counts were used in every training fold, so posterior odds
    # estimate p_constgold(x)/p_halfshear(x).  Normalization is immaterial for
    # weighted means and is deliberately done only at use time.
    pclip = np.clip(ph.astype(float), 1.0e-5, 1.0 - 1.0e-5)
    odds = pclip / (1.0 - pclip)
    return odds, {
        "crossfit_auc": float(auc),
        "mean_p_const_on_half": float(ph.mean()),
        "mean_p_const_on_constgold": float(pcg.mean()),
        "folds": fold_rows,
        "raw_odds_quantiles": {
            str(q): float(np.quantile(odds, q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    return float(np.sum(values[good] * weights[good]) / np.sum(weights[good]))


def effective_n(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def feature_balance(
    xh: np.ndarray,
    xc: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    seed: int,
) -> dict:
    rows = []
    rng = np.random.default_rng(seed)
    # Fixed pooled-sample edges make a cheap marginal-distribution check in
    # addition to standardized means.  TV=0 is identical; TV=1 is disjoint.
    sh = sample_indices(np.ones(len(xh), bool), min(150_000, len(xh)), rng)
    sc = sample_indices(np.ones(len(xc), bool), min(150_000, len(xc)), rng)
    for j, name in enumerate(names):
        h, c = xh[:, j].astype(float), xc[:, j].astype(float)
        mh, mc = h.mean(), c.mean()
        scale = np.sqrt(0.5 * (h.var() + c.var()))
        mw = weighted_mean(h, weights)
        smd0 = (mh - mc) / scale if scale > 0 else 0.0
        smdw = (mw - mc) / scale if scale > 0 else 0.0
        edges = np.unique(np.quantile(
            np.concatenate([h[sh], c[sc]]), np.linspace(0.0, 1.0, 41)
        ))
        if len(edges) > 1:
            hh = np.histogram(h, bins=edges)[0].astype(float)
            hc = np.histogram(c, bins=edges)[0].astype(float)
            hw = np.histogram(h, bins=edges, weights=weights)[0].astype(float)
            hh /= hh.sum(); hc /= hc.sum(); hw /= hw.sum()
            tv0 = 0.5 * np.abs(hh - hc).sum()
            tvw = 0.5 * np.abs(hw - hc).sum()
        else:
            tv0 = tvw = 0.0
        rows.append({
            "feature": name,
            "half_mean": float(mh),
            "constgold_mean": float(mc),
            "weighted_half_mean": float(mw),
            "smd_unweighted": float(smd0),
            "smd_weighted": float(smdw),
            "marginal_tv_unweighted": float(tv0),
            "marginal_tv_weighted": float(tvw),
        })
    return {
        "features": rows,
        "max_abs_smd_unweighted": float(max(abs(row["smd_unweighted"]) for row in rows)),
        "max_abs_smd_weighted": float(max(abs(row["smd_weighted"]) for row in rows)),
        "max_tv_unweighted": float(max(row["marginal_tv_unweighted"] for row in rows)),
        "max_tv_weighted": float(max(row["marginal_tv_weighted"] for row in rows)),
    }


def mean_sem(values: list[float]) -> dict:
    x = np.asarray(values, float)
    return {
        "mean": float(x.mean()),
        "seed_sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
        "seed_values": x.tolist(),
    }


def closure_reference(path: str, arms: list[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    out = {}
    for arm in arms:
        rows = data["constgold"][arm]["profile_by_R_blend"]["bins"]
        n = sum(row["n"] for row in rows)
        out[arm] = {
            "total_closure_residual": float(sum(
                row["n"] * row["flow_minus_demand_variant"]["mean"] for row in rows
            ) / n),
            "m": data["constgold"][arm]["m_variant"],
        }
    rows = data["constgold"][arms[0]]["profile_by_R_blend"]["bins"]
    n = sum(row["n"] for row in rows)
    out["v22"] = {"total_closure_residual": float(sum(
        row["n"] * row["flow_minus_demand_baseline"]["mean"] for row in rows
    ) / n)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-half", required=True)
    ap.add_argument("--features-const", required=True)
    ap.add_argument("--const-catalogue", required=True)
    ap.add_argument("--crowd-lookup", required=True)
    ap.add_argument("--constgold-glob", required=True)
    ap.add_argument("--baseline-template", required=True)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--closure-json", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=140)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-per-domain", type=int, default=350_000)
    ap.add_argument("--classifier-seed", type=int, default=31008)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    half_columns = KEYS + [
        "e1_input_rot0_p", "e2_input_rot0_p", "sersic_n_input_p",
        "r_input_p", "Re_input_p", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
        "r_blend",
    ]
    print("Loading exact half-shear conditioner rows", flush=True)
    half = pd.read_feather(args.features_half, columns=half_columns).rename(columns={
        "e1_input_rot0_p": "e1_input_p", "e2_input_rot0_p": "e2_input_p",
        "r_blend": "R_blend",
    })
    if half.duplicated(KEYS).any():
        raise RuntimeError("duplicate half-shear keys")

    print("Loading exact constgold scored population", flush=True)
    score_paths = glob.glob(args.constgold_glob)
    if len(score_paths) != 10:
        raise RuntimeError(f"expected ten constgold shards, found {len(score_paths)}")
    const = read_unique(score_paths, KEYS + ["R_blend"])
    primary = pd.read_feather(args.features_const, columns=KEYS + ["r_input_p", "Re_input_p"])
    const = const.merge(primary, on=KEYS, how="left", validate="one_to_one", sort=False)

    intrinsic = case_filtered_table(
        args.const_catalogue,
        KEYS + ["axis_ratio_input_p", "position_angle_input_p", "sersic_n_input_p"],
        args.min_case,
        args.max_case,
    )
    e1, e2 = ellipticity_from_axis_ratio_angle(
        intrinsic["axis_ratio_input_p"].to_numpy(float),
        intrinsic["position_angle_input_p"].to_numpy(float),
    )
    intrinsic["e1_input_p"] = e1
    intrinsic["e2_input_p"] = e2
    intrinsic = intrinsic[KEYS + ["e1_input_p", "e2_input_p", "sersic_n_input_p"]]
    const = const.merge(intrinsic, on=KEYS, how="left", validate="one_to_one", sort=False)
    del intrinsic

    crowd = case_filtered_table(
        args.crowd_lookup,
        KEYS + ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"],
        args.min_case,
        args.max_case,
    )
    const = const.merge(crowd, on=KEYS, how="left", validate="one_to_one", sort=False)
    del crowd, primary
    needed = FLOW_FEATURES + ["R_blend"]
    if const[needed].isna().any().any():
        missing = const[needed].isna().mean()
        raise RuntimeError(f"constgold feature join incomplete: {missing[missing > 0].to_dict()}")
    if not np.isfinite(half[needed].to_numpy(float)).all():
        raise RuntimeError("non-finite half-shear conditioner")
    if not np.isfinite(const[needed].to_numpy(float)).all():
        raise RuntimeError("non-finite constgold conditioner")
    print(f"  half={len(half):,}, constgold={len(const):,}", flush=True)

    first_path = args.baseline_template.format(seed=args.seeds[0])
    first = attach_score(half[KEYS], first_path, f"R_flow_s{args.seeds[0]}")
    truth = first["r_sim_self"].to_numpy(float)
    model_residuals: dict[str, list[tuple[int, np.ndarray]]] = {"v22": []}
    for arm in args.arms:
        model_residuals[arm] = []
    for seed in args.seeds:
        print(f"Loading response residuals for seed {seed}", flush=True)
        base = attach_score(
            half[KEYS], args.baseline_template.format(seed=seed), f"R_flow_s{seed}"
        )
        if not np.allclose(base["r_sim_self"], truth, equal_nan=True):
            raise RuntimeError("half-shear truth differs by seed")
        model_residuals["v22"].append(
            (seed, base[f"R_flow_s{seed}"].to_numpy(float) - truth)
        )
        del base
        for arm in args.arms:
            path = os.path.join(args.variant_dir, f"{arm}_s{seed}.feather")
            frame = attach_score(half[KEYS], path, f"R_flow_s{seed}")
            model_residuals[arm].append(
                (seed, frame[f"R_flow_s{seed}"].to_numpy(float) - truth)
            )
            del frame

    result = {
        "status": "diagnostic importance weighting only; no correction fitted or applied",
        "method": (
            "case-cross-fitted balanced domain classifier; odds estimate "
            "p_constgold(x)/p_halfshear(x)"
        ),
        "seeds": args.seeds,
        "constgold_reference": closure_reference(args.closure_json, args.arms),
        "feature_sets": {},
    }
    sets = {
        "deployed_flow8": FLOW_FEATURES,
        "deployed_flow8_plus_R_blend": FLOW_FEATURES + ["R_blend"],
    }
    case_h = half["case"].to_numpy(np.int64)
    case_c = const["case"].to_numpy(np.int64)
    for set_index, (name, features) in enumerate(sets.items()):
        print(f"\nCross-fitting density ratio: {name} ({len(features)}D)", flush=True)
        xh = half[features].to_numpy(np.float32)
        xc = const[features].to_numpy(np.float32)
        raw, classifier = crossfit_odds(
            xh, xc, case_h, case_c, args.folds, args.train_per_domain,
            args.classifier_seed + 100 * set_index,
        )
        entry = {"features": features, "classifier": classifier, "caps": {}}
        for cap in (5.0, 10.0, 20.0, 50.0, None):
            label = "raw" if cap is None else f"cap_{int(cap)}"
            weights = raw if cap is None else np.minimum(raw, cap)
            models = {}
            for model, seed_arrays in model_residuals.items():
                vals = [weighted_mean(residual, weights) for _, residual in seed_arrays]
                models[model] = mean_sem(vals)
            entry["caps"][label] = {
                "weight_cap": cap,
                "effective_n": effective_n(weights),
                "effective_fraction": effective_n(weights) / len(weights),
                "truth_mean": weighted_mean(truth, weights),
                "models": models,
            }
            if label in {"cap_10", "raw"}:
                entry["caps"][label]["balance"] = feature_balance(
                    xh, xc, weights, features, args.classifier_seed + 2000 + set_index
                )
            print(
                f"  {label}: ESS={entry['caps'][label]['effective_fraction']:.1%}; "
                + "; ".join(
                    f"{model}={models[model]['mean']:+.6f}" for model in models
                ),
                flush=True,
            )
        result["feature_sets"][name] = entry
        del xh, xc, raw

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
