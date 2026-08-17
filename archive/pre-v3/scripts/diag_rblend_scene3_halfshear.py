"""Compare V2.2 and scene-aware BlendEMU response to the held-out half-shear ruler.

This is deliberately simulation-only.  It uses the cached ap7 pair ruler, sums the
per-neighbour response over each primary, and attaches intrinsic full-scene flux
features by (case, input_index).  No constgold catalogue or response is read.

The headline calibration axis is the *baseline predicted summed response*.  Truth
quantiles are not used because the per-primary half-shear ruler is noisy and binning
on that noisy dependent variable would create regression-to-the-mean artefacts.
Uncertainties are delete-one-case jackknife standard errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


PAIR_FEATURES = [
    "Re_input_p", "r_input_p", "sersic_n_input_p",
    "Re_input_s", "r_input_s", "sersic_n_input_s", "distance",
]
SCENE_SOURCE = ["logflux_near_0_1", "logflux_mid_1_3", "logflux_far_3_10"]
SCENE_MODEL = [f"scene_{name}_input_p" for name in SCENE_SOURCE]
PAIRSET_COLUMNS = [
    "case", "input_index", "pid", "blend_truth", "blend_null", *PAIR_FEATURES,
]
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"

# Exact V2.2 regression support, in the source_select_reg convention.
V22_CUTS = {
    "r_input_s": (13.0, 29.0),
    "r_input_p": (18.0, 25.8),
    "Re_input_s": (0.0, 10.0),
    "Re_input_p": (0.5, 1.5),
    "distance": (0.0, 10.0),
}

# Fixed edges from the established V2.2 constgold blendness diagnostic.  They are
# used only as an axis definition, never to train or calibrate either model.
OLD_BLENDNESS_EDGES = np.array([-np.inf, 0.0200, 0.0455, 0.1026, 0.3127, np.inf])


def read_pairset(path: str, case_min: int, case_max: int) -> pd.DataFrame:
    parts = []
    nraw = 0
    t0 = time.time()
    with ipc.open_file(path) as reader:
        missing = set(PAIRSET_COLUMNS) - set(reader.schema.names)
        if missing:
            raise KeyError(f"pairset lacks {sorted(missing)}")
        for bi in range(reader.num_record_batches):
            batch = reader.get_batch(bi)
            # The pairset is case-sorted, but use row masks rather than relying on it.
            b = pa.Table.from_batches([batch]).select(PAIRSET_COLUMNS).to_pandas()
            nraw += len(b)
            c = b["case"].to_numpy(np.int32)
            if c.min() >= case_max:
                break
            b = b[(c >= case_min) & (c < case_max)]
            if len(b):
                parts.append(b)
    if not parts:
        raise RuntimeError("no pairset rows in requested case window")
    out = pd.concat(parts, ignore_index=True)
    mask = np.isfinite(out["blend_truth"].to_numpy(float))
    mask &= np.isfinite(out["blend_null"].to_numpy(float))
    for col, (lo, hi) in V22_CUTS.items():
        v = out[col].to_numpy(float)
        mask &= np.isfinite(v) & (v >= lo) & (v <= hi)
    out = out.loc[mask].reset_index(drop=True)
    print(f"pairset: scanned {nraw:,}, kept {len(out):,} V2.2-support rows "
          f"in cases [{case_min},{case_max}) ({time.time()-t0:.1f}s)", flush=True)
    return out


def read_scene_lookup(paths: list[str], case_min: int, case_max: int) -> pd.DataFrame:
    parts = []
    t0 = time.time()
    need = ["case", "input_index", *SCENE_SOURCE]
    for path in paths:
        with ipc.open_file(path) as reader:
            missing = set(need) - set(reader.schema.names)
            if missing:
                raise KeyError(f"{path} lacks {sorted(missing)}")
            for bi in range(reader.num_record_batches):
                b = pa.Table.from_batches([reader.get_batch(bi)]).select(need).to_pandas()
                c = b["case"].to_numpy(np.int32)
                b = b[(c >= case_min) & (c < case_max)]
                if len(b):
                    parts.append(b)
    if not parts:
        raise RuntimeError("no scene lookup rows in requested case window")
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["case", "input_index"]).any():
        raise RuntimeError("scene lookup has duplicate (case,input_index) keys")
    print(f"scene lookup: {len(out):,} unique keys ({time.time()-t0:.1f}s)", flush=True)
    return out


def attach_scene(pair: pd.DataFrame, scene: pd.DataFrame) -> pd.DataFrame:
    primary = pair[["case", "input_index"]].drop_duplicates()
    matched = primary.merge(scene, on=["case", "input_index"], how="left", validate="one_to_one",
                            indicator=True)
    bad = matched["_merge"].ne("both")
    if bad.any():
        sample = matched.loc[bad, ["case", "input_index"]].head().to_dict("records")
        raise RuntimeError(f"scene lookup misses {bad.sum():,}/{len(matched):,} primaries: {sample}")
    matched = matched.drop(columns="_merge").rename(
        columns=dict(zip(SCENE_SOURCE, SCENE_MODEL)))
    out = pair.merge(matched, on=["case", "input_index"], how="left", validate="many_to_one")
    if not np.isfinite(out[SCENE_MODEL].to_numpy(float)).all():
        raise RuntimeError("non-finite scene features after exact join")
    print(f"scene join: exact coverage for {len(matched):,} primaries / {len(out):,} pairs",
          flush=True)
    return out


def score(pair: pd.DataFrame, tag: str, chunk: int) -> np.ndarray:
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(MODEL_DIR, tag=tag, conditions=COND, device="cpu")
    raw_scene = [n for n in predictor.bst_reg.feature_names if n.startswith("scene_")]
    cols = [*PAIR_FEATURES, *raw_scene]
    missing = set(cols) - set(pair.columns)
    if missing:
        raise KeyError(f"model {tag} needs absent raw features {sorted(missing)}")
    ans = np.empty(len(pair), np.float32)
    t0 = time.time()
    for lo in range(0, len(pair), chunk):
        hi = min(lo + chunk, len(pair))
        ans[lo:hi] = predictor.predict_on_pairs(
            pair.iloc[lo:hi][cols].copy(), task="response")["response"].to_numpy(np.float32)
        print(f"  {tag}: {hi:,}/{len(pair):,} ({time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(ans).all():
        raise RuntimeError(f"{tag} produced non-finite predictions")
    return ans


def factorize_primary(pair: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    keys = pd.MultiIndex.from_frame(pair[["case", "input_index"]])
    pidx, unique = pd.factorize(keys, sort=False)
    prim = unique.to_frame(index=False)
    prim.columns = ["case", "input_index"]
    # factorize assigns contiguous ids in first-seen order, so the returned first
    # positions are aligned with primary ids 0..N-1.
    _, first = np.unique(pidx, return_index=True)
    for col in ["r_input_p", "Re_input_p", "sersic_n_input_p", *SCENE_MODEL]:
        prim[col] = pair[col].to_numpy()[first]
    return pidx.astype(np.int32), prim


def jackknife_mean(values: np.ndarray, cases: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    good = mask & np.isfinite(values)
    n = int(good.sum())
    if n == 0:
        return np.nan, np.nan, 0
    total = float(values[good].sum(dtype=np.float64))
    mean = total / n
    uc, inv = np.unique(cases[good], return_inverse=True)
    if len(uc) < 2:
        return mean, np.nan, n
    cs = np.bincount(inv, weights=values[good], minlength=len(uc))
    cn = np.bincount(inv, minlength=len(uc))
    keep = n - cn
    valid = keep > 0
    loo = (total - cs[valid]) / keep[valid]
    center = loo.mean()
    sem = np.sqrt((len(loo) - 1.0) / len(loo) * np.square(loo - center).sum())
    return mean, float(sem), n


def binned_table(name: str, key: np.ndarray, edges: np.ndarray, series: dict[str, np.ndarray],
                 cases: np.ndarray, level: str) -> pd.DataFrame:
    rows = []
    idx = np.digitize(key, edges[1:-1], right=False)
    for bi in range(len(edges) - 1):
        m = idx == bi
        row = dict(axis=name, level=level, bin=bi, lo=float(edges[bi]), hi=float(edges[bi + 1]),
                   x_mean=float(np.mean(key[m])) if m.any() else np.nan)
        for label, values in series.items():
            mean, sem, n = jackknife_mean(values, cases, m)
            row[f"{label}_mean"] = mean
            row[f"{label}_sem"] = sem
            row["n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_edges(x: np.ndarray, nbin: int) -> np.ndarray:
    edges = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    if np.unique(edges).size != edges.size:
        raise RuntimeError("non-unique quantile edges")
    return edges


def panel(ax, tab: pd.DataFrame, xlabel: str, level: str, *, zero=False) -> None:
    t = tab[(tab.level == level) & (tab.n >= 200)].copy()
    x = np.arange(len(t))
    colors = {"truth": "#000000", "v22": "#0072B2", "scene3": "#D55E00"}
    styles = {"truth": ("o", "-"), "v22": ("s", "--"), "scene3": ("^", "-.")}
    for label in ("truth", "v22", "scene3"):
        marker, ls = styles[label]
        ax.errorbar(x, t[f"{label}_mean"], yerr=t[f"{label}_sem"], label=label,
                    color=colors[label], marker=marker, linestyle=ls, lw=1.2, ms=4, capsize=2)
    if zero:
        ax.axhline(0, color="0.75", lw=0.8)
    labels = []
    for lo, hi in zip(t.lo, t.hi):
        l = "-inf" if np.isneginf(lo) else f"{lo:g}"
        h = "inf" if np.isposinf(hi) else f"{hi:g}"
        labels.append(f"{l}–{h}")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mean $R_{\rm blend}$")
    ax.spines[["top", "right"]].set_visible(False)


def calibration_panel(ax, tab: pd.DataFrame) -> None:
    t = tab[(tab.axis == "predicted_blendness_decile") & (tab.n >= 200)].copy()
    colors = {"truth": "#000000", "v22": "#0072B2", "scene3": "#D55E00"}
    markers = {"truth": "o", "v22": "s", "scene3": "^"}
    x = t["v22_mean"].to_numpy(float)
    for label in ("truth", "v22", "scene3"):
        ax.errorbar(x, t[f"{label}_mean"], yerr=t[f"{label}_sem"], color=colors[label],
                    marker=markers[label], lw=1.2, ms=4, capsize=2, label=label)
    lo = min(np.nanmin(x), np.nanmin(t["truth_mean"]))
    hi = max(np.nanmax(x), np.nanmax(t["truth_mean"]))
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.9, ls=":", label="1:1")
    ax.set_xlabel(r"V2.2 predicted summed $R_{\rm blend}$")
    ax.set_ylabel(r"Mean summed $R_{\rm blend}$")
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(tables: pd.DataFrame, output_prefix: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), constrained_layout=True)
    calibration_panel(axes[0, 0], tables)
    panel(axes[0, 1], tables[tables.axis == "old_blendness"],
          "V2.2 predicted blendness bin", "primary", zero=True)
    panel(axes[0, 2], tables[tables.axis == "primary_mag"], r"Primary true $r$ (mag)",
          "primary", zero=True)
    panel(axes[0, 3], tables[tables.axis == "primary_size"],
          r"Primary true $R_e$ (arcsec)", "primary", zero=True)
    panel(axes[1, 0], tables[tables.axis == "separation"], "Pair separation (arcsec)",
          "pair", zero=True)
    panel(axes[1, 1], tables[tables.axis == "secondary_mag"], r"Secondary true $r$ (mag)",
          "pair", zero=True)
    panel(axes[1, 2], tables[tables.axis == "secondary_size"],
          r"Secondary true $R_e$ (arcsec)", "pair", zero=True)
    panel(axes[1, 3], tables[tables.axis == "mag_contrast"],
          r"$r_s-r_p$ (mag)", "pair", zero=True)
    for label, ax in zip("ABCDEFGH", axes.flat):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Held-out half-shear ruler (cases 0–39; exact V2.2 support)", y=1.055)
    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairset", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
                                               "blendflow/blend_pairset_ap7.feather")
    ap.add_argument("--scene-lookup", action="append", required=True)
    ap.add_argument("--case-min", type=int, default=0)
    ap.add_argument("--case-max", type=int, default=40)
    ap.add_argument("--tag-baseline", default="lsst_r_extnbr_v22")
    ap.add_argument("--tag-scene", default="lsst_r_extnbr_v22_scene3")
    ap.add_argument("--score-chunk", type=int, default=500_000)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    pair = attach_scene(read_pairset(args.pairset, args.case_min, args.case_max),
                        read_scene_lookup(args.scene_lookup, args.case_min, args.case_max))
    pair["pred_v22"] = score(pair, args.tag_baseline, args.score_chunk)
    pair["pred_scene3"] = score(pair, args.tag_scene, args.score_chunk)

    pidx, prim = factorize_primary(pair)
    npid = len(prim)
    prim["truth"] = np.bincount(pidx, weights=pair["blend_truth"], minlength=npid)
    prim["null"] = np.bincount(pidx, weights=pair["blend_null"], minlength=npid)
    prim["v22"] = np.bincount(pidx, weights=pair["pred_v22"], minlength=npid)
    prim["scene3"] = np.bincount(pidx, weights=pair["pred_scene3"], minlength=npid)
    prim["k"] = np.bincount(pidx, minlength=npid)

    primary_series = {k: prim[k].to_numpy(float) for k in ("truth", "null", "v22", "scene3")}
    primary_series.update({
        "v22_minus_truth": primary_series["v22"] - primary_series["truth"],
        "scene3_minus_truth": primary_series["scene3"] - primary_series["truth"],
        "scene3_minus_v22": primary_series["scene3"] - primary_series["v22"],
    })
    pair_series = {
        "truth": pair["blend_truth"].to_numpy(float),
        "null": pair["blend_null"].to_numpy(float),
        "v22": pair["pred_v22"].to_numpy(float),
        "scene3": pair["pred_scene3"].to_numpy(float),
    }
    pair_series.update({
        "v22_minus_truth": pair_series["v22"] - pair_series["truth"],
        "scene3_minus_truth": pair_series["scene3"] - pair_series["truth"],
        "scene3_minus_v22": pair_series["scene3"] - pair_series["v22"],
    })
    pcases = prim["case"].to_numpy(np.int32)
    rcases = pair["case"].to_numpy(np.int32)

    tables = []
    qedges = quantile_edges(prim["v22"].to_numpy(float), 10)
    tables.append(binned_table("predicted_blendness_decile", prim["v22"].to_numpy(float), qedges,
                               primary_series, pcases, "primary"))
    tables.append(binned_table("old_blendness", prim["v22"].to_numpy(float), OLD_BLENDNESS_EDGES,
                               primary_series, pcases, "primary"))
    tables.append(binned_table("primary_mag", prim["r_input_p"].to_numpy(float),
                               np.array([18, 22, 23, 24, 25, 25.8]), primary_series, pcases,
                               "primary"))
    tables.append(binned_table("primary_size", prim["Re_input_p"].to_numpy(float),
                               np.array([0.5, 0.6, 0.75, 1.0, 1.5]), primary_series, pcases,
                               "primary"))
    tables.append(binned_table("separation", pair["distance"].to_numpy(float),
                               np.array([0, 1, 2, 3, 4, 5, 7]), pair_series, rcases, "pair"))
    tables.append(binned_table("secondary_mag", pair["r_input_s"].to_numpy(float),
                               np.array([13, 18, 22, 24, 26, 28, 29]), pair_series, rcases,
                               "pair"))
    tables.append(binned_table("secondary_size", pair["Re_input_s"].to_numpy(float),
                               np.array([0, 0.1, 0.3, 0.5, 1, 1.5, 3, 10]), pair_series, rcases,
                               "pair"))
    contrast = pair["r_input_s"].to_numpy(float) - pair["r_input_p"].to_numpy(float)
    tables.append(binned_table("mag_contrast", contrast,
                               np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8, 16]), pair_series,
                               rcases, "pair"))
    tables = pd.concat(tables, ignore_index=True)

    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    tables.to_csv(str(out) + ".csv", index=False)
    summary = {
        "case_window": [args.case_min, args.case_max],
        "pair_rows": int(len(pair)), "primaries": int(npid),
        "mean_pairs_per_primary": float(prim["k"].mean()),
        "old_blendness_edges": OLD_BLENDNESS_EDGES.tolist(),
        "calibration_decile_edges": qedges.tolist(),
        "overall": {},
    }
    for label in ("truth", "null", "v22", "scene3"):
        mean, sem, n = jackknife_mean(prim[label].to_numpy(float), pcases,
                                      np.ones(npid, bool))
        summary["overall"][label] = {"mean": mean, "case_jackknife_sem": sem, "n": n}
    for label in ("v22", "scene3"):
        diff = prim[label].to_numpy(float) - prim["truth"].to_numpy(float)
        mean, sem, n = jackknife_mean(diff, pcases, np.ones(npid, bool))
        summary["overall"][label + "_minus_truth"] = {
            "mean": mean, "case_jackknife_sem": sem, "n": n,
            "relative_percent": 100 * mean / summary["overall"]["truth"]["mean"],
        }
    delta = prim["scene3"].to_numpy(float) - prim["v22"].to_numpy(float)
    mean, sem, n = jackknife_mean(delta, pcases, np.ones(npid, bool))
    summary["overall"]["scene3_minus_v22"] = {
        "mean": mean, "case_jackknife_sem": sem, "n": n,
        "relative_to_v22_percent": 100 * mean / summary["overall"]["v22"]["mean"],
    }
    with open(str(out) + ".json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    make_figure(tables, str(out))
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"saved {out}.csv/.json/.png/.pdf", flush=True)
    print("RBLEND_SCENE3_HALFSHEAR_DONE", flush=True)


if __name__ == "__main__":
    main()
