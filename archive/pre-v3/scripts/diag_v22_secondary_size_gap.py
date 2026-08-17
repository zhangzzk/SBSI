"""Test whether compact secondaries carry the V2.2 coherent-anchor gap.

Two deliberately separate data products are made.

1. The exact half-shear c40--199 training population is streamed from the
   response catalogue.  Its measured pair label (delta_et1 / shear) and the
   frozen V2.2 pair prediction are profiled against secondary input size.  Both
   a conditional per-pair residual and its additive contribution per eligible
   primary are retained.
2. The coherent-anchor feature table is cut on the response-dominant
   secondary's input size.  Anchors with no deployed pair are retained.  The
   residual before and after removing Re_s < 0.4 arcsec is evaluated separately
   on train, tuning, and final-test case windows.

The half-shear plot is an in-sample model audit.  The anchor cut was suggested
after inspecting the final-test conditional curves and is therefore explicitly
post-hoc rather than a new confirmatory test.  Constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy import stats

from sbs_shear.domain import in_domain


BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
MODEL_DIR = os.path.join(BLENDEMU_ROOT, "models")
COND = dict(
    pixel_size=0.2,
    zero_point=30.0,
    psf_fwhm=0.73,
    moffat_beta=2.224,
    pixel_rms=0.312,
)
PAIR_FEATURES = [
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
]
PAIR_COLUMNS = [
    "case",
    "input_index",
    "delta_et1",
    "delta_et2",
    *PAIR_FEATURES,
]
CUT_NAMES = [
    "r_input_s",
    "r_input_p",
    "Re_input_s",
    "Re_input_p",
    "distance",
]
SIZE_EDGES = np.asarray(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0,
     1.25, 1.5, 2.0, 3.0, 5.0, 10.0],
    dtype=float,
)
ANCHOR_SPLITS = {
    "train": (400, 599),
    "tune": (600, 699),
    "final_test": (700, 899),
    "all": (400, 899),
}


def finite_stat(values: np.ndarray, *, test_zero: bool = True) -> dict[str, Any]:
    """Summarize independent case values with a case-level SEM."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite case values")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    result: dict[str, Any] = {
        "mean": mean,
        "case_sd": sd,
        "case_sem": sd / np.sqrt(values.size),
        "n_cases": int(values.size),
    }
    if test_zero:
        if sd == 0.0:
            result.update({
                "t": 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf),
                "p": 1.0 if mean == 0.0 else 0.0,
            })
        else:
            test = stats.ttest_1samp(values, 0.0)
            result.update({"t": float(test.statistic), "p": float(test.pvalue)})
    return result


def ratio_by_case(numerator: np.ndarray, denominator: np.ndarray,
                  *, zero_if_empty: bool = False) -> np.ndarray:
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator shapes differ")
    if zero_if_empty:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    good = denominator > 0
    return numerator[good] / denominator[good]


def summarize_half_shear_accumulators(
    counts: np.ndarray,
    label_sum: np.ndarray,
    prediction_sum: np.ndarray,
    null_sum: np.ndarray,
    size_sum: np.ndarray,
    n_primary: np.ndarray,
    edges: np.ndarray,
    cut: float,
) -> dict[str, Any]:
    """Turn case x size-bin accumulators into curves and additive groups."""
    arrays = (counts, label_sum, prediction_sum, null_sum, size_sum)
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("half-shear accumulators have inconsistent shapes")
    counts = np.asarray(counts, np.int64)
    label_sum = np.asarray(label_sum, float)
    prediction_sum = np.asarray(prediction_sum, float)
    null_sum = np.asarray(null_sum, float)
    size_sum = np.asarray(size_sum, float)
    n_primary = np.asarray(n_primary, np.int64)
    edges = np.asarray(edges, float)
    if counts.ndim != 2 or counts.shape[1] != len(edges) - 1:
        raise ValueError("size edges do not match accumulator columns")
    if n_primary.shape != (counts.shape[0],) or np.any(n_primary <= 0):
        raise ValueError("every case must have eligible primaries")
    cut_locations = np.flatnonzero(np.isclose(edges, cut, rtol=0, atol=1e-12))
    if len(cut_locations) != 1 or cut_locations[0] in (0, len(edges) - 1):
        raise ValueError("cut must be one interior size edge")
    cut_index = int(cut_locations[0])

    residual_sum = label_sum - prediction_sum

    def group_summary(indices: np.ndarray, name: str) -> dict[str, Any]:
        local_count = counts[:, indices].sum(axis=1)
        local_label = label_sum[:, indices].sum(axis=1)
        local_prediction = prediction_sum[:, indices].sum(axis=1)
        local_null = null_sum[:, indices].sum(axis=1)
        local_residual = local_label - local_prediction
        pair_label = ratio_by_case(local_label, local_count)
        pair_prediction = ratio_by_case(local_prediction, local_count)
        pair_residual = ratio_by_case(local_residual, local_count)
        additive_label = local_label / n_primary
        additive_prediction = local_prediction / n_primary
        additive_residual = local_residual / n_primary
        return {
            "name": name,
            "n_pairs": int(local_count.sum()),
            "n_primaries": int(n_primary.sum()),
            "pooled_pairs_per_primary": float(
                local_count.sum() / n_primary.sum()
            ),
            "pair_fraction": float(local_count.sum() / counts.sum()),
            "pair_label": finite_stat(pair_label, test_zero=False),
            "pair_prediction": finite_stat(pair_prediction, test_zero=False),
            "pair_label_minus_prediction": finite_stat(pair_residual),
            "additive_label_per_primary": finite_stat(
                additive_label, test_zero=False
            ),
            "additive_prediction_per_primary": finite_stat(
                additive_prediction, test_zero=False
            ),
            "additive_label_minus_prediction_per_primary": finite_stat(
                additive_residual
            ),
            "additive_null_per_primary": finite_stat(local_null / n_primary),
        }

    bins = []
    for index in range(counts.shape[1]):
        local_count = counts[:, index]
        if local_count.sum() == 0:
            continue
        pair_label = ratio_by_case(label_sum[:, index], local_count)
        pair_prediction = ratio_by_case(prediction_sum[:, index], local_count)
        pair_residual = ratio_by_case(residual_sum[:, index], local_count)
        additive_residual = residual_sum[:, index] / n_primary
        bins.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "secondary_size_mean": float(
                size_sum[:, index].sum() / local_count.sum()
            ),
            "n_pairs": int(local_count.sum()),
            "n_cases_with_pairs": int((local_count > 0).sum()),
            "pair_fraction": float(local_count.sum() / counts.sum()),
            "pair_label": finite_stat(pair_label, test_zero=False),
            "pair_prediction": finite_stat(pair_prediction, test_zero=False),
            "pair_label_minus_prediction": finite_stat(pair_residual),
            "additive_label_minus_prediction_per_primary": finite_stat(
                additive_residual
            ),
        })

    all_group = group_summary(np.arange(counts.shape[1]), "all")
    below = group_summary(np.arange(cut_index), f"Re_s < {cut:g}")
    above = group_summary(
        np.arange(cut_index, counts.shape[1]), f"Re_s >= {cut:g}"
    )
    all_residual = all_group[
        "additive_label_minus_prediction_per_primary"
    ]["mean"]
    for item in (below, above):
        contribution = item[
            "additive_label_minus_prediction_per_primary"
        ]["mean"]
        item["share_of_global_additive_residual"] = (
            float(contribution / all_residual) if all_residual != 0 else None
        )

    per_case_total = residual_sum.sum(axis=1) / n_primary
    per_case_parts = residual_sum[:, :cut_index].sum(axis=1) / n_primary
    per_case_parts += residual_sum[:, cut_index:].sum(axis=1) / n_primary
    if not np.allclose(per_case_total, per_case_parts, rtol=0, atol=2e-15):
        raise RuntimeError("secondary-size residual groups do not add to total")
    return {
        "size_edges": edges.tolist(),
        "cut": float(cut),
        "bins": bins,
        "groups": {"all": all_group, "below_cut": below, "at_or_above_cut": above},
        "maximum_additive_closure_error": float(
            np.max(np.abs(per_case_total - per_case_parts))
        ),
    }


def score_pairs(predictor: Any, frame: pd.DataFrame, chunk: int) -> np.ndarray:
    result = np.empty(len(frame), dtype=np.float64)
    for lo in range(0, len(frame), chunk):
        hi = min(lo + chunk, len(frame))
        result[lo:hi] = predictor.predict_on_pairs(
            frame.iloc[lo:hi][PAIR_FEATURES], task="response"
        )["response"].to_numpy(float)
    if not np.isfinite(result).all():
        raise RuntimeError("V2.2 produced a non-finite pair prediction")
    return result


def stream_half_shear(
    catalogue: str,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    cut: float,
    score_chunk: int,
) -> tuple[dict[str, Any], list[list[float]]]:
    """Stream the large response catalogue and return a size profile."""
    if BLENDEMU_ROOT not in sys.path:
        sys.path.insert(0, BLENDEMU_ROOT)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=tag, conditions=COND, device="cpu"
    )
    raw_cuts, _, _ = predictor._select("regression")
    if len(raw_cuts) != len(CUT_NAMES):
        raise RuntimeError(f"unexpected regression cuts: {raw_cuts}")
    cuts = dict(zip(CUT_NAMES, raw_cuts))
    n_case = case_max - case_min + 1
    n_bin = len(SIZE_EDGES) - 1
    counts = np.zeros((n_case, n_bin), dtype=np.int64)
    label_sum = np.zeros((n_case, n_bin), dtype=np.float64)
    prediction_sum = np.zeros((n_case, n_bin), dtype=np.float64)
    null_sum = np.zeros((n_case, n_bin), dtype=np.float64)
    size_sum = np.zeros((n_case, n_bin), dtype=np.float64)
    primaries = [set() for _ in range(n_case)]
    previous_min = -1
    processed_batches = 0
    scanned_rows = 0

    with ipc.open_file(catalogue) as reader:
        missing = set(PAIR_COLUMNS) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                PAIR_COLUMNS
            ).to_pandas()
            if frame.empty:
                continue
            batch_min = int(frame.case.min())
            if batch_min < previous_min:
                raise RuntimeError("response catalogue cases are not ordered")
            previous_min = batch_min
            scanned_rows += len(frame)
            frame = frame.loc[frame.case.between(case_min, case_max)]
            if frame.empty:
                if batch_min > case_max:
                    break
                continue
            processed_batches += 1
            finite_label = np.isfinite(
                frame[["delta_et1", "delta_et2"]].to_numpy(float)
            ).all(axis=1)
            frame = frame.loc[finite_label]
            primary_ok = (
                frame.r_input_p.between(*cuts["r_input_p"], inclusive="neither")
                & frame.Re_input_p.between(
                    *cuts["Re_input_p"], inclusive="neither"
                )
            )
            primary_ok &= in_domain(
                frame.r_input_p.to_numpy(float),
                frame.Re_input_p.to_numpy(float),
            )
            eligible = frame.loc[primary_ok, ["case", "input_index"]]
            for case, local in eligible.groupby("case", sort=False):
                primaries[int(case) - case_min].update(
                    local.input_index.to_numpy(np.int64).tolist()
                )

            pair_ok = primary_ok.copy()
            for column in ("r_input_s", "Re_input_s", "distance"):
                pair_ok &= frame[column].between(
                    *cuts[column], inclusive="neither"
                )
            pairs = frame.loc[pair_ok].copy()
            if pairs.empty:
                continue
            if not np.isfinite(pairs[PAIR_FEATURES].to_numpy(float)).all():
                raise RuntimeError("supported pair has a non-finite model feature")
            prediction = score_pairs(predictor, pairs, score_chunk)
            label = pairs.delta_et1.to_numpy(float) / shear
            null = pairs.delta_et2.to_numpy(float) / shear
            size = pairs.Re_input_s.to_numpy(float)
            bins = np.searchsorted(SIZE_EDGES, size, side="right") - 1
            if np.any((bins < 0) | (bins >= n_bin)):
                bad = size[(bins < 0) | (bins >= n_bin)][:5]
                raise RuntimeError(f"supported secondary size outside plot edges: {bad}")
            case_index = pairs.case.to_numpy(np.int64) - case_min
            flat = case_index * n_bin + bins
            minlength = n_case * n_bin
            counts += np.bincount(flat, minlength=minlength).reshape(n_case, n_bin)
            label_sum += np.bincount(
                flat, weights=label, minlength=minlength
            ).reshape(n_case, n_bin)
            prediction_sum += np.bincount(
                flat, weights=prediction, minlength=minlength
            ).reshape(n_case, n_bin)
            null_sum += np.bincount(
                flat, weights=null, minlength=minlength
            ).reshape(n_case, n_bin)
            size_sum += np.bincount(
                flat, weights=size, minlength=minlength
            ).reshape(n_case, n_bin)
            if processed_batches % 100 == 0:
                print(
                    f"processed {processed_batches} selected batches; "
                    f"active pairs={counts.sum():,}",
                    flush=True,
                )

    n_primary = np.asarray([len(item) for item in primaries], dtype=np.int64)
    if np.any(n_primary == 0):
        missing_cases = (np.flatnonzero(n_primary == 0) + case_min).tolist()
        raise RuntimeError(f"cases without eligible primaries: {missing_cases[:10]}")
    profile = summarize_half_shear_accumulators(
        counts,
        label_sum,
        prediction_sum,
        null_sum,
        size_sum,
        n_primary,
        SIZE_EDGES,
        cut,
    )
    profile.update({
        "tag": tag,
        "catalogue": os.path.abspath(catalogue),
        "case_window": [int(case_min), int(case_max)],
        "shear": float(shear),
        "v21_primary_domain": True,
        "regression_cuts": [[float(x) for x in pair] for pair in raw_cuts],
        "n_cases": int(n_case),
        "n_primaries": int(n_primary.sum()),
        "n_pairs": int(counts.sum()),
        "processed_batches": int(processed_batches),
        "scanned_rows_through_last_batch": int(scanned_rows),
        "uncertainty_unit": "rendered simulation case",
        "evaluation_status": "in-sample audit on V2.2 half-shear training cases",
    })
    return profile, profile["regression_cuts"]


def validate_reference_closure(profile: dict[str, Any], path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)
    summary = reference["summary"]
    observed = profile["groups"]["all"]
    checks = {
        "case_window": profile["case_window"] == reference["case_window"],
        "n_primaries": profile["n_primaries"] == summary["n_primaries"],
        "n_pairs": profile["n_pairs"] == summary["n_pairs"],
        "label_sum_mean": np.isclose(
            observed["additive_label_per_primary"]["mean"],
            summary["label_sum_mean"], rtol=0, atol=2e-10,
        ),
        "prediction_sum_mean": np.isclose(
            observed["additive_prediction_per_primary"]["mean"],
            summary["prediction_sum_mean"], rtol=0, atol=2e-8,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training profile fails reference closure: {checks}")
    return {
        "path": os.path.abspath(path),
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def anchor_removed_mask(frame: pd.DataFrame, cut: float) -> np.ndarray:
    paired = frame.has_deployed_pair.to_numpy(bool)
    size = np.power(
        10.0, frame.log10_dominant_secondary_size.to_numpy(float)
    )
    if not np.isfinite(size[paired]).all():
        raise RuntimeError("deployed anchor lacks dominant-secondary size")
    return paired & (size < cut)


def subset_case_table(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    local = frame.loc[mask, ["case", "R_blend_truth", "scene_prediction"]].copy()
    if local.empty:
        raise RuntimeError("anchor subset is empty")
    local["residual"] = local.R_blend_truth - local.scene_prediction
    result = local.groupby("case", sort=True).agg(
        n=("residual", "size"),
        truth=("R_blend_truth", "mean"),
        prediction=("scene_prediction", "mean"),
        residual=("residual", "mean"),
        residual_sum=("residual", "sum"),
    )
    return result


def summarize_anchor_cut(frame: pd.DataFrame, cut: float) -> dict[str, Any]:
    """Summarize a dominant-secondary cut, blocking uncertainty by case."""
    if frame.case.nunique() < 2:
        raise ValueError("need at least two coherent-anchor cases")
    removed_mask = anchor_removed_mask(frame, cut)
    kept_mask = ~removed_mask
    all_table = subset_case_table(frame, np.ones(len(frame), dtype=bool))
    removed_table = subset_case_table(frame, removed_mask).reindex(all_table.index)
    kept_table = subset_case_table(frame, kept_mask).reindex(all_table.index)
    if removed_table.isna().any().any() or kept_table.isna().any().any():
        raise RuntimeError("a coherent case is empty on one side of the size cut")

    def population(table: pd.DataFrame, name: str) -> dict[str, Any]:
        residual = finite_stat(table.residual.to_numpy(float))
        truth = finite_stat(table.truth.to_numpy(float), test_zero=False)
        prediction = finite_stat(
            table.prediction.to_numpy(float), test_zero=False
        )
        return {
            "name": name,
            "n_rows": int(table.n.sum()),
            "n_cases": int(len(table)),
            "truth": truth,
            "prediction": prediction,
            "truth_minus_prediction": residual,
            "prediction_over_truth_minus_one": float(
                prediction["mean"] / truth["mean"] - 1.0
            ),
        }

    all_population = population(all_table, "all anchors")
    removed_population = population(
        removed_table, f"removed: dominant Re_s < {cut:g}"
    )
    kept_population = population(
        kept_table, f"kept: no pair or dominant Re_s >= {cut:g}"
    )
    removed_contribution = removed_table.residual_sum / all_table.n
    kept_contribution = kept_table.residual_sum / all_table.n
    all_residual = all_table.residual
    closure = removed_contribution + kept_contribution - all_residual
    if not np.allclose(closure, 0.0, rtol=0, atol=2e-15):
        raise RuntimeError("coherent-anchor cut does not add to the full gap")
    change = kept_table.residual - all_table.residual
    original_gap = all_population["truth_minus_prediction"]["mean"]
    kept_gap = kept_population["truth_minus_prediction"]["mean"]
    return {
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "cut": float(cut),
        "n_no_deployed_pair": int((~frame.has_deployed_pair.to_numpy(bool)).sum()),
        "removed_fraction_pooled": float(removed_mask.mean()),
        "removed_fraction_by_case": finite_stat(
            (removed_table.n / all_table.n).to_numpy(float), test_zero=False
        ),
        "populations": {
            "all": all_population,
            "removed": removed_population,
            "kept": kept_population,
        },
        "additive_gap_decomposition": {
            "removed_contribution": finite_stat(
                removed_contribution.to_numpy(float)
            ),
            "kept_contribution": finite_stat(kept_contribution.to_numpy(float)),
            "maximum_case_closure_error": float(np.max(np.abs(closure))),
            "removed_share_of_original_gap": (
                float(removed_contribution.mean() / original_gap)
                if original_gap != 0 else None
            ),
        },
        "kept_minus_original_gap": finite_stat(change.to_numpy(float)),
        "fraction_of_original_gap_remaining_after_cut": (
            float(kept_gap / original_gap) if original_gap != 0 else None
        ),
        "fraction_of_original_gap_reduced_after_cut": (
            float(1.0 - kept_gap / original_gap) if original_gap != 0 else None
        ),
    }


def analyze_anchor_table(path: str, cut: float) -> dict[str, Any]:
    columns = [
        "case",
        "input_index",
        "R_blend_truth",
        "scene_prediction",
        "has_deployed_pair",
        "log10_dominant_secondary_size",
    ]
    frame = pd.read_feather(path, columns=columns)
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate coherent-anchor key")
    if not np.isfinite(
        frame[["R_blend_truth", "scene_prediction"]].to_numpy(float)
    ).all():
        raise RuntimeError("non-finite coherent-anchor response")
    observed = set(frame.case.unique().tolist())
    expected = set(range(400, 900))
    if observed != expected:
        raise RuntimeError("coherent-anchor table does not cover c400--899 exactly")
    splits = {}
    for name, (lo, hi) in ANCHOR_SPLITS.items():
        local = frame.loc[frame.case.between(lo, hi)].copy()
        item = summarize_anchor_cut(local, cut)
        item["case_window"] = [lo, hi]
        splits[name] = item
    return {
        "feature_table": os.path.abspath(path),
        "cut_definition": (
            f"remove has_deployed_pair AND dominant_secondary_size < {cut:g} arcsec; "
            "keep no-pair anchors"
        ),
        "cut": float(cut),
        "target": "R_blend_truth - R_blend_lsst_r_extnbr_v22",
        "positive_target_meaning": "V2.2 underprediction",
        "uncertainty_unit": "rendered simulation case",
        "post_hoc": True,
        "post_hoc_reason": (
            "the 0.4 arcsec threshold was proposed after inspecting the opened "
            "final-test secondary-size curve"
        ),
        "splits": splits,
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_training_profile(profile: dict[str, Any], stem: str) -> None:
    configure_style()
    bins = profile["bins"]
    x = np.asarray([item["secondary_size_mean"] for item in bins])
    label = np.asarray([item["pair_label"]["mean"] for item in bins])
    label_sem = np.asarray([item["pair_label"]["case_sem"] for item in bins])
    prediction = np.asarray([item["pair_prediction"]["mean"] for item in bins])
    prediction_sem = np.asarray([
        item["pair_prediction"]["case_sem"] for item in bins
    ])
    pair_residual = np.asarray([
        item["pair_label_minus_prediction"]["mean"] for item in bins
    ])
    pair_residual_sem = np.asarray([
        item["pair_label_minus_prediction"]["case_sem"] for item in bins
    ])
    additive = np.asarray([
        item["additive_label_minus_prediction_per_primary"]["mean"]
        for item in bins
    ])
    additive_sem = np.asarray([
        item["additive_label_minus_prediction_per_primary"]["case_sem"]
        for item in bins
    ])

    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.15), constrained_layout=True
    )
    axes[0].errorbar(
        x, label, yerr=label_sem, color="#000000", marker="o",
        linewidth=1.1, markersize=3.8, capsize=2.0, label="Half-shear label",
    )
    axes[0].errorbar(
        x, prediction, yerr=prediction_sem, color="#0072B2", marker="s",
        linestyle="--", linewidth=1.1, markersize=3.5, capsize=2.0,
        label="V2.2 emulator",
    )
    axes[0].set_ylabel(r"Mean pair $R_{\rm blend}$")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        x, pair_residual, yerr=pair_residual_sem, color="#D55E00",
        marker="o", linewidth=1.1, markersize=3.8, capsize=2.0,
    )
    axes[1].set_ylabel(r"Pair label $-$ V2.2")

    axes[2].errorbar(
        x, additive, yerr=additive_sem, color="#009E73", marker="o",
        linewidth=1.1, markersize=3.8, capsize=2.0,
    )
    axes[2].set_ylabel(
        "Additive label - V2.2\ncontribution per primary"
    )

    titles = (
        "Conditional response",
        "Conditional pair residual",
        "Population-weighted contribution",
    )
    for panel, (axis, title) in enumerate(zip(axes, titles)):
        axis.axhline(0.0, color="0.60", linestyle=":", linewidth=0.7)
        axis.axvline(
            profile["cut"], color="#CC79A7", linestyle="--", linewidth=1.0,
            label=r"$0.4''$ cut" if panel == 2 else None,
        )
        axis.set_xscale("log")
        axis.set_xlabel(r"Secondary input $R_e$ (arcsec; log scale)")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.15, 1.06, chr(ord("A") + panel), transform=axis.transAxes,
            fontweight="bold", fontsize=10, va="top",
        )
    axes[2].legend(frameon=False)
    fig.suptitle(
        "V2.2 on its half-shear training population (cases 40-199)\n"
        "Error bars are SEM across rendered cases",
        fontsize=9.5,
    )
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def flatten_training_csv(profile: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in profile["bins"]:
        row = {
            key: item[key]
            for key in (
                "bin", "lower", "upper", "secondary_size_mean", "n_pairs",
                "n_cases_with_pairs", "pair_fraction",
            )
        }
        for name in (
            "pair_label",
            "pair_prediction",
            "pair_label_minus_prediction",
            "additive_label_minus_prediction_per_primary",
        ):
            row[f"{name}_mean"] = item[name]["mean"]
            row[f"{name}_case_sem"] = item[name]["case_sem"]
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_anchor_csv(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, item in payload["splits"].items():
        for population, result in item["populations"].items():
            rows.append({
                "split": split,
                "case_min": item["case_window"][0],
                "case_max": item["case_window"][1],
                "population": population,
                "n_rows": result["n_rows"],
                "removed_fraction_pooled": item["removed_fraction_pooled"],
                "truth_mean": result["truth"]["mean"],
                "prediction_mean": result["prediction"]["mean"],
                "truth_minus_prediction_mean": result[
                    "truth_minus_prediction"
                ]["mean"],
                "truth_minus_prediction_case_sem": result[
                    "truth_minus_prediction"
                ]["case_sem"],
                "truth_minus_prediction_t": result[
                    "truth_minus_prediction"
                ]["t"],
                "truth_minus_prediction_p": result[
                    "truth_minus_prediction"
                ]["p"],
                "fraction_original_gap_remaining": item[
                    "fraction_of_original_gap_remaining_after_cut"
                ],
            })
    return pd.DataFrame(rows)


def training_markdown(payload: dict[str, Any]) -> str:
    all_item = payload["groups"]["all"]
    lines = [
        "# V2.2 half-shear training response versus secondary size",
        "",
        "This is an in-sample audit of V2.2 on half-shear training cases "
        "40--199. The label is `delta_et1 / 0.2`; no coherent-anchor or "
        "constgold measurement enters this plot. Error bars use rendered case "
        "as the uncertainty unit.",
        "",
        f"- Eligible primaries: `{payload['n_primaries']:,}`.",
        f"- Active supported pairs: `{payload['n_pairs']:,}`.",
        f"- Global label sum per primary: "
        f"`{all_item['additive_label_per_primary']['mean']:+.8f} +- "
        f"{all_item['additive_label_per_primary']['case_sem']:.8f}`.",
        f"- Global V2.2 sum per primary: "
        f"`{all_item['additive_prediction_per_primary']['mean']:+.8f} +- "
        f"{all_item['additive_prediction_per_primary']['case_sem']:.8f}`.",
        f"- Global label minus V2.2: "
        f"`{all_item['additive_label_minus_prediction_per_primary']['mean']:+.8f} "
        f"+- {all_item['additive_label_minus_prediction_per_primary']['case_sem']:.8f}`.",
        "",
        "## Split at 0.4 arcsec",
        "",
        "| secondary size | pair fraction | label sum / primary | V2.2 sum / primary | label - V2.2 / primary | case SEM | share of global residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("below_cut", "at_or_above_cut"):
        item = payload["groups"][key]
        lines.append(
            f"| {item['name']} | {item['pair_fraction']:.3%} | "
            f"{item['additive_label_per_primary']['mean']:+.6f} | "
            f"{item['additive_prediction_per_primary']['mean']:+.6f} | "
            f"{item['additive_label_minus_prediction_per_primary']['mean']:+.6f} | "
            f"{item['additive_label_minus_prediction_per_primary']['case_sem']:.6f} | "
            f"{item['share_of_global_additive_residual']:.1%} |"
        )
    lines.extend([
        "",
        "The per-pair panel answers conditional calibration. The additive panel "
        "also includes how frequently each size interval occurs, so its bins sum "
        "to the global training residual.",
        "",
    ])
    return "\n".join(lines)


def anchor_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Coherent-anchor dominant-secondary size cut",
        "",
        "Residual is `R_blend_truth - V2.2`; positive means underprediction. "
        "The cut removes anchors whose response-dominant deployed neighbour has "
        "input `Re < 0.4 arcsec`; the rare no-pair anchor is retained. Case is "
        "the uncertainty unit.",
        "",
        "This is **post-hoc**: 0.4 arcsec was proposed after inspection of the "
        "opened final-test curve, so final-test p-values are descriptive rather "
        "than a fresh confirmation.",
        "",
        "| split | rows removed | original gap | gap after cut | case SEM after cut | remaining gap | removed additive share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "tune", "final_test", "all"):
        item = payload["splits"][name]
        original = item["populations"]["all"]["truth_minus_prediction"]
        kept = item["populations"]["kept"]["truth_minus_prediction"]
        share = item["additive_gap_decomposition"][
            "removed_share_of_original_gap"
        ]
        lines.append(
            f"| {name} ({item['case_window'][0]}-{item['case_window'][1]}) | "
            f"{item['removed_fraction_pooled']:.2%} | {original['mean']:+.6f} | "
            f"{kept['mean']:+.6f} | {kept['case_sem']:.6f} | "
            f"{item['fraction_of_original_gap_remaining_after_cut']:.1%} | "
            f"{share:.1%} |"
        )
    lines.extend(["", "All additive decompositions close case by case.", ""])
    return "\n".join(lines)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def refuse_outputs(paths: list[str]) -> None:
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--anchor-features", required=True)
    ap.add_argument("--reference-closure", required=True)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=199)
    ap.add_argument("--shear", type=float, default=0.2)
    ap.add_argument("--size-cut", type=float, default=0.4)
    ap.add_argument("--score-chunk", type=int, default=500_000)
    ap.add_argument("--training-output-prefix", required=True)
    ap.add_argument("--anchor-output-prefix", required=True)
    args = ap.parse_args()
    if args.case_min > args.case_max or args.shear <= 0 or args.size_cut <= 0:
        raise ValueError("invalid case window, shear, or size cut")

    training_outputs = [
        f"{args.training_output_prefix}.{suffix}"
        for suffix in ("json", "md", "csv", "png", "pdf")
    ]
    anchor_outputs = [
        f"{args.anchor_output_prefix}.{suffix}"
        for suffix in ("json", "md", "csv")
    ]
    refuse_outputs([*training_outputs, *anchor_outputs])
    for prefix in (args.training_output_prefix, args.anchor_output_prefix):
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    training, _ = stream_half_shear(
        args.catalogue,
        args.tag,
        args.case_min,
        args.case_max,
        args.shear,
        args.size_cut,
        args.score_chunk,
    )
    training["reference_closure"] = validate_reference_closure(
        training, args.reference_closure
    )
    anchor = analyze_anchor_table(args.anchor_features, args.size_cut)

    training = json_clean(training)
    anchor = json_clean(anchor)
    with open(f"{args.training_output_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(training, handle, indent=2, allow_nan=False)
    with open(f"{args.anchor_output_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(anchor, handle, indent=2, allow_nan=False)
    Path(f"{args.training_output_prefix}.md").write_text(
        training_markdown(training), encoding="utf-8"
    )
    Path(f"{args.anchor_output_prefix}.md").write_text(
        anchor_markdown(anchor), encoding="utf-8"
    )
    flatten_training_csv(training).to_csv(
        f"{args.training_output_prefix}.csv", index=False
    )
    flatten_anchor_csv(anchor).to_csv(
        f"{args.anchor_output_prefix}.csv", index=False
    )
    plot_training_profile(training, args.training_output_prefix)
    print(json.dumps({
        "training_global": training["groups"]["all"],
        "training_size_groups": {
            key: training["groups"][key]
            for key in ("below_cut", "at_or_above_cut")
        },
        "anchor_final_test": anchor["splits"]["final_test"],
    }, indent=2, allow_nan=False), flush=True)
    print("V22_SECONDARY_SIZE_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
