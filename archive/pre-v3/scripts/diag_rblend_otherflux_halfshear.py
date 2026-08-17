"""Held-out pair response versus third-or-later intrinsic scene flux.

For every primary--secondary row, remove that designated secondary from the
primary's full-input-scene flux in the 0--1, 1--3 or 3--10 arcsec shell that
contains it. The resulting pair-specific coordinates include all remaining
rendered galaxies and no measured/detected quantity. This is the scene context
that Zhang et al. Appendix A predicts will suppress a secondary's marginal
R_blend.

The diagnostic uses held-out half-shear cases 0--39, the exact V2.2 regression
support, the independent ap7 ruler truth/null, and the baseline V2.2 prediction.
No constgold quantity or empirical adjustment is read. Error bars are
delete-one-case jackknife standard errors. Curves are descriptive marginal
profiles; the prediction curve exposes pair-population correlations directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
if "/home/z/Zekang.Zhang/blendemu" not in sys.path:
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")

from blendemu.scene_features import PAIR_OTHER_FLUX_COLUMNS, pair_other_flux_ratios  # noqa: E402
from scripts.diag_rblend_scene3_halfshear import (  # noqa: E402
    SCENE_MODEL,
    attach_scene,
    binned_table,
    jackknife_mean,
    read_pairset,
    read_scene_lookup,
    score,
)


SHELL_LABELS = (
    r"Other-galaxy flux, 0--1 arcsec",
    r"Other-galaxy flux, 1--3 arcsec",
    r"Other-galaxy flux, 3--10 arcsec",
)
ABSOLUTE_OTHER_FLUX_COLUMNS = (
    "scene_logfluxabs_other_near_0_1",
    "scene_logfluxabs_other_mid_1_3",
    "scene_logfluxabs_other_far_3_10",
)


def zero_plus_positive_quartile_edges(x: np.ndarray, min_zero: int = 200) -> np.ndarray:
    """Use a zero class only when it is large enough to estimate a mean."""
    x = np.asarray(x, float)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("other-flux coordinate must be finite and non-negative")
    positive = x[x > 0]
    if not len(positive):
        raise RuntimeError("other-flux coordinate has no positive rows")
    q = np.unique(np.quantile(positive, [0.25, 0.50, 0.75]))
    if np.count_nonzero(x == 0) >= min_zero:
        return np.r_[-np.inf, np.nextafter(0.0, np.inf), q, np.inf]
    # Rare exact zeros join the lowest positive-flux quartile.
    qall = np.unique(np.quantile(x, [0.25, 0.50, 0.75]))
    return np.r_[-np.inf, qall, np.inf]


def contrast_jackknife(
    values: np.ndarray,
    cases: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[float, float]:
    """High-minus-low mean with delete-one-case jackknife error."""
    good = np.isfinite(values)
    low = low & good
    high = high & good
    if not low.any() or not high.any():
        return np.nan, np.nan
    low_sum = float(values[low].sum(dtype=np.float64))
    high_sum = float(values[high].sum(dtype=np.float64))
    low_n, high_n = int(low.sum()), int(high.sum())
    estimate = high_sum / high_n - low_sum / low_n

    # Algebraically identical to deleting each case mask in turn, but O(N)
    # rather than O(N * n_case): aggregate each group's sums/counts once.
    unique = np.unique(cases[low | high])
    case_index = np.searchsorted(unique, cases)
    low_case_sum = np.bincount(
        case_index[low], weights=values[low], minlength=len(unique),
    )
    high_case_sum = np.bincount(
        case_index[high], weights=values[high], minlength=len(unique),
    )
    low_case_n = np.bincount(case_index[low], minlength=len(unique))
    high_case_n = np.bincount(case_index[high], minlength=len(unique))
    valid = (low_n > low_case_n) & (high_n > high_case_n)
    loo = ((high_sum - high_case_sum[valid]) / (high_n - high_case_n[valid])
           - (low_sum - low_case_sum[valid]) / (low_n - low_case_n[valid]))
    center = loo.mean()
    sem = np.sqrt((len(loo) - 1.0) / len(loo) * np.square(loo - center).sum())
    return estimate, float(sem)


def _labels(table: pd.DataFrame) -> list[str]:
    labels = []
    for lo, hi, xm in zip(table.lo, table.hi, table.x_mean):
        labels.append("zero" if lo < 0 and hi <= np.nextafter(0.0, np.inf) else f"{xm:.2f}")
    return labels


def flux_axis_expression(flux_mode: str) -> str:
    if flux_mode == "absolute":
        return r"$\log_{10}(1+F_{\rm other}\ [{\rm counts}])$"
    return r"$\log_{10}(1+F_{\rm other}/F_p)$"


def profile_panel(ax, table: pd.DataFrame, xlabel: str, flux_mode: str) -> None:
    table = table[table.n >= 200].copy()
    x = np.arange(len(table))
    for name, color, marker, linestyle in (
        ("truth", "#000000", "o", "-"),
        ("v22", "#0072B2", "s", "--"),
    ):
        ax.errorbar(
            x, table[f"{name}_mean"], yerr=table[f"{name}_sem"],
            color=color, marker=marker, linestyle=linestyle, lw=1.2, ms=4,
            capsize=2, label={"truth": "half-shear truth", "v22": "V2.2"}[name],
        )
    ax.set_xticks(x, _labels(table))
    ax.set_xlabel(xlabel + "\n" + flux_axis_expression(flux_mode))
    ax.set_ylabel(r"Mean pair $R_{\rm blend}$")
    ax.spines[["top", "right"]].set_visible(False)


def residual_panel(ax, table: pd.DataFrame, xlabel: str, flux_mode: str) -> None:
    table = table[table.n >= 200].copy()
    x = np.arange(len(table))
    ax.errorbar(
        x, table["v22_minus_truth_mean"], yerr=table["v22_minus_truth_sem"],
        color="#D55E00", marker="o", lw=1.2, ms=4, capsize=2,
        label="V2.2 - truth",
    )
    ax.errorbar(
        x, table["null_mean"], yerr=table["null_sem"], color="0.45",
        marker="x", linestyle="--", lw=1.0, ms=4, capsize=2,
        label=r"45$^\circ$ null",
    )
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xticks(x, _labels(table))
    ax.set_xlabel(xlabel + "\n" + flux_axis_expression(flux_mode))
    ax.set_ylabel(r"Pair-response residual")
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(
    tables: pd.DataFrame,
    output_prefix: str,
    flux_mode: str = "relative",
    axis_prefix: str = "other_shell",
    case_min: int = 0,
    case_max: int = 40,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8), constrained_layout=True)
    for i, label in enumerate(SHELL_LABELS):
        table = tables[tables.axis == f"{axis_prefix}_{i}"].copy()
        profile_panel(axes[0, i], table, label, flux_mode)
        residual_panel(axes[1, i], table, label, flux_mode)
    for letter, ax in zip("ABCDEF", axes.flat):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontweight="bold", va="top")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.suptitle(
        f"Half-shear pair response versus {flux_mode} other-galaxy scene flux\n"
        f"Designated secondary excluded; cases {case_min}--{case_max - 1}; "
        "error bars: case jackknife SEM",
        fontsize=9,
    )
    fig.savefig(output_prefix + ".png", dpi=300)
    fig.savefig(output_prefix + ".pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairset", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
        "blendflow/blend_pairset_ap7.feather",
    )
    ap.add_argument("--scene-lookup", action="append", required=True)
    ap.add_argument("--case-min", type=int, default=0)
    ap.add_argument("--case-max", type=int, default=40)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--score-chunk", type=int, default=500_000)
    ap.add_argument("--flux-mode", choices=("relative", "absolute"), default="relative")
    ap.add_argument("--zero-point", type=float, default=30.0)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    pair = attach_scene(
        read_pairset(args.pairset, args.case_min, args.case_max),
        read_scene_lookup(args.scene_lookup, args.case_min, args.case_max),
    )
    other = pair_other_flux_ratios(
        pair[SCENE_MODEL].to_numpy(float),
        pair["r_input_p"].to_numpy(float),
        pair["r_input_s"].to_numpy(float),
        pair["distance"].to_numpy(float),
    )
    if args.flux_mode == "absolute":
        primary_flux = np.exp(
            -0.4 * np.log(10.0) * (pair["r_input_p"].to_numpy(float) - args.zero_point)
        )
        other_ratio = np.expm1(other * np.log(10.0))
        coordinates = np.log10(1.0 + other_ratio * primary_flux[:, None])
        feature_names = ABSOLUTE_OTHER_FLUX_COLUMNS
        axis_prefix = "absolute_other_shell"
        feature_definition = (
            "log10(1 + intrinsic absolute flux in simulation counts of every rendered input "
            "galaxy except the primary and this response row's designated secondary); "
            f"zero point={args.zero_point:g}; no primary-flux normalization"
        )
    else:
        coordinates = other
        feature_names = PAIR_OTHER_FLUX_COLUMNS
        axis_prefix = "other_shell"
        feature_definition = (
            "intrinsic full-input-scene shell flux relative to primary, excluding primary "
            "and the response row's designated secondary"
        )
    for i, name in enumerate(feature_names):
        pair[name] = coordinates[:, i].astype(np.float32)
    pair["v22"] = score(pair, args.tag, args.score_chunk)

    truth = pair["blend_truth"].to_numpy(float)
    null = pair["blend_null"].to_numpy(float)
    pred = pair["v22"].to_numpy(float)
    series = {
        "truth": truth,
        "null": null,
        "v22": pred,
        "v22_minus_truth": pred - truth,
    }
    cases = pair["case"].to_numpy(np.int32)

    tables = []
    summary = {
        "case_window": [args.case_min, args.case_max],
        "n_pairs": int(len(pair)),
        "flux_mode": args.flux_mode,
        "zero_point": args.zero_point,
        "feature_definition": feature_definition,
        "shells_arcsec": [[0, 1], [1, 3], [3, 10]],
        "axes": {},
    }
    for i, name in enumerate(feature_names):
        x = pair[name].to_numpy(float)
        edges = zero_plus_positive_quartile_edges(x)
        table = binned_table(f"{axis_prefix}_{i}", x, edges, series, cases, "pair")
        tables.append(table)
        ids = np.digitize(x, edges[1:-1], right=False)
        low = ids == 0
        high = ids == len(edges) - 2
        contrasts = {}
        for label, values in series.items():
            estimate, sem = contrast_jackknife(values, cases, low, high)
            contrasts[label] = {"high_minus_low": estimate, "case_jackknife_sem": sem}
        summary["axes"][name] = {
            "edges": ["-inf" if np.isneginf(v) else "inf" if np.isposinf(v) else float(v)
                      for v in edges],
            "zero_fraction": float(np.mean(x == 0)),
            "contrasts": contrasts,
        }

    # Stored ap7 distance is input-to-input and was independently checked
    # against coordinates to 4.548e-4 arcsec. Count the only rows whose shell
    # identity could conceivably be sensitive to that maximum discrepancy.
    distance = pair["distance"].to_numpy(float)
    summary["shell_boundary_audit"] = {
        "coordinate_vs_stored_max_arcsec": 4.548e-4,
        "within_that_distance_of_1arcsec": int((np.abs(distance - 1.0) <= 4.548e-4).sum()),
        "within_that_distance_of_3arcsec": int((np.abs(distance - 3.0) <= 4.548e-4).sum()),
    }
    overall = {}
    all_rows = np.ones(len(pair), dtype=bool)
    for label, values in series.items():
        mean, sem, n = jackknife_mean(values, cases, all_rows)
        overall[label] = {"mean": mean, "case_jackknife_sem": sem, "n": n}
    summary["overall"] = overall

    output = pd.concat(tables, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    output.to_csv(args.output_prefix + ".csv", index=False)
    with open(args.output_prefix + ".json", "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    make_figure(
        output, args.output_prefix, args.flux_mode, axis_prefix,
        args.case_min, args.case_max,
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print("DIAG_RBLEND_OTHERFLUX_HALFSHEAR_DONE", flush=True)


if __name__ == "__main__":
    main()
