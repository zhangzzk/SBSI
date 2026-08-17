"""Plot held-out 1D bias curves inside panel E's high-response tail.

The subset is frozen by the lower edge of the rightmost ``scene_prediction``
bin in the already-produced all-anchor curve.  Development rows inside that
same subset define fresh quantile edges for every plotted coordinate; those
edges are then applied to final-test rows.  The globally trained full bias
emulator is only evaluated here--it is not refit on the tail.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from string import ascii_uppercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats

from scripts.train_anchor_bias_emulator import (
    KEY,
    TARGET,
    conditional_curve,
    configure_style,
    finite_stat,
    quantile_edges,
    save_figure,
)


POPULATION = "scene_prediction_right_tail"


def frozen_tail_definition(curves: pd.DataFrame) -> dict:
    """Return the frozen lower edge and audit row for panel E's last bin."""
    required = {"population", "feature", "bin", "lower", "upper", "n_rows"}
    if missing := required - set(curves):
        raise KeyError(f"parent curves lack {sorted(missing)}")
    local = curves.loc[
        (curves.population == "all") & (curves.feature == "scene_prediction")
    ].sort_values("bin")
    if local.empty:
        raise RuntimeError("parent curves have no all-anchor scene_prediction panel")
    row = local.iloc[-1]
    if not np.isfinite(row.lower) or not np.isposinf(row.upper):
        raise RuntimeError("last scene_prediction bin is not a finite-threshold tail")
    return {
        "feature": "scene_prediction",
        "operator": ">=",
        "threshold": float(row.lower),
        "parent_bin": int(row.bin),
        "parent_test_rows": int(row.n_rows),
        "parent_curve_bins": int(local.bin.max() + 1),
    }


def tail_mask(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    values = frame.scene_prediction.to_numpy(float)
    return np.isfinite(values) & (values >= threshold)


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return finite_stat(values, hypothesis_test=False)


def extreme_contrast(development: pd.DataFrame, test: pd.DataFrame,
                     prediction: np.ndarray, feature: str,
                     n_bins: int) -> dict | None:
    """Case-paired highest-minus-lowest occupied feature-bin contrast."""
    dev = development.loc[np.isfinite(development[feature])]
    finite = np.isfinite(test[feature].to_numpy(float))
    local = test.loc[finite, ["case", TARGET, feature]].copy()
    local["prediction"] = np.asarray(prediction, float)[finite]
    if dev[feature].nunique() < 2 or local.empty:
        return None
    edges = quantile_edges(dev[feature].to_numpy(float), n_bins)
    local["bin"] = np.digitize(local[feature].to_numpy(float), edges[1:-1])
    occupied = sorted(
        index for index, count in local.groupby("bin").case.nunique().items()
        if count >= 2
    )
    if len(occupied) < 2:
        return None
    low_bin, high_bin = occupied[0], occupied[-1]
    grouped = local.groupby(["bin", "case"], sort=True)[
        [TARGET, "prediction"]
    ].mean()
    low = grouped.loc[low_bin]
    high = grouped.loc[high_bin]
    paired = low.join(high, lsuffix="_low", rsuffix="_high", how="inner")
    if len(paired) < 2:
        return None
    target_difference = paired[f"{TARGET}_high"] - paired[f"{TARGET}_low"]
    prediction_difference = paired.prediction_high - paired.prediction_low
    target = finite_stat(target_difference.to_numpy(float))
    predicted = finite_stat(prediction_difference.to_numpy(float))
    return {
        "feature": feature,
        "low_bin": int(low_bin),
        "high_bin": int(high_bin),
        "n_paired_cases": int(len(paired)),
        "target_high_minus_low": target,
        "prediction_high_minus_low": predicted,
    }


def plot_curves(curves: pd.DataFrame, features: list[str], labels: dict,
                threshold: float, n_test: int, stem: str) -> None:
    configure_style()
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.0, 2.45 * nrows), constrained_layout=True,
        squeeze=False,
    )
    bound_values = []
    for feature in features:
        local = curves.loc[curves.feature == feature]
        bound_values.extend(np.abs(local.target_mean).to_list())
        bound_values.extend(np.abs(local.prediction_mean).to_list())
        bound_values.extend(np.abs(local.target_mean + local.target_case_sem).to_list())
        bound_values.extend(np.abs(local.target_mean - local.target_case_sem).to_list())
    bound = max(float(np.nanmax(bound_values) * 1.08), 0.01)
    for panel, (axis, feature) in enumerate(zip(axes.flat, features)):
        local = curves.loc[curves.feature == feature].sort_values("bin")
        x = local.feature_median.to_numpy(float)
        axis.errorbar(
            x, local.target_mean, yerr=local.target_case_sem,
            color="#000000", marker="o", linewidth=1.0, markersize=3.3,
            capsize=1.8, label="Held-out residual",
        )
        axis.plot(
            x, local.prediction_mean, color="#0072B2", marker="s",
            markersize=3.0, linestyle="--", linewidth=1.0,
            label="Bias emulator",
        )
        axis.axhline(0, color="0.60", linestyle=":", linewidth=0.6)
        axis.set_ylim(-bound, bound)
        axis.set_xlabel(labels.get(feature, feature))
        axis.set_ylabel("truth - V2.2")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14, 1.05, ascii_uppercase[panel], transform=axis.transAxes,
            fontweight="bold", fontsize=9, va="top",
        )
        if panel == 0:
            axis.legend(frameon=False, loc="best")
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    fig.suptitle(
        "Held-out 1D conditional bias — panel E high-response tail\n"
        f"V2.2 scene response >= {threshold:.5f}; n={n_test:,}",
        fontsize=10,
    )
    save_figure(fig, stem)


def markdown(payload: dict) -> str:
    global_result = payload["test_global"]
    lines = [
        "# Panel E high-response-tail conditional bias",
        "",
        "The subset is the frozen rightmost bin of the original all-anchor "
        "`scene_prediction` panel. The full bias emulator is unchanged and is "
        "not refit on this subset.",
        "",
        f"- Frozen rule: `scene_prediction >= "
        f"{payload['tail_definition']['threshold']:.9g}`.",
        f"- Development rows: `{payload['rows']['development']:,}`; final-test "
        f"rows: `{payload['rows']['test']:,}`.",
        f"- Held-out measured bias: "
        f"`{global_result['target']['mean']:+.6f} +- "
        f"{global_result['target']['case_sem']:.6f}`.",
        f"- Bias-emulator mean: "
        f"`{global_result['prediction']['mean']:+.6f} +- "
        f"{global_result['prediction']['case_sem']:.6f}`.",
        f"- Measured minus emulator: "
        f"`{global_result['target_minus_prediction']['mean']:+.6f} +- "
        f"{global_result['target_minus_prediction']['case_sem']:.6f}`.",
        "",
        "## Extreme-bin contrasts",
        "",
        "These are exploratory high-minus-low feature-bin contrasts on the "
        "already-opened final-test subset; case is the uncertainty unit.",
        "",
        "| feature | measured high-low | case SEM | emulator high-low |",
        "|---|---:|---:|---:|",
    ]
    ordered = sorted(
        payload["extreme_contrasts"],
        key=lambda item: abs(item["target_high_minus_low"]["t"]),
        reverse=True,
    )
    for item in ordered:
        observed = item["target_high_minus_low"]
        predicted = item["prediction_high_minus_low"]
        lines.append(
            f"| `{item['feature']}` | {observed['mean']:+.6f} | "
            f"{observed['case_sem']:.6f} | {predicted['mean']:+.6f} |"
        )
    lines.extend([
        "",
        "Black points in the figure are held-out measured residuals with "
        "case-level SEM; blue squares are predictions from the same globally "
        "trained full bias emulator.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--parent-json", required=True)
    ap.add_argument("--parent-curves", required=True)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    outputs = [
        args.output_prefix + suffix for suffix in
        (".json", ".md", "_curves.csv", ".png", ".pdf")
    ]
    for path in outputs:
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    with open(args.parent_json, encoding="utf-8") as handle:
        parent = json.load(handle)
    display_features = parent["selected_curve_features"]
    labels = parent["feature_labels"]
    windows = parent["case_windows"]
    train_min = int(windows["train"][0])
    tune_max = int(windows["tune"][1])
    test_min, test_max = map(int, windows["test"])

    parent_curves = pd.read_csv(args.parent_curves)
    definition = frozen_tail_definition(parent_curves)
    threshold = definition["threshold"]
    n_bins = definition["parent_curve_bins"]

    frame = pd.read_feather(
        args.features, columns=[*KEY, TARGET, *display_features]
    )
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate feature-table key")
    development = frame.loc[frame.case.between(train_min, tune_max)].copy()
    test_all = frame.loc[frame.case.between(test_min, test_max)].copy()

    replay_threshold = quantile_edges(
        development.scene_prediction.to_numpy(float), n_bins
    )[-2]
    if not np.isclose(replay_threshold, threshold, rtol=0.0, atol=2e-12):
        raise RuntimeError(
            f"panel-E threshold does not replay: {replay_threshold} != {threshold}"
        )
    development = development.loc[tail_mask(development, threshold)].copy()
    test = test_all.loc[tail_mask(test_all, threshold)].copy()
    if len(test) != definition["parent_test_rows"]:
        raise RuntimeError(
            f"tail row count does not replay panel E: "
            f"{len(test)} != {definition['parent_test_rows']}"
        )

    prediction = pd.read_feather(
        args.predictions, columns=[*KEY, "predicted_bias_full"]
    )
    if prediction.duplicated(KEY).any():
        raise RuntimeError("duplicate prediction key")
    test = test.merge(prediction, on=KEY, how="left", validate="one_to_one")
    if test.predicted_bias_full.isna().any():
        raise RuntimeError("tail rows lack full-model predictions")
    predicted = test.predicted_bias_full.to_numpy(float)

    rows = []
    contrasts = []
    for feature in display_features:
        rows.extend(conditional_curve(
            development, test, predicted, feature, POPULATION, n_bins
        ))
        contrast = extreme_contrast(
            development, test, predicted, feature, n_bins
        )
        if contrast is not None:
            contrasts.append(contrast)
    curves = pd.DataFrame(rows)
    if set(curves.feature) != set(display_features):
        missing = sorted(set(display_features) - set(curves.feature))
        raise RuntimeError(f"missing displayed tail curves: {missing}")

    test_global_frame = test[["case", TARGET]].copy()
    test_global_frame["prediction"] = predicted
    test_global_frame["error"] = test_global_frame[TARGET] - predicted
    payload = {
        "design": (
            "post-hoc localization inside frozen panel-E rightmost bin; "
            "same globally trained full bias emulator, no tail refit"
        ),
        "target": TARGET,
        "target_sign": "positive means V2.2 underpredicts",
        "case_windows": windows,
        "tail_definition": definition,
        "curve_definition": (
            "development-tail quantile edges applied to final-test tail; "
            "measured target and emulator prediction are case-balanced means"
        ),
        "curve_bins": n_bins,
        "display_features": display_features,
        "rows": {
            "development": int(len(development)),
            "test": int(len(test)),
            "test_all": int(len(test_all)),
            "test_fraction": float(len(test) / len(test_all)),
            "test_cases": int(test.case.nunique()),
        },
        "test_global": {
            "target": case_stat(test_global_frame, TARGET),
            "prediction": case_stat(test_global_frame, "prediction"),
            "target_minus_prediction": case_stat(test_global_frame, "error"),
        },
        "extreme_contrasts": contrasts,
        "artifacts": {
            "curves_csv": os.path.abspath(args.output_prefix + "_curves.csv"),
            "figure_png": os.path.abspath(args.output_prefix + ".png"),
            "figure_pdf": os.path.abspath(args.output_prefix + ".pdf"),
        },
    }

    curves.to_csv(args.output_prefix + "_curves.csv", index=False)
    with open(args.output_prefix + ".json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    Path(args.output_prefix + ".md").write_text(
        markdown(payload), encoding="utf-8"
    )
    plot_curves(
        curves, display_features, labels, threshold, len(test),
        args.output_prefix,
    )
    print(json.dumps(payload["test_global"], indent=2, sort_keys=True))
    print(f"wrote {args.output_prefix}.*\nANCHOR_BIAS_SCENE_TAIL_CURVES_DONE")


if __name__ == "__main__":
    main()
