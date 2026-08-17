"""Compare V2.2 and the pair-other-flux retrain on held-out pair truth.

Cases 0--39 are excluded from training.  Per-pair predictions are summed by
primary before the global comparison; conditional profiles use fixed physical
pair inputs or bins defined without ruler truth.  Uncertainties are delete-one-
case jackknife SEMs.  No constgold quantity is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blendemu.scene_features import (
    PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS,
    pair_other_absolute_flux,
)
from scripts.diag_rblend_otherflux_halfshear import zero_plus_positive_quartile_edges
from scripts.diag_rblend_scene3_halfshear import (
    OLD_BLENDNESS_EDGES,
    SCENE_MODEL,
    attach_scene,
    binned_table,
    factorize_primary,
    jackknife_mean,
    quantile_edges,
    read_pairset,
    read_scene_lookup,
    score,
)


BASELINE = "v22"
CANDIDATE = "other3abs"


def overall_stat(values, cases):
    mean, sem, n = jackknife_mean(
        np.asarray(values, float), np.asarray(cases), np.ones(len(values), bool),
    )
    return {"mean": mean, "case_jackknife_sem": sem, "n": n}


def residual_metrics(table, label):
    use = table.axis.str.startswith("other_abs_") & (table.n >= 200)
    values = table.loc[use, f"{label}_minus_truth_mean"].to_numpy(float)
    return {
        "n_bins": int(len(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(np.max(np.abs(values))),
    }


def make_figure(table, prefix):
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=True)
    colors = {"truth": "#000000", BASELINE: "#0072B2", CANDIDATE: "#D55E00"}
    markers = {"truth": "o", BASELINE: "s", CANDIDATE: "^"}
    for shell in range(3):
        subset = table[(table.axis == f"other_abs_{shell}") & (table.n >= 200)].copy()
        x = np.arange(len(subset))
        for label in ("truth", BASELINE, CANDIDATE):
            axes[0, shell].errorbar(
                x, subset[f"{label}_mean"], yerr=subset[f"{label}_sem"],
                color=colors[label], marker=markers[label], lw=1.1, capsize=2,
                label=label,
            )
        for label in (BASELINE, CANDIDATE):
            axes[1, shell].errorbar(
                x, subset[f"{label}_minus_truth_mean"],
                yerr=subset[f"{label}_minus_truth_sem"],
                color=colors[label], marker=markers[label], lw=1.1, capsize=2,
                label=f"{label} - truth",
            )
        axes[1, shell].axhline(0, color="0.7", lw=0.8)
        labels = ["zero" if row.lo < 0 and row.hi <= np.nextafter(0.0, 1.0)
                  else f"{row.x_mean:.2f}" for row in subset.itertuples()]
        for row in (0, 1):
            axes[row, shell].set_xticks(x, labels)
            axes[row, shell].spines[["top", "right"]].set_visible(False)
        axes[0, shell].set_title(("0--1", "1--3", "3--10")[shell] + " arcsec")
        axes[1, shell].set_xlabel(r"mean $\log_{10}(1+F_{other})$")
    axes[0, 0].set_ylabel(r"pair $R_{blend}$")
    axes[1, 0].set_ylabel("prediction - truth")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.suptitle("Held-out pair response; primary and designated secondary excluded")
    fig.savefig(prefix + ".png", dpi=300)
    fig.savefig(prefix + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairset", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
        "blendflow/blend_pairset_ap7.feather",
    )
    ap.add_argument("--scene-lookup", action="append", required=True)
    ap.add_argument("--case-min", type=int, default=0)
    ap.add_argument("--case-max", type=int, default=40)
    ap.add_argument("--tag-baseline", default="lsst_r_extnbr_v22")
    ap.add_argument("--tag-candidate", default="lsst_r_extnbr_v22_other3abs")
    ap.add_argument("--score-chunk", type=int, default=500_000)
    ap.add_argument("--zero-point", type=float, default=30.0)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    pair = attach_scene(
        read_pairset(args.pairset, args.case_min, args.case_max),
        read_scene_lookup(args.scene_lookup, args.case_min, args.case_max),
    )
    features = pair_other_absolute_flux(
        pair[SCENE_MODEL].to_numpy(float),
        pair["r_input_p"].to_numpy(float), pair["r_input_s"].to_numpy(float),
        pair["distance"].to_numpy(float), zero_point=args.zero_point,
    )
    for index, name in enumerate(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS):
        pair[name] = features[:, index].astype(np.float32)
    pair["pred_v22"] = score(pair, args.tag_baseline, args.score_chunk)
    pair["pred_other3abs"] = score(pair, args.tag_candidate, args.score_chunk)

    primary_index, primary = factorize_primary(pair)
    n_primary = len(primary)
    primary["truth"] = np.bincount(
        primary_index, weights=pair["blend_truth"], minlength=n_primary,
    )
    primary["null"] = np.bincount(
        primary_index, weights=pair["blend_null"], minlength=n_primary,
    )
    primary[BASELINE] = np.bincount(
        primary_index, weights=pair["pred_v22"], minlength=n_primary,
    )
    primary[CANDIDATE] = np.bincount(
        primary_index, weights=pair["pred_other3abs"], minlength=n_primary,
    )
    primary["n_pairs"] = np.bincount(primary_index, minlength=n_primary)

    primary_series = {
        label: primary[label].to_numpy(float)
        for label in ("truth", "null", BASELINE, CANDIDATE)
    }
    primary_series.update({
        f"{BASELINE}_minus_truth": primary_series[BASELINE] - primary_series["truth"],
        f"{CANDIDATE}_minus_truth": primary_series[CANDIDATE] - primary_series["truth"],
        f"{CANDIDATE}_minus_{BASELINE}": primary_series[CANDIDATE] - primary_series[BASELINE],
    })
    pair_series = {
        "truth": pair["blend_truth"].to_numpy(float),
        "null": pair["blend_null"].to_numpy(float),
        BASELINE: pair["pred_v22"].to_numpy(float),
        CANDIDATE: pair["pred_other3abs"].to_numpy(float),
    }
    pair_series.update({
        f"{BASELINE}_minus_truth": pair_series[BASELINE] - pair_series["truth"],
        f"{CANDIDATE}_minus_truth": pair_series[CANDIDATE] - pair_series["truth"],
        f"{CANDIDATE}_minus_{BASELINE}": pair_series[CANDIDATE] - pair_series[BASELINE],
    })
    primary_cases = primary["case"].to_numpy(np.int32)
    pair_cases = pair["case"].to_numpy(np.int32)

    tables = []
    baseline_sum = primary[BASELINE].to_numpy(float)
    tables.append(binned_table(
        "predicted_blendness_decile", baseline_sum, quantile_edges(baseline_sum, 10),
        primary_series, primary_cases, "primary",
    ))
    tables.append(binned_table(
        "old_blendness", baseline_sum, OLD_BLENDNESS_EDGES,
        primary_series, primary_cases, "primary",
    ))
    for shell, name in enumerate(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS):
        coordinate = pair[name].to_numpy(float)
        tables.append(binned_table(
            f"other_abs_{shell}", coordinate,
            zero_plus_positive_quartile_edges(coordinate),
            pair_series, pair_cases, "pair",
        ))
    tables.append(binned_table(
        "separation", pair["distance"].to_numpy(float),
        np.asarray([0, 1, 2, 3, 5, 7, 10]), pair_series, pair_cases, "pair",
    ))
    tables.append(binned_table(
        "secondary_mag", pair["r_input_s"].to_numpy(float),
        np.asarray([13, 18, 22, 24, 26, 28, 29]), pair_series, pair_cases, "pair",
    ))
    table = pd.concat(tables, ignore_index=True)

    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(str(output) + ".csv", index=False)
    overall = {
        label: overall_stat(primary_series[label], primary_cases)
        for label in ("truth", "null", BASELINE, CANDIDATE,
                      f"{BASELINE}_minus_truth", f"{CANDIDATE}_minus_truth",
                      f"{CANDIDATE}_minus_{BASELINE}")
    }
    metrics = {
        BASELINE: residual_metrics(table, BASELINE),
        CANDIDATE: residual_metrics(table, CANDIDATE),
    }
    gates = {
        "heldout_global_abs_residual_smaller": (
            abs(overall[f"{CANDIDATE}_minus_truth"]["mean"])
            < abs(overall[f"{BASELINE}_minus_truth"]["mean"])
        ),
        "other_flux_equal_bin_rms_smaller": (
            metrics[CANDIDATE]["rms"] < metrics[BASELINE]["rms"]
        ),
        "other_flux_max_abs_residual_smaller": (
            metrics[CANDIDATE]["max_abs"] < metrics[BASELINE]["max_abs"]
        ),
    }
    summary = {
        "design": "held-out half-shear pair ruler; candidate frozen before scoring",
        "case_window": [args.case_min, args.case_max - 1],
        "pair_rows": int(len(pair)), "primaries": int(n_primary),
        "mean_pairs_per_primary": float(primary["n_pairs"].mean()),
        "feature_names": list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS),
        "feature_definition": (
            "log10(1+absolute intrinsic other-galaxy flux); primary and designated "
            "secondary excluded separately for every pair row"
        ),
        "overall": overall, "other_flux_conditional_metrics": metrics,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "constgold_opened": False,
    }
    with open(str(output) + ".json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    make_figure(table, str(output))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"saved {output}.csv/.json/.png/.pdf", flush=True)
    print("RBLEND_OTHER3ABS_HALFSHEAR_DONE", flush=True)


if __name__ == "__main__":
    main()
