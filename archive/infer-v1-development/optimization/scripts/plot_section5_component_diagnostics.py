#!/usr/bin/env python3
# Archived Infer V1 optimization diagnostic.
"""Visualize g1/g2 asymmetry and finite-shear non-quadraticity.

This is an analysis-only script.  It reads the retained Section 5 closure
products and the deployed flow checkpoint; it performs no flow evaluation and
does not regenerate mocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#6B7280"
GREEN = "#009E73"


def _json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _component_entry(result: dict, component: str, n_draws: int) -> dict:
    matches = [
        row
        for row in result["result"]["estimates"]
        if row["component"] == component and row["n_draws"] == n_draws
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {component}, M={n_draws} entry; found {len(matches)}"
        )
    return matches[0]


def _profile(
    root: Path,
    component: str,
    pattern: str = "finite_profile_g002_{component}_positive_v1",
) -> tuple[dict, dict]:
    result = _json(root / pattern.format(component=component) / "result.json")
    rung = max(result["profile"]["rungs"], key=lambda row: row["n_draws"])
    zero = min(result["zero_expansions"], key=lambda row: abs(row["h"] - 0.00125))
    return rung, zero


def _curvature_summary(
    root: Path,
    component: str,
    pattern: str = "curvature_scan_g002_{component}_m16384_v1",
    magnitude_range: tuple[float, float] | None = None,
) -> dict:
    run = root / pattern.format(component=component)
    arrays = np.load(run / "object_curvature.npz")
    result = _json(run / "result.json")
    centers = arrays["centers"].astype(float)
    marginal = arrays["m16384_marginal_information"].astype(float)
    true_atom = arrays["m16384_true_atom_information"].astype(float)
    conditional = arrays["m16384_posterior_conditional_information"].astype(float)
    posterior_variance = arrays["m16384_posterior_score_variance"].astype(float)

    measurements = pd.read_parquet(
        Path(result["config"]["mock_input"]) / "measurements.parquet"
    )
    e1 = measurements["measured_ngmix_g1"].to_numpy(float)
    e2 = measurements["measured_ngmix_g2"].to_numpy(float)
    e_abs = np.hypot(e1, e2)
    magnitude = measurements["measured_mag_auto"].to_numpy(float)
    log_radius = measurements["measured_log_flux_radius"].to_numpy(float)

    delta_information = marginal[-1] - marginal[0]
    abs_delta = np.abs(delta_information)
    top = abs_delta >= np.quantile(abs_delta, 0.99)
    magnitude_edges = np.quantile(magnitude, np.linspace(0.0, 1.0, 6))
    magnitude_fraction = []
    magnitude_centers = []
    for index, (lower, upper) in enumerate(zip(magnitude_edges[:-1], magnitude_edges[1:])):
        include = (magnitude >= lower) & (
            (magnitude <= upper) if index == 4 else (magnitude < upper)
        )
        magnitude_centers.append(float(np.median(magnitude[include])))
        magnitude_fraction.append(
            float(delta_information[include].sum() / delta_information.sum())
        )

    def _mean_sem(values: np.ndarray) -> tuple[list[float], list[float]]:
        means = values.mean(axis=1)
        sems = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
        return means.tolist(), sems.tolist()

    marginal_mean, marginal_sem = _mean_sem(marginal)
    true_mean, true_sem = _mean_sem(true_atom)
    conditional_mean, conditional_sem = _mean_sem(conditional)
    variance_mean, variance_sem = _mean_sem(posterior_variance)

    summary = {
        "centers": centers.tolist(),
        "marginal_information_mean": marginal_mean,
        "marginal_information_sem": marginal_sem,
        "true_atom_information_mean": true_mean,
        "true_atom_information_sem": true_sem,
        "posterior_conditional_information_mean": conditional_mean,
        "posterior_conditional_information_sem": conditional_sem,
        "posterior_score_variance_mean": variance_mean,
        "posterior_score_variance_sem": variance_sem,
        "fraction_negative_marginal_information": (marginal < 0).mean(axis=1).tolist(),
        "delta_information_mean": float(delta_information.mean()),
        "delta_information_sem": float(
            delta_information.std(ddof=1) / np.sqrt(delta_information.size)
        ),
        "top_one_percent_signed_fraction": float(
            delta_information[top].sum() / delta_information.sum()
        ),
        "top_one_percent_absolute_fraction": float(abs_delta[top].sum() / abs_delta.sum()),
        "magnitude_spearman_abs_delta": float(spearmanr(magnitude, abs_delta).statistic),
        "log_radius_spearman_abs_delta": float(spearmanr(log_radius, abs_delta).statistic),
        "ellipticity_spearman_abs_delta": float(spearmanr(e_abs, abs_delta).statistic),
        "all_medians": {
            "measured_magnitude": float(np.median(magnitude)),
            "measured_log_radius": float(np.median(log_radius)),
            "measured_ellipticity": float(np.median(e_abs)),
        },
        "top_one_percent_medians": {
            "measured_magnitude": float(np.median(magnitude[top])),
            "measured_log_radius": float(np.median(log_radius[top])),
            "measured_ellipticity": float(np.median(e_abs[top])),
        },
        "magnitude_quintile_edges": magnitude_edges.tolist(),
        "magnitude_quintile_centers": magnitude_centers,
        "magnitude_quintile_signed_fraction": magnitude_fraction,
    }
    if magnitude_range is not None:
        lower_limit, upper_limit = magnitude_range
        in_range = (magnitude >= lower_limit) & (magnitude <= upper_limit)
        if in_range.sum() < 5:
            raise ValueError(
                f"only {in_range.sum()} objects fall within magnitude range "
                f"[{lower_limit}, {upper_limit}]"
            )
        range_edges = np.quantile(magnitude[in_range], np.linspace(0.0, 1.0, 6))
        range_edges[0] = lower_limit
        range_edges[-1] = upper_limit
        range_fraction_total = []
        range_fraction_selected = []
        range_centers = []
        range_delta = delta_information[in_range].sum()
        for index, (lower, upper) in enumerate(
            zip(range_edges[:-1], range_edges[1:])
        ):
            include = in_range & (magnitude >= lower) & (
                (magnitude <= upper) if index == 4 else (magnitude < upper)
            )
            range_centers.append(float(np.median(magnitude[include])))
            bin_delta = delta_information[include].sum()
            range_fraction_total.append(float(bin_delta / delta_information.sum()))
            range_fraction_selected.append(float(bin_delta / range_delta))
        summary["magnitude_range_zoom"] = {
            "limits": [float(lower_limit), float(upper_limit)],
            "n_objects": int(in_range.sum()),
            "fraction_of_objects": float(in_range.mean()),
            "signed_fraction_of_total": float(
                range_delta / delta_information.sum()
            ),
            "quintile_edges": range_edges.tolist(),
            "quintile_centers": range_centers,
            "quintile_signed_fraction_of_total": range_fraction_total,
            "quintile_signed_fraction_within_range": range_fraction_selected,
        }
    return summary


def _mock_mean_response(root: Path, run_name: str, amplitude: float) -> dict:
    run = root / run_name
    matrix = np.empty((2, 2), dtype=float)
    standard_error = np.empty((2, 2), dtype=float)
    columns = ["measured_ngmix_g1", "measured_ngmix_g2"]
    for injected_index, component in enumerate(("g1", "g2")):
        positive = pd.read_parquet(
            run / f"mock_{component}_positive" / "measurements.parquet", columns=columns
        ).to_numpy(float)
        negative = pd.read_parquet(
            run / f"mock_{component}_negative" / "measurements.parquet", columns=columns
        ).to_numpy(float)
        response = (positive - negative) / (2.0 * amplitude)
        matrix[:, injected_index] = response.mean(axis=0)
        standard_error[:, injected_index] = response.std(axis=0, ddof=1) / np.sqrt(len(response))
    return {
        "amplitude": float(amplitude),
        "matrix": matrix.tolist(),
        "standard_error": standard_error.tolist(),
    }


def _intrinsic_jacobian_for_mock_rows(root: Path, run_name: str) -> dict:
    truth = pd.read_parquet(
        root / run_name / "mock_g1_positive" / "truth.parquet", columns=["scene_row"]
    )
    rows = truth["scene_row"].to_numpy(np.int64)
    parquet = pq.ParquetFile(root / "model_cache_section5_v1" / "flow_zero.parquet")
    ellipticity = np.empty((len(rows), 2), dtype=float)
    start = 0
    for row_group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        stop = start + count
        include = (rows >= start) & (rows < stop)
        if np.any(include):
            frame = parquet.read_row_group(
                row_group, columns=["e1_input_p", "e2_input_p"]
            ).to_pandas()
            local = rows[include] - start
            ellipticity[include, 0] = frame["e1_input_p"].to_numpy()[local]
            ellipticity[include, 1] = frame["e2_input_p"].to_numpy()[local]
        start = stop
    e1 = ellipticity[:, 0]
    e2 = ellipticity[:, 1]
    a = e1**2 - e2**2
    b = 2.0 * e1 * e2
    jacobian = np.empty((len(e1), 2, 2), dtype=float)
    jacobian[:, 0, 0] = 1.0 - a
    jacobian[:, 0, 1] = -b
    jacobian[:, 1, 0] = -b
    jacobian[:, 1, 1] = 1.0 + a
    return {
        "n_objects": int(len(rows)),
        "mean": jacobian.mean(axis=0).tolist(),
        "standard_error": (
            jacobian.std(axis=0, ddof=1) / np.sqrt(len(jacobian))
        ).tolist(),
    }


def build_summary(root: Path, checkpoint: Path, simulation_response: Path) -> dict:
    exact = _json(root / "powered_exact_h00125_v1" / "result.json")
    importance = _json(root / "powered_importance_n10000_h00125_v1" / "result.json")
    exact_g1 = next(row for row in exact["exact_result"]["estimates"] if row["component"] == "g1")
    exact_g2 = next(row for row in exact["exact_result"]["estimates"] if row["component"] == "g2")
    importance_g1 = _component_entry(importance, "g1", 65536)
    importance_g2 = _component_entry(importance, "g2", 65536)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target_scales = np.asarray(checkpoint_payload["target_transform"]["scales"], dtype=float)
    model_config = checkpoint_payload["model_config"]

    profiles = {}
    for component in ("g1", "g2"):
        rung, zero = _profile(root, component)
        profiles[component] = {
            "n_draws": int(rung["n_draws"]),
            "shears": [float(row["shear"]) for row in rung["points"]],
            "log_likelihood": [float(row["log_likelihood_sum"]) for row in rung["points"]],
            "profile_estimate": float(rung["estimated_shear"]),
            "profile_information": float(rung["quadratic_information"]),
            "zero_score": float(zero["score_sum"]),
            "zero_information": float(zero["information_sum"]),
            "zero_one_step_estimate": float(zero["one_step_shear"]),
        }

    g002_response = _mock_mean_response(
        root, "nonzero_plain_n10000_both_g002_k32768_m16384_seed8701_v1", 0.02
    )
    g005_response = _mock_mean_response(
        root, "nonzero_plain_n10000_both_g005_k32768_m16384_seed8701_v1", 0.05
    )
    response_matrix = np.asarray(g002_response["matrix"])
    simulation = _json(simulation_response)
    simulation_matrix = np.asarray(simulation["response_matrix"])
    return {
        "source_root": str(root),
        "measurement_checkpoint": str(checkpoint),
        "flow": {
            "flow_type": model_config["flow_type"],
            "target_scales": target_scales.tolist(),
            "shape_target_scale_ratio_g2_over_g1": float(target_scales[1] / target_scales[0]),
            "response_regularizer": "trace_over_two",
            "response_delta": 0.02,
        },
        "measured_mock_mean_response": {
            "g002": g002_response,
            "g005": g005_response,
        },
        "simulation_component_response": simulation,
        "intrinsic_shear_jacobian_selected_rows": _intrinsic_jacobian_for_mock_rows(
            root, "nonzero_plain_n10000_both_g002_k32768_m16384_seed8701_v1"
        ),
        "component_ratios": {
            "simulation_squared_mean_response_g2_over_g1": float(
                (simulation_matrix[1, 1] / simulation_matrix[0, 0]) ** 2
            ),
            "squared_mean_response_g2_over_g1": float(
                (response_matrix[1, 1] / response_matrix[0, 0]) ** 2
            ),
            "exact_null_information_g2_over_g1": float(
                exact_g2["information_mean"] / exact_g1["information_mean"]
            ),
            "full_prior_null_information_g2_over_g1": float(
                importance_g2["information_mean"] / importance_g1["information_mean"]
            ),
            "exact_null_robust_se_g2_over_g1": float(
                exact_g2["robust_standard_error"] / exact_g1["robust_standard_error"]
            ),
            "full_prior_null_robust_se_g2_over_g1": float(
                importance_g2["robust_standard_error"]
                / importance_g1["robust_standard_error"]
            ),
        },
        "exact_null": {"g1": exact_g1, "g2": exact_g2},
        "full_prior_null": {"g1": importance_g1, "g2": importance_g2},
        "g002_curvature": {
            "g1": _curvature_summary(root, "g1"),
            "g2": _curvature_summary(root, "g2"),
        },
        "g002_profiles": profiles,
    }


def build_curvature_summary(
    root: Path,
    checkpoint: Path,
    profile_pattern: str,
    curvature_pattern: str,
    magnitude_range: tuple[float, float] | None = None,
) -> dict:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    profiles = {}
    for component in ("g1", "g2"):
        rung, zero = _profile(root, component, profile_pattern)
        profiles[component] = {
            "n_draws": int(rung["n_draws"]),
            "shears": [float(row["shear"]) for row in rung["points"]],
            "log_likelihood": [float(row["log_likelihood_sum"]) for row in rung["points"]],
            "profile_estimate": float(rung["estimated_shear"]),
            "profile_information": float(rung["quadratic_information"]),
            "zero_score": float(zero["score_sum"]),
            "zero_information": float(zero["information_sum"]),
            "zero_one_step_estimate": float(zero["one_step_shear"]),
        }
    return {
        "source_root": str(root),
        "measurement_checkpoint": str(checkpoint),
        "flow": {
            "flow_type": checkpoint_payload["model_config"]["flow_type"],
            "response_regularizer": "full_matrix",
            "response_delta": 0.02,
        },
        "g002_curvature": {
            component: _curvature_summary(
                root,
                component,
                curvature_pattern,
                magnitude_range=magnitude_range,
            )
            for component in ("g1", "g2")
        },
        "g002_profiles": profiles,
    }


def make_figure(summary: dict, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    # A: directional ratios at three increasingly marginalized stages.
    ax = axes[0, 0]
    ratios = summary["component_ratios"]
    labels = [
        "simulation response\n(squared)",
        "flow response\n(squared)",
        "known-atom\ninformation",
        "catalogue-marginal\ninformation",
    ]
    values = [
        ratios["simulation_squared_mean_response_g2_over_g1"],
        ratios["squared_mean_response_g2_over_g1"],
        ratios["exact_null_information_g2_over_g1"],
        ratios["full_prior_null_information_g2_over_g1"],
    ]
    bars = ax.bar(np.arange(4), values, color=[GREEN, GREY, BLUE, ORANGE], width=0.68)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylim(0.6, 1.035)
    ax.set_ylabel(r"ratio $g_2/g_1$")
    ax.set_xticks(np.arange(4), labels)
    ax.set_title("a  The two shear components are not equivalent")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    ax.text(
        0.02,
        0.04,
        "robust SE ratio $g_2/g_1$: "
        f"{ratios['exact_null_robust_se_g2_over_g1']:.2f} (known atom), "
        f"{ratios['full_prior_null_robust_se_g2_over_g1']:.2f} (catalogue)\n"
        "mean cross-response: $R_{12}=-0.0210$ (12.0$\sigma$), "
        "$R_{21}=-0.0190$ (10.7$\sigma$)",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8.5,
    )

    # B: the generating-atom flow curvature is nearly flat while catalogue curvature changes.
    ax = axes[0, 1]
    for component, color in (("g1", BLUE), ("g2", ORANGE)):
        curv = summary["g002_curvature"][component]
        centers = np.asarray(curv["centers"])
        ax.errorbar(
            centers,
            curv["true_atom_information_mean"],
            yerr=curv["true_atom_information_sem"],
            color=color,
            marker="o",
            linestyle="--",
            capsize=2,
        )
        ax.errorbar(
            centers,
            curv["marginal_information_mean"],
            yerr=curv["marginal_information_sem"],
            color=color,
            marker="s",
            linestyle="-",
            capsize=2,
        )
    ax.set_xlabel("likelihood expansion centre")
    ax.set_ylabel("mean observed information")
    ax.set_title(r"b  Non-quadraticity appears after catalogue marginalization ($g_{true}=0.02$)")
    ax.set_xticks([0.0, 0.01, 0.02])
    component_handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="$g_1$ injection"),
        Line2D([0], [0], color=ORANGE, lw=2, label="$g_2$ injection"),
        Line2D([0], [0], color="black", marker="o", ls="--", label="generating atom"),
        Line2D([0], [0], color="black", marker="s", ls="-", label="catalogue marginal"),
    ]
    ax.legend(handles=component_handles, ncol=2, loc="center right")

    # C: full profiles versus the quadratic extrapolation made at zero.
    ax = axes[1, 0]
    for component, color in (("g1", BLUE), ("g2", ORANGE)):
        profile = summary["g002_profiles"][component]
        shear = np.asarray(profile["shears"])
        loglike = np.asarray(profile["log_likelihood"])
        keep = (shear >= -0.005) & (shear <= 0.04)
        dense_shear = np.linspace(-0.005, 0.055, 241)
        quadratic = (
            profile["zero_score"] * dense_shear
            - 0.5 * profile["zero_information"] * dense_shear**2
        )
        normalization = loglike.max()
        ax.plot(shear[keep], (loglike - normalization)[keep], color=color, marker="o", ms=3.5)
        ax.plot(dense_shear, quadratic - quadratic.max(), color=color, linestyle="--")
    ax.axvline(0.02, color="black", linewidth=1.0, linestyle=":")
    ax.set_ylim(-430, 20)
    ax.set_xlim(-0.005, 0.055)
    ax.set_xlabel("trial shear")
    ax.set_ylabel(r"$\Delta\log\mathcal{L}$ from profile maximum")
    ax.set_title("c  A zero-centred quadratic is local, not valid out to 0.02")
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="$g_1$"),
            Line2D([0], [0], color=ORANGE, label="$g_2$"),
            Line2D([0], [0], color="black", ls="-", label="full profile"),
            Line2D([0], [0], color="black", ls="--", label="quadratic at zero"),
        ],
        ncol=2,
        loc="lower center",
    )

    # D: signed contribution to the curvature change, split by measured magnitude.
    ax = axes[1, 1]
    width = 0.36
    positions = np.arange(5)
    for component, color, shift in (("g1", BLUE, -width / 2), ("g2", ORANGE, width / 2)):
        curv = summary["g002_curvature"][component]
        ax.bar(
            positions + shift,
            curv["magnitude_quintile_signed_fraction"],
            width=width,
            color=color,
            label=f"${component}$",
        )
    edges = summary["g002_curvature"]["g1"]["magnitude_quintile_edges"]
    tick_labels = [f"{edges[i]:.1f}\u2013{edges[i + 1]:.1f}" for i in range(5)]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, tick_labels, rotation=25, ha="right")
    ax.set_ylabel("fraction of total curvature change")
    ax.set_xlabel("measured-magnitude quintile (bright to faint)")
    ax.set_ylim(-0.08, 1.08)
    ax.set_title("d  Bright, large objects drive the curvature change")
    ax.legend(loc="upper right")
    top_g1 = summary["g002_curvature"]["g1"]["top_one_percent_signed_fraction"]
    top_g2 = summary["g002_curvature"]["g2"]["top_one_percent_signed_fraction"]
    ax.text(
        0.98,
        0.72,
        f"top 1% of objects:\n{top_g1:.0%} ($g_1$), {top_g2:.0%} ($g_2$)",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    fig.suptitle("Section 5 catalogue-prior closure: component asymmetry and finite-shear behavior", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_curvature_figure(summary: dict, output: Path, label: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)

    ax = axes[0]
    for component, color in (("g1", BLUE), ("g2", ORANGE)):
        curv = summary["g002_curvature"][component]
        centers = np.asarray(curv["centers"])
        ax.errorbar(
            centers,
            curv["true_atom_information_mean"],
            yerr=curv["true_atom_information_sem"],
            color=color,
            marker="o",
            linestyle="--",
            capsize=2,
        )
        ax.errorbar(
            centers,
            curv["marginal_information_mean"],
            yerr=curv["marginal_information_sem"],
            color=color,
            marker="s",
            linestyle="-",
            capsize=2,
        )
    ax.set_xlabel("likelihood expansion centre")
    ax.set_ylabel("mean observed information")
    ax.set_title(r"a  Curvature before and after catalogue marginalization ($g_{true}=0.02$)")
    ax.set_xticks([0.0, 0.01, 0.02])
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2, label="$g_1$ injection"),
            Line2D([0], [0], color=ORANGE, lw=2, label="$g_2$ injection"),
            Line2D([0], [0], color="black", marker="o", ls="--", label="generating atom"),
            Line2D([0], [0], color="black", marker="s", ls="-", label="catalogue marginal"),
        ],
        ncol=2,
        loc="best",
    )

    ax = axes[1]
    for component, color in (("g1", BLUE), ("g2", ORANGE)):
        profile = summary["g002_profiles"][component]
        shear = np.asarray(profile["shears"])
        loglike = np.asarray(profile["log_likelihood"])
        keep = (shear >= -0.005) & (shear <= 0.04)
        dense_shear = np.linspace(-0.005, 0.055, 241)
        quadratic = (
            profile["zero_score"] * dense_shear
            - 0.5 * profile["zero_information"] * dense_shear**2
        )
        normalization = loglike.max()
        ax.plot(shear[keep], (loglike - normalization)[keep], color=color, marker="o", ms=3.5)
        ax.plot(dense_shear, quadratic - quadratic.max(), color=color, linestyle="--")
    ax.axvline(0.02, color="black", linewidth=1.0, linestyle=":")
    ax.set_ylim(-430, 20)
    ax.set_xlim(-0.005, 0.055)
    ax.set_xlabel("trial shear")
    ax.set_ylabel(r"$\Delta\log\mathcal{L}$ from profile maximum")
    ax.set_title("b  Full profiles versus the quadratic expansion at zero")
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="$g_1$"),
            Line2D([0], [0], color=ORANGE, label="$g_2$"),
            Line2D([0], [0], color="black", ls="-", label="full profile"),
            Line2D([0], [0], color="black", ls="--", label="quadratic at zero"),
        ],
        ncol=2,
        loc="lower center",
    )

    ax = axes[2]
    width = 0.36
    positions = np.arange(5)
    for component, color, shift in (("g1", BLUE, -width / 2), ("g2", ORANGE, width / 2)):
        curv = summary["g002_curvature"][component]
        ax.bar(
            positions + shift,
            curv["magnitude_quintile_signed_fraction"],
            width=width,
            color=color,
            label=f"${component}$",
        )
    edges = summary["g002_curvature"]["g1"]["magnitude_quintile_edges"]
    tick_labels = [f"{edges[i]:.1f}\u2013{edges[i + 1]:.1f}" for i in range(5)]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, tick_labels, rotation=25, ha="right")
    ax.set_ylabel("fraction of total curvature change")
    ax.set_xlabel("measured-magnitude quintile (bright to faint)")
    ax.set_ylim(-0.08, 1.08)
    ax.set_title("c  Objects driving the curvature change")
    ax.legend(loc="upper right")
    top_g1 = summary["g002_curvature"]["g1"]["top_one_percent_signed_fraction"]
    top_g2 = summary["g002_curvature"]["g2"]["top_one_percent_signed_fraction"]
    ax.text(
        0.98,
        0.72,
        f"top 1% of objects:\n{top_g1:.0%} ($g_1$), {top_g2:.0%} ($g_2$)",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    fig.suptitle(f"{label}: Section 5 finite-shear curvature", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_magnitude_zoom_figure(summary: dict, output: Path, label: str) -> None:
    """Plot the contribution of equal-count bins within a magnitude interval."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    width = 0.36
    positions = np.arange(5)
    for component, color, shift, hatch in (
        ("g1", BLUE, -width / 2, None),
        ("g2", ORANGE, width / 2, "//"),
    ):
        zoom = summary["g002_curvature"][component]["magnitude_range_zoom"]
        ax.bar(
            positions + shift,
            zoom["quintile_signed_fraction_of_total"],
            width=width,
            color=color,
            edgecolor=color,
            hatch=hatch,
            label=f"${component}$",
        )
    zoom_g1 = summary["g002_curvature"]["g1"]["magnitude_range_zoom"]
    zoom_g2 = summary["g002_curvature"]["g2"]["magnitude_range_zoom"]
    edges = zoom_g1["quintile_edges"]
    tick_labels = [f"{edges[index]:.1f}–{edges[index + 1]:.1f}" for index in range(5)]
    values = np.concatenate(
        [
            summary["g002_curvature"][component]["magnitude_range_zoom"][
                "quintile_signed_fraction_of_total"
            ]
            for component in ("g1", "g2")
        ]
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, tick_labels, rotation=25, ha="right")
    ax.set_ylabel("fraction of total curvature change")
    ax.set_xlabel("equal-count bins within the bright magnitude range")
    ax.set_ylim(min(-0.03, 1.12 * values.min()), max(0.08, 1.14 * values.max()))
    lower, upper = zoom_g1["limits"]
    ax.set_title(
        rf"c$^\prime$  Curvature contribution within ${lower:g}\leq m\leq{upper:g}$"
    )
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.76,
        "range contribution to total:\n"
        f"{zoom_g1['signed_fraction_of_total']:.1%} ($g_1$), "
        f"{zoom_g2['signed_fraction_of_total']:.1%} ($g_2$)\n"
        f"objects: {zoom_g1['n_objects']:,} ($g_1$), "
        f"{zoom_g2['n_objects']:,} ($g_2$)",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.suptitle(f"{label}: bright-object curvature decomposition", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/"
            "section5_galsbi_100cases_v1"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/"
            "measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/generated/section5_component_diagnostics"),
    )
    parser.add_argument(
        "--simulation-response",
        type=Path,
        default=Path("doc/generated/simulation_component_response_v22.json"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--curvature-only", action="store_true")
    mode.add_argument("--magnitude-panel-only", action="store_true")
    parser.add_argument(
        "--magnitude-range",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(18.0, 23.1),
    )
    parser.add_argument(
        "--profile-pattern",
        default="finite_profile_g002_{component}_positive_v1",
    )
    parser.add_argument(
        "--curvature-pattern",
        default="curvature_scan_g002_{component}_m16384_v1",
    )
    parser.add_argument("--label", default="Experiment E")
    args = parser.parse_args()

    if args.curvature_only or args.magnitude_panel_only:
        summary = build_curvature_summary(
            args.root,
            args.checkpoint,
            args.profile_pattern,
            args.curvature_pattern,
            magnitude_range=(tuple(args.magnitude_range) if args.magnitude_panel_only else None),
        )
        if args.magnitude_panel_only:
            make_magnitude_zoom_figure(summary, args.output, args.label)
        else:
            make_curvature_figure(summary, args.output, args.label)
    else:
        summary = build_summary(args.root, args.checkpoint, args.simulation_response)
        make_figure(summary, args.output)
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
