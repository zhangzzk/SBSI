#!/usr/bin/env python
"""Compare a measurement-flow checkpoint with its effective training distribution.

The script reconstructs the checkpoint's seeded multi-catalogue training reservoir,
keeps only the grouped training split, samples rows with the same per-target weights
used by the NLL, and draws one conditional flow realisation per sampled row.  The
primary comparison applies the production measured cuts independently to data and
flow, then fixes balanced 100k samples.  Contours also show a frozen ConstGold
sample; flow--training divergence is evaluated in true-primary-magnitude bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from sbsi.flow_training import load_measurement_data
from sbsi.measurement_model import load_measurement_model
from sbsi.training import per_target_weights, split_data


TARGETS = [
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
]
PROPERTIES = ["g1", "g2", "mag", "radius"]
PROPERTY_LABELS = [
    r"Measured $\hat g_1$",
    r"Measured $\hat g_2$",
    "Measured MAG_AUTO (mag)",
    "Measured flux radius (arcsec)",
]
TRAINING_COLOR = "#D55E00"
FLOW_COLOR = "#0072B2"
CONSTGOLD_COLOR = "#009E73"
JS_COLOR = "#009E73"
NULL_COLOR = "#666666"
CONTOUR_MASSES = (0.95, 0.80, 0.50)
CONTOUR_STYLES = (":", "--", "-")
CONTOUR_WIDTHS = (0.9, 1.2, 1.7)
TRUE_MAG_EDGES = np.r_[18.0, 19.5, np.arange(20.0, 25.6, 0.5), 25.8]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--g005-catalogue", type=Path, required=True)
    parser.add_argument("--constgold-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--candidate-contexts-per-leg", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--shape-max", type=float, default=0.6)
    parser.add_argument("--mag-max", type=float, default=25.8)
    parser.add_argument("--radius-min", type=float, default=0.75)
    parser.add_argument("--feature-bins", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--dirichlet-total", type=float, default=0.5)
    return parser.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_load_args(args):
    """Exact data-loading knobs from job 15985365 arm E."""
    return SimpleNamespace(
        seed=501,
        catalogue=str(args.g0_catalogue),
        additional_catalogue=[str(args.g005_catalogue)],
        max_rows=4_000_000,
        target_column="detected",
        response_weight=450.0,
        coupling_weight=0.0,
        response_target_perobj=None,
        fold_catalogue_shear=True,
        shear_case=None,
        max_cases=200,
        max_read_batches=None,
        primary_mag_max=25.8,
        primary_re_min=0.5,
        pixel_rms=0.312,
        pixel_size=0.2,
        zero_mag=30.0,
        psf_fwhm=0.73,
        moffat_beta=2.224,
        noise_photoz=0.0,
        noise_sersic_frac=0.0,
        progress_every=100,
        expected_case_count=200,
    )


def load_exact_training_frame(args, checkpoint_metadata):
    conditions = list(checkpoint_metadata["condition_features"])
    targets = list(checkpoint_metadata["target_features"])
    if targets != TARGETS:
        raise ValueError(f"unexpected checkpoint targets: {targets}")
    frame = load_measurement_data(training_load_args(args), conditions, targets)
    train, validation = split_data(frame, 501, 0.2, group_column="case")
    expected = (3_200_042, 799_958)
    if (len(train), len(validation)) != expected:
        raise ValueError(
            f"reconstructed split {(len(train), len(validation))} != expected {expected}"
        )
    train = train.reset_index(drop=True)
    weights = per_target_weights(train)
    if weights is None:
        raise ValueError("training reconstruction lacks input_index for per-target weights")
    train["__training_weight"] = weights
    return train, validation


def weighted_context_sample(train, args):
    if args.sample_size % 2:
        raise ValueError("--sample-size must be even for balanced shear legs")
    if args.candidate_contexts_per_leg < args.sample_size // 2:
        raise ValueError("candidate pool per leg is smaller than the requested balanced sample")
    rng = np.random.default_rng(args.seed)
    parts = []
    diagnostics = {}
    for leg in (0, 1):
        subset = train.loc[train["__catalogue_index"] == leg].reset_index(drop=True)
        weights = subset["__training_weight"].to_numpy(float)
        probabilities = weights / weights.sum()
        indices = rng.choice(
            len(subset),
            size=args.candidate_contexts_per_leg,
            replace=True,
            p=probabilities,
        )
        sampled = subset.iloc[indices].reset_index(drop=True)
        sampled["leg"] = "g0" if leg == 0 else "g005"
        sampled["candidate_index_within_leg"] = np.arange(len(sampled), dtype=np.int64)
        unique_targets = sampled[["case", "input_index"]].drop_duplicates().shape[0]
        diagnostics[str(leg)] = {
            "training_rows": int(len(subset)),
            "training_weight_sum": float(weights.sum()),
            "candidate_contexts": int(len(sampled)),
            "unique_case_input_targets": int(unique_targets),
            "unique_target_fraction": float(unique_targets / len(sampled)),
        }
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True), diagnostics


def draw_flow(checkpoint, contexts, device, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    bundle = load_measurement_model(checkpoint, device=device)
    bundle.model.eval()
    values = bundle.sample(contexts, n_samples=1, batch_size=65_536, qmc=False)[:, 0, :]
    return pd.DataFrame(values, columns=TARGETS)


def measured_values(frame, pixel_size):
    values = pd.DataFrame(
        {
            "g1": frame["measured_ngmix_g1"].to_numpy(float),
            "g2": frame["measured_ngmix_g2"].to_numpy(float),
            "mag": frame["measured_mag_auto"].to_numpy(float),
            "radius": pixel_size
            * np.exp(frame["measured_log_flux_radius"].to_numpy(float)),
        }
    )
    return values


def load_constgold_sample(args):
    frame = pd.read_parquet(args.constgold_sample)
    missing = sorted(set(TARGETS) - set(frame.columns))
    if missing:
        raise KeyError(f"ConstGold sample lacks measurement columns: {missing}")
    if len(frame) != args.sample_size:
        raise ValueError(
            f"ConstGold sample has {len(frame):,} rows; expected {args.sample_size:,}"
        )
    values = measured_values(frame, args.pixel_size)
    if not measured_cut(values, args).all():
        raise ValueError("ConstGold overlay contains rows outside the production measured cuts")
    return values


def measured_cut(values, args):
    array = values.to_numpy(float)
    return (
        np.isfinite(array).all(axis=1)
        & (np.hypot(array[:, 0], array[:, 1]) < args.shape_max)
        & (array[:, 2] < args.mag_max)
        & (array[:, 3] >= args.radius_min)
    )


def fix_selected_samples(contexts, training_values, flow_values, args):
    rng = np.random.default_rng(args.seed + 1)
    n_leg = args.sample_size // 2
    outputs = {"training": [], "flow": []}
    cut_counts = {}
    for leg_index, leg_name in ((0, "g0"), (1, "g005")):
        leg = contexts["__catalogue_index"].to_numpy(int) == leg_index
        cut_counts[leg_name] = {}
        for label, values in (("training", training_values), ("flow", flow_values)):
            eligible = np.flatnonzero(leg & measured_cut(values, args))
            cut_counts[leg_name][f"{label}_passing"] = int(len(eligible))
            if len(eligible) < n_leg:
                raise ValueError(
                    f"{label} {leg_name} has only {len(eligible):,} selected candidates; "
                    f"need {n_leg:,}"
                )
            chosen = rng.choice(eligible, size=n_leg, replace=False)
            part = values.iloc[chosen].reset_index(drop=True)
            part["catalogue_index"] = leg_index
            part["leg"] = leg_name
            part["case"] = contexts.iloc[chosen]["case"].to_numpy(np.int64)
            part["input_index"] = contexts.iloc[chosen]["input_index"].to_numpy(np.int64)
            part["true_mag"] = contexts.iloc[chosen]["r_input_p"].to_numpy(float)
            outputs[label].append(part)
    return (
        pd.concat(outputs["training"], ignore_index=True),
        pd.concat(outputs["flow"], ignore_index=True),
        cut_counts,
    )


def hdr_threshold(histogram, mass):
    density = np.asarray(histogram, dtype=float)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    index = min(int(np.searchsorted(cumulative, mass * cumulative[-1])), len(ordered) - 1)
    return float(ordered[index])


def axis_limits(training, flow, constgold, args):
    joined = pd.concat([training, flow, constgold], ignore_index=True)
    return [
        (-args.shape_max, args.shape_max),
        (-args.shape_max, args.shape_max),
        (float(np.quantile(joined["mag"], 0.001)), args.mag_max),
        (args.radius_min, float(np.quantile(joined["radius"], 0.997))),
    ]


def draw_corner(training, flow, constgold, output, title, args):
    datasets = [
        (training, "Half-shear training data", TRAINING_COLOR),
        (flow, "Conditional flow draws", FLOW_COLOR),
        (constgold, r"ConstGold $+g$ measurements", CONSTGOLD_COLOR),
    ]
    limits = axis_limits(training, flow, constgold, args)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 4, figsize=(9.0, 9.0))
    bins = 150
    for row in range(4):
        for column in range(4):
            ax = axes[row, column]
            if row < column:
                ax.axis("off")
                continue
            if row == column:
                edges = np.linspace(*limits[column], bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                for frame, _, color in datasets:
                    density, _ = np.histogram(frame[PROPERTIES[column]], bins=edges, density=True)
                    ax.plot(
                        centers,
                        gaussian_filter1d(density, sigma=1.2),
                        color=color,
                        linewidth=1.5,
                    )
                ax.set_xlim(limits[column])
                ax.set_yticks([])
            else:
                xedges = np.linspace(*limits[column], bins + 1)
                yedges = np.linspace(*limits[row], bins + 1)
                xc = 0.5 * (xedges[:-1] + xedges[1:])
                yc = 0.5 * (yedges[:-1] + yedges[1:])
                for frame, _, color in datasets:
                    histogram, _, _ = np.histogram2d(
                        frame[PROPERTIES[column]], frame[PROPERTIES[row]], bins=(xedges, yedges)
                    )
                    density = gaussian_filter(histogram.T, sigma=2.0)
                    for mass, style, width in zip(
                        CONTOUR_MASSES, CONTOUR_STYLES, CONTOUR_WIDTHS
                    ):
                        threshold = hdr_threshold(density, mass)
                        if density.min() < threshold < density.max():
                            ax.contour(
                                xc,
                                yc,
                                density,
                                levels=[threshold],
                                colors=[color],
                                linestyles=[style],
                                linewidths=[width],
                            )
                ax.set_xlim(limits[column])
                ax.set_ylim(limits[row])
            if row == 3:
                ax.set_xlabel(PROPERTY_LABELS[column])
            else:
                ax.tick_params(labelbottom=False)
            if column == 0 and row > 0:
                ax.set_ylabel(PROPERTY_LABELS[row])
            elif row > column:
                ax.tick_params(labelleft=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, 2].axis("off")
    axes[0, 2].legend(
        handles=[mlines.Line2D([], [], color=c, linewidth=2, label=l) for _, l, c in datasets],
        loc="center",
        frameon=False,
        fontsize=9,
    )
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.5,
        0.5,
        "Contours enclose\n50% (solid), 80% (dashed),\n95% (dotted)",
        ha="center",
        va="center",
        linespacing=1.5,
    )
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.90, wspace=0.12, hspace=0.10)
    fig.savefig(output / "properties_contours.png", dpi=300)
    fig.savefig(output / "properties_contours.pdf")
    plt.close(fig)


def probabilities(counts, total_pseudocount=0.0):
    counts = np.asarray(counts, dtype=np.float64)
    if total_pseudocount:
        counts = counts + total_pseudocount / counts.size
    return counts / counts.sum()


def kl(p, q):
    active = p > 0
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def js_from_counts(a, b):
    p, q = probabilities(a), probabilities(b)
    middle = 0.5 * (p + q)
    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def fixed_cells(training, flow, bins):
    pool = pd.concat([training, flow], ignore_index=True)
    coordinates = []
    edge_summary = {}
    for feature in ("g1", "g2", "mag", "log_radius"):
        source = np.log(pool["radius"].to_numpy(float)) if feature == "log_radius" else pool[feature].to_numpy(float)
        edges = np.quantile(source, np.linspace(0, 1, bins + 1))
        if np.unique(edges).size != edges.size:
            raise ValueError(f"cannot form {bins} distinct bins for {feature}")
        edge_summary[feature] = edges.tolist()
        coordinates.append(np.searchsorted(edges[1:-1], source, side="right"))
    cells = np.ravel_multi_index(tuple(coordinates), (bins,) * len(coordinates))
    return cells[: len(training)], cells[len(training) :], edge_summary


def conditional_divergence(training, flow, args, seed):
    train_cells, flow_cells, feature_edges = fixed_cells(training, flow, args.feature_bins)
    rng = np.random.default_rng(seed)
    n_cells = args.feature_bins**4
    rows = []
    tmag = training["true_mag"].to_numpy(float)
    fmag = flow["true_mag"].to_numpy(float)
    for bin_index, (low, high) in enumerate(
        zip(TRUE_MAG_EDGES[:-1], TRUE_MAG_EDGES[1:])
    ):
        tm = (tmag >= low) & (tmag < high)
        fm = (fmag >= low) & (fmag < high)
        tc = np.bincount(train_cells[tm], minlength=n_cells)
        fc = np.bincount(flow_cells[fm], minlength=n_cells)
        if tc.sum() == 0 or fc.sum() == 0:
            raise ValueError(
                f"empty sample in true-magnitude bin {bin_index}: [{low}, {high})"
            )
        p = probabilities(tc, args.dirichlet_total)
        q = probabilities(fc, args.dirichlet_total)
        observed_js = js_from_counts(tc, fc)
        pooled = tc + fc
        null = np.empty(args.permutations)
        for index in range(args.permutations):
            permuted_training = rng.multivariate_hypergeometric(pooled, int(tc.sum()))
            null[index] = js_from_counts(permuted_training, pooled - permuted_training)
        q025, median, q975 = np.quantile(null, (0.025, 0.5, 0.975))
        rows.append(
            {
                "true_mag_bin": bin_index,
                "true_mag_low": float(low),
                "true_mag_high": float(high),
                "true_mag_center": float(0.5 * (low + high)),
                "n_training": int(tc.sum()),
                "n_flow": int(fc.sum()),
                "training_fraction": float(tc.sum() / len(training)),
                "flow_fraction": float(fc.sum() / len(flow)),
                "kl_training_to_flow_nats": kl(p, q),
                "kl_flow_to_training_nats": kl(q, p),
                "js_nats": observed_js,
                "js_null_q025_nats": float(q025),
                "js_null_median_nats": float(median),
                "js_null_q975_nats": float(q975),
                "js_excess_over_null_median_nats": float(observed_js - median),
                "permutation_tail_fraction": float(
                    (1 + np.count_nonzero(null >= observed_js)) / (args.permutations + 1)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["weighted_kl_training_to_flow_nats"] = (
        result["training_fraction"] * result["kl_training_to_flow_nats"]
    )
    result["weighted_kl_flow_to_training_nats"] = (
        result["flow_fraction"] * result["kl_flow_to_training_nats"]
    )
    pmag = probabilities(result["n_training"], args.dirichlet_total)
    qmag = probabilities(result["n_flow"], args.dirichlet_total)
    decomposition = {
        "true_magnitude_marginal_kl_training_to_flow_nats": kl(pmag, qmag),
        "true_magnitude_marginal_kl_flow_to_training_nats": kl(qmag, pmag),
        "weighted_conditional_kl_training_to_flow_nats": float(
            result["weighted_kl_training_to_flow_nats"].sum()
        ),
        "weighted_conditional_kl_flow_to_training_nats": float(
            result["weighted_kl_flow_to_training_nats"].sum()
        ),
        "training_weighted_js_nats": float(
            (result["training_fraction"] * result["js_nats"]).sum()
        ),
        "training_weighted_js_null_median_nats": float(
            (result["training_fraction"] * result["js_null_median_nats"]).sum()
        ),
        "training_weighted_js_excess_over_null_median_nats": float(
            (
                result["training_fraction"]
                * result["js_excess_over_null_median_nats"]
            ).sum()
        ),
    }
    decomposition["decomposed_joint_kl_training_to_flow_nats"] = (
        decomposition["true_magnitude_marginal_kl_training_to_flow_nats"]
        + decomposition["weighted_conditional_kl_training_to_flow_nats"]
    )
    decomposition["decomposed_joint_kl_flow_to_training_nats"] = (
        decomposition["true_magnitude_marginal_kl_flow_to_training_nats"]
        + decomposition["weighted_conditional_kl_flow_to_training_nats"]
    )
    return result, feature_edges, decomposition


def draw_divergence(result, output, title):
    centers = result["true_mag_center"].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    upper, lower = axes
    upper.fill_between(
        centers,
        result["js_null_q025_nats"].to_numpy(float),
        result["js_null_q975_nats"].to_numpy(float),
        color=NULL_COLOR,
        alpha=0.18,
        linewidth=0,
        label="Permutation-null JSD 95% interval",
    )
    upper.plot(centers, result["js_null_median_nats"], color=NULL_COLOR, linestyle=":", label="Null median")
    upper.plot(centers, result["kl_training_to_flow_nats"], color=TRAINING_COLOR, marker="o", markersize=3.5, label=r"$D_{KL}(\mathrm{training}\Vert\mathrm{flow})$")
    upper.plot(centers, result["kl_flow_to_training_nats"], color=FLOW_COLOR, marker="s", markersize=3.2, linestyle="--", label=r"$D_{KL}(\mathrm{flow}\Vert\mathrm{training})$")
    upper.plot(centers, result["js_nats"], color=JS_COLOR, marker="^", markersize=3.5, label="Jensen–Shannon")
    upper.set_yscale("log")
    upper.set_ylabel("Conditional divergence (nats)")
    upper.legend(frameon=False, ncol=2, fontsize=8)
    lower.plot(centers, result["weighted_kl_training_to_flow_nats"], color=TRAINING_COLOR, marker="o", markersize=3.5, label="Training-weighted contribution")
    lower.plot(centers, result["weighted_kl_flow_to_training_nats"], color=FLOW_COLOR, marker="s", markersize=3.2, linestyle="--", label="Flow-weighted contribution")
    lower.set_yscale("log")
    lower.set_ylabel("Contribution to conditional KL (nats)")
    lower.set_xlabel(r"True primary magnitude $r_{\rm input,p}$ bin")
    lower.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axvline(22.0, color="black", linestyle="--", linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(TRUE_MAG_EDGES[0], TRUE_MAG_EDGES[-1])
    fig.suptitle(
        title
        + "\n"
        + r"Conditional measured $(\hat g_1,\hat g_2,m,\log R_{\rm flux})$; "
        + "fixed pooled-quantile cells",
        fontsize=10.5,
        y=0.985,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.87, hspace=0.14)
    fig.savefig(output / "conditional_divergence_by_true_magnitude.png", dpi=300)
    fig.savefig(output / "conditional_divergence_by_true_magnitude.pdf")
    plt.close(fig)


def distribution_summary(values):
    summary = {}
    for name in PROPERTIES:
        x = values[name].to_numpy(float)
        summary[name] = {
            "mean": float(x.mean()),
            "standard_deviation": float(x.std(ddof=1)),
            "q01_q50_q99": np.quantile(x, (0.01, 0.5, 0.99)).tolist(),
        }
    return summary


def comparison_summary(training, flow):
    summary = {}
    for name in PROPERTIES:
        x, y = training[name].to_numpy(float), flow[name].to_numpy(float)
        pooled_sd = np.sqrt(0.5 * (x.var(ddof=1) + y.var(ddof=1)))
        # Equal-sized empirical samples: the 1-D Wasserstein distance is the mean
        # absolute quantile separation.  Compute KS directly as well, avoiding
        # scipy.stats' scheduler-side libstdc++ dependency.
        xs, ys = np.sort(x), np.sort(y)
        wd = float(np.mean(np.abs(xs - ys)))
        pooled = np.sort(np.r_[xs, ys])
        ks = float(
            np.max(
                np.abs(
                    np.searchsorted(xs, pooled, side="right") / len(xs)
                    - np.searchsorted(ys, pooled, side="right") / len(ys)
                )
            )
        )
        summary[name] = {
            "flow_minus_training_mean": float(y.mean() - x.mean()),
            "standardized_mean_difference": float((y.mean() - x.mean()) / pooled_sd),
            "ks_statistic": ks,
            "wasserstein_distance": wd,
            "wasserstein_over_pooled_sd": float(wd / pooled_sd),
        }
    return summary


def analyze_subset(training, flow, constgold, output, title, args, seed):
    output.mkdir(parents=True)
    contour_title = (
        title
        + f"\ntraining/flow n={len(training):,} each; ConstGold n={len(constgold):,}"
    )
    draw_corner(training, flow, constgold, output, contour_title, args)
    divergence, feature_edges, decomposition = conditional_divergence(training, flow, args, seed)
    divergence.to_csv(
        output / "conditional_divergence_by_true_magnitude.csv", index=False
    )
    draw_divergence(divergence, output, title)
    summary = {
        "sample_counts": {
            "training": int(len(training)),
            "flow": int(len(flow)),
            "constgold_overlay": int(len(constgold)),
        },
        "training": distribution_summary(training),
        "flow": distribution_summary(flow),
        "constgold_overlay": distribution_summary(constgold),
        "differences": comparison_summary(training, flow),
        "conditional_histogram_edges": feature_edges,
        "divergence_decomposition": decomposition,
        "true_magnitude_bins": divergence.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.feature_bins < 2 or args.permutations < 1 or args.dirichlet_total <= 0:
        raise ValueError("invalid divergence settings")
    started = time.time()
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    train, validation = load_exact_training_frame(args, metadata)
    contexts, sampling = weighted_context_sample(train, args)
    training_values = measured_values(contexts, args.pixel_size)
    flow_frame = draw_flow(args.checkpoint, contexts, args.device, args.seed + 2)
    flow_values = measured_values(flow_frame, args.pixel_size)
    training, flow, cut_counts = fix_selected_samples(
        contexts, training_values, flow_values, args
    )
    constgold = load_constgold_sample(args)

    args.output.mkdir(parents=True)
    training.to_parquet(args.output / "training_selected_100k.parquet", index=False)
    flow.to_parquet(args.output / "flow_selected_100k.parquet", index=False)

    analyses = {}
    subsets = [
        ("combined", training, flow, "Flow versus effective half-shear training distribution"),
        ("g0", training.loc[training["leg"] == "g0"], flow.loc[flow["leg"] == "g0"], "Flow versus $g=0$ training leg"),
        ("g005", training.loc[training["leg"] == "g005"], flow.loc[flow["leg"] == "g005"], r"Flow versus $|g|=0.05$ half-shear training leg"),
    ]
    for index, (name, train_part, flow_part, title) in enumerate(subsets):
        analyses[name] = analyze_subset(
            train_part.reset_index(drop=True),
            flow_part.reset_index(drop=True),
            constgold,
            args.output / name,
            title,
            args,
            args.seed + 100 + index,
        )

    root_summary = {
        "comparison": "conditional measurement-flow draws vs effective weighted training distribution",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "catalogues": {
            "g0": str(args.g0_catalogue.resolve()),
            "g005_half_shear": str(args.g005_catalogue.resolve()),
            "constgold_overlay": str(args.constgold_sample.resolve()),
            "constgold_overlay_sha256": sha256(args.constgold_sample),
        },
        "constgold_overlay": {
            "sample_rows": int(len(constgold)),
            "applied_shear": [0.02, 0.0],
            "role": "contours only; excluded from flow--training divergence",
        },
        "training_reconstruction": {
            "reservoir_seed": 501,
            "reservoir_rows": 4_000_000,
            "train_rows": int(len(train)),
            "validation_rows_excluded": int(len(validation)),
            "group_split": "case, 160 train / 40 validation cases, seed 501",
            "sampling_measure": "per-target weighted empirical training objective; balanced 50/50 by catalogue leg",
            "sampling": sampling,
        },
        "flow_sampling": {
            "seed": int(args.seed + 2),
            "draws_per_condition": 1,
            "qmc": False,
        },
        "measured_selection": {
            "definition": f"hypot(g1,g2)<{args.shape_max}; mag<{args.mag_max}; radius>={args.radius_min} arcsec",
            "applied_independently": True,
            "candidate_counts": cut_counts,
            "fixed_selected_sample_each": int(args.sample_size),
        },
        "divergence": {
            "units": "natural-log nats",
            "independent_variable": "true primary magnitude r_input_p",
            "true_magnitude_edges": TRUE_MAG_EDGES.tolist(),
            "conditional_features": [
                "g1",
                "g2",
                "measured_mag_auto",
                "natural_log_radius_arcsec",
            ],
            "bins_per_conditional_feature": int(args.feature_bins),
            "directed_kl_smoothing": f"symmetric Dirichlet total concentration {args.dirichlet_total:g}",
            "js_smoothing": "none",
            "permutation_reference": "exact multivariate-hypergeometric relabeling within true-magnitude bin",
            "permutations_per_bin": int(args.permutations),
        },
        "unmatched_rows": 0,
        "unmatched_note": "No catalogue join is performed. Flow and data begin from aligned conditions; measured cuts are then applied independently to compare selected marginals.",
        "analysis_summaries": {
            key: {
                "differences": value["differences"],
                "divergence_decomposition": value["divergence_decomposition"],
            }
            for key, value in analyses.items()
        },
        "wall_seconds": float(time.time() - started),
    }
    (args.output / "summary.json").write_text(json.dumps(root_summary, indent=2) + "\n")
    print(json.dumps(root_summary, indent=2))


if __name__ == "__main__":
    main()
