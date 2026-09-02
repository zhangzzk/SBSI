#!/usr/bin/env python
"""Compare selected likelihood-mock measurements with ConstGold measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


CONSTGOLD_COLUMNS = [
    "case",
    "input_index",
    "neighbored",
    "distance",
    "r_input_p",
    "Re_input_p",
    "applied_g1",
    "applied_g2",
    "measured_e1_plus",
    "measured_e2_plus",
    "measured_mag_auto_plus",
    "measured_flux_radius_plus",
]
PROPERTY_NAMES = ["g1", "g2", "mag", "radius"]
PROPERTY_LABELS = [
    r"Measured $\hat g_1$",
    r"Measured $\hat g_2$",
    "Measured MAG_AUTO (mag)",
    "Measured flux radius (arcsec)",
]
MOCK_COLOR = "#0072B2"
CONSTGOLD_COLOR = "#D55E00"
CONTOUR_MASSES = (0.95, 0.80, 0.50)
CONTOUR_STYLES = (":", "--", "-")
CONTOUR_WIDTHS = (0.9, 1.2, 1.7)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--mock-manifest", type=Path, required=True)
    parser.add_argument("--constgold", type=Path, required=True)
    parser.add_argument(
        "--constgold-sample-cache",
        type=Path,
        help="reuse a previously frozen selected ConstGold sample instead of rescanning",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--shape-max", type=float, default=0.6)
    parser.add_argument("--mag-max", type=float, default=25.8)
    parser.add_argument("--radius-min", type=float, default=0.75)
    parser.add_argument(
        "--analysis-mag-max",
        type=float,
        help="after fixing both samples, retain only measurements below this magnitude",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="suffix added to figure filenames, for example _mlt22",
    )
    return parser.parse_args(argv)


def _and(*masks):
    result = masks[0]
    for mask in masks[1:]:
        result = pc.and_(result, mask)
    return pc.fill_null(result, False)


def _count(mask):
    return int(pc.sum(pc.cast(mask, "int64")).as_py())


def load_constgold(args):
    table = feather.read_table(
        args.constgold,
        columns=CONSTGOLD_COLUMNS,
        memory_map=True,
        use_threads=True,
    )
    case_mask = pc.greater_equal(table["case"], args.min_case)
    broad_mask = _and(
        case_mask,
        pc.greater(table["r_input_p"], 18.0),
        pc.less(table["r_input_p"], 28.0),
        pc.greater(table["Re_input_p"], 0.1),
        pc.less(table["Re_input_p"], 1.5),
        pc.or_(
            _and(
                pc.greater(table["distance"], 0.0),
                pc.less(table["distance"], 5.0),
            ),
            pc.invert(table["neighbored"]),
        ),
    )
    domain_mask = _and(
        broad_mask,
        pc.less(table["r_input_p"], 25.8),
        pc.greater(table["Re_input_p"], 0.5),
    )
    counts = {
        "catalogue_rows": int(table.num_rows),
        "case_ge_40": _count(case_mask),
        "after_source_selection": _count(broad_mask),
        "after_v3_2_true_domain": _count(domain_mask),
    }
    frame = table.filter(domain_mask).select(
        [
            "case",
            "input_index",
            "applied_g1",
            "applied_g2",
            "measured_e1_plus",
            "measured_e2_plus",
            "measured_mag_auto_plus",
            "measured_flux_radius_plus",
        ]
    ).to_pandas()
    values = pd.DataFrame(
        {
            "g1": frame["measured_e1_plus"].to_numpy(float),
            "g2": frame["measured_e2_plus"].to_numpy(float),
            "mag": frame["measured_mag_auto_plus"].to_numpy(float),
            "radius": args.pixel_size
            * frame["measured_flux_radius_plus"].to_numpy(float),
        }
    )
    applied = frame[["applied_g1", "applied_g2"]].to_numpy(float)
    if not np.allclose(applied, (0.02, 0.0), rtol=0.0, atol=1.0e-12):
        raise ValueError("ConstGold plus leg is not uniformly at g=(0.02, 0)")
    array = values.to_numpy(float)
    finite = np.isfinite(array).all(axis=1)
    selected = (
        finite
        & (np.hypot(array[:, 0], array[:, 1]) < args.shape_max)
        & (array[:, 2] < args.mag_max)
        & (array[:, 3] >= args.radius_min)
    )
    counts["finite_plus_measurements"] = int(finite.sum())
    counts["after_measured_selection"] = int(selected.sum())
    selected_values = values.loc[selected].reset_index(drop=True)
    if len(selected_values) < args.sample_size:
        raise ValueError(
            f"only {len(selected_values):,} ConstGold rows pass; "
            f"cannot draw {args.sample_size:,}"
        )
    rng = np.random.default_rng(args.seed)
    index = rng.choice(len(selected_values), size=args.sample_size, replace=False)
    selected_values = selected_values.iloc[index].reset_index(drop=True)
    selected_identity = frame.loc[selected, ["case", "input_index"]].reset_index(drop=True)
    selected_identity = selected_identity.iloc[index].reset_index(drop=True)
    return selected_values, selected_identity, counts


def load_mock(args):
    frame = pd.read_parquet(args.mock)
    required = {
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"mock lacks columns: {missing}")
    values = pd.DataFrame(
        {
            "g1": frame["measured_ngmix_g1"].to_numpy(float),
            "g2": frame["measured_ngmix_g2"].to_numpy(float),
            "mag": frame["measured_mag_auto"].to_numpy(float),
            "radius": args.pixel_size
            * np.exp(frame["measured_log_flux_radius"].to_numpy(float)),
        }
    )
    array = values.to_numpy(float)
    selected = (
        np.isfinite(array).all(axis=1)
        & (np.hypot(array[:, 0], array[:, 1]) < args.shape_max)
        & (array[:, 2] < args.mag_max)
        & (array[:, 3] >= args.radius_min)
    )
    if not selected.all():
        raise ValueError(
            f"mock contains {int((~selected).sum())} rows outside its recorded selection"
        )
    if len(values) != args.sample_size:
        raise ValueError(
            f"mock has {len(values):,} rows; expected exactly {args.sample_size:,}"
        )
    manifest = json.loads(args.mock_manifest.read_text())
    injected = (float(manifest["injected_g1"]), float(manifest["injected_g2"]))
    if not np.allclose(injected, (0.02, 0.0), rtol=0.0, atol=1.0e-12):
        raise ValueError(f"mock injected shear {injected} does not match ConstGold plus")
    return values, manifest


def load_constgold_sample_cache(args):
    frame = pd.read_parquet(args.constgold_sample_cache)
    required = {
        "case",
        "input_index",
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"ConstGold sample cache lacks columns: {missing}")
    if len(frame) != args.sample_size:
        raise ValueError(
            f"ConstGold sample cache has {len(frame):,} rows; "
            f"expected {args.sample_size:,}"
        )
    values = pd.DataFrame(
        {
            "g1": frame["measured_ngmix_g1"].to_numpy(float),
            "g2": frame["measured_ngmix_g2"].to_numpy(float),
            "mag": frame["measured_mag_auto"].to_numpy(float),
            "radius": args.pixel_size
            * np.exp(frame["measured_log_flux_radius"].to_numpy(float)),
        }
    )
    identity = frame[["case", "input_index"]].copy()
    parent_summary = args.constgold_sample_cache.parent / "summary.json"
    counts = {"selected_sample_cache_rows": int(len(frame))}
    if parent_summary.is_file():
        parent = json.loads(parent_summary.read_text())
        counts.update(parent.get("population", {}).get("constgold_counts", {}))
    return values, identity, counts


def _axis_limits(mock, constgold, mag_hi=25.8):
    joined = pd.concat([mock, constgold], ignore_index=True)
    radius_hi = float(np.quantile(joined["radius"], 0.997))
    mag_lo = float(np.quantile(joined["mag"], 0.001))
    return [
        (-0.6, 0.6),
        (-0.6, 0.6),
        (mag_lo, mag_hi),
        (0.75, radius_hi),
    ]


def _hdr_threshold(histogram, mass):
    density = np.asarray(histogram, dtype=float)
    total = density.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empty histogram cannot define an HDR contour")
    ordered = np.sort(density.ravel())[::-1]
    index = min(int(np.searchsorted(np.cumsum(ordered), mass * total)), len(ordered) - 1)
    return float(ordered[index])


def _draw_corner(
    mock,
    constgold,
    output,
    *,
    output_tag="",
    mag_hi=25.8,
    title=None,
    limits=None,
):
    datasets = [
        (mock, "Complete-likelihood mock", MOCK_COLOR),
        (constgold, "ConstGold +g measurements", CONSTGOLD_COLOR),
    ]
    limits = _axis_limits(mock, constgold, mag_hi=mag_hi) if limits is None else limits
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
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
                    density, _ = np.histogram(
                        frame[PROPERTY_NAMES[column]], bins=edges, density=True
                    )
                    density = gaussian_filter1d(density, sigma=1.2)
                    ax.plot(centers, density, color=color, linewidth=1.5)
                ax.set_xlim(limits[column])
                ax.set_yticks([])
            else:
                xedges = np.linspace(*limits[column], bins + 1)
                yedges = np.linspace(*limits[row], bins + 1)
                xcenters = 0.5 * (xedges[:-1] + xedges[1:])
                ycenters = 0.5 * (yedges[:-1] + yedges[1:])
                for frame, _, color in datasets:
                    histogram, _, _ = np.histogram2d(
                        frame[PROPERTY_NAMES[column]],
                        frame[PROPERTY_NAMES[row]],
                        bins=(xedges, yedges),
                    )
                    density = gaussian_filter(histogram.T, sigma=2.0)
                    for mass, style, width in zip(
                        CONTOUR_MASSES, CONTOUR_STYLES, CONTOUR_WIDTHS
                    ):
                        threshold = _hdr_threshold(density, mass)
                        if density.min() < threshold < density.max():
                            ax.contour(
                                xcenters,
                                ycenters,
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

    legend_ax = axes[0, 2]
    legend_ax.axis("off")
    handles = [
        mlines.Line2D([], [], color=color, linewidth=2.0, label=label)
        for _, label, color in datasets
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=9)
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.5,
        0.5,
        "Contours enclose\n50% (solid), 80% (dashed),\n95% (dotted)",
        ha="center",
        va="center",
        fontsize=8,
        linespacing=1.5,
    )
    fig.suptitle(
        title
        or "Selected measurement distributions at $g=(0.02, 0)$ — 100,000 each\n"
        "Central view (radius shown through the pooled 99.7th percentile)",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.10, bottom=0.08, right=0.98, top=0.90, wspace=0.12, hspace=0.10
    )
    fig.savefig(output / f"mock_vs_constgold_properties_100k{output_tag}.png", dpi=300)
    fig.savefig(output / f"mock_vs_constgold_properties_100k{output_tag}.pdf")
    plt.close(fig)


def _draw_radius_tail(mock, constgold, output, *, output_tag="", title=None):
    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    for frame, label, color in (
        (mock, "Complete-likelihood mock", MOCK_COLOR),
        (constgold, "ConstGold +g measurements", CONSTGOLD_COLOR),
    ):
        radius = np.sort(frame["radius"].to_numpy(float))
        survival = (len(radius) - np.arange(len(radius))) / len(radius)
        ax.step(radius, survival, where="post", color=color, linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.75, None)
    ax.set_ylim(0.5 / len(mock), 1.0)
    ax.set_xlabel("Measured flux radius (arcsec)")
    ax.set_ylabel(r"Survival fraction $P(R_\mathrm{flux}>R)$")
    ax.set_title(title or "Measured-size tail — 100,000 selected measurements each")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output / f"mock_vs_constgold_radius_tail_100k{output_tag}.png", dpi=300)
    fig.savefig(output / f"mock_vs_constgold_radius_tail_100k{output_tag}.pdf")
    plt.close(fig)


def _distribution_summary(values):
    result = {}
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999, 0.9999)
    for name in PROPERTY_NAMES:
        x = values[name].to_numpy(float)
        result[name] = {
            "mean": float(x.mean()),
            "standard_deviation": float(x.std(ddof=1)),
            "quantiles": {
                str(q): float(value)
                for q, value in zip(quantiles, np.quantile(x, quantiles))
            },
            "minimum": float(x.min()),
            "maximum": float(x.max()),
        }
    result["correlation"] = values[PROPERTY_NAMES].corr().to_numpy().tolist()
    result["spearman_correlation"] = spearmanr(
        values[PROPERTY_NAMES].to_numpy(float), axis=0
    ).statistic.tolist()
    radius = values["radius"].to_numpy(float)
    result["radius_tail_fractions"] = {
        f"above_{threshold:g}_arcsec": float(np.mean(radius > threshold))
        for threshold in (2.0, 3.0, 5.0, 10.0, 100.0)
    }
    return result


def _comparison_summary(mock, constgold):
    result = {}
    for name in PROPERTY_NAMES:
        x = mock[name].to_numpy(float)
        y = constgold[name].to_numpy(float)
        pooled_sd = np.sqrt(0.5 * (x.var(ddof=1) + y.var(ddof=1)))
        ks = ks_2samp(x, y, method="asymp")
        distance = wasserstein_distance(x, y)
        result[name] = {
            "mock_minus_constgold_mean": float(x.mean() - y.mean()),
            "standardized_mean_difference": float((x.mean() - y.mean()) / pooled_sd),
            "ks_statistic": float(ks.statistic),
            "wasserstein_distance": float(distance),
            "wasserstein_over_pooled_sd": float(distance / pooled_sd),
            "mean_difference_standard_error": float(
                np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            ),
        }
    mock_corr = mock[PROPERTY_NAMES].corr().to_numpy()
    constgold_corr = constgold[PROPERTY_NAMES].corr().to_numpy()
    result["correlation_difference_frobenius"] = float(
        np.linalg.norm(mock_corr - constgold_corr)
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    mock, manifest = load_mock(args)
    if args.constgold_sample_cache is None:
        constgold, constgold_identity, constgold_counts = load_constgold(args)
    else:
        constgold, constgold_identity, constgold_counts = load_constgold_sample_cache(args)
    parent_counts = {"mock": int(len(mock)), "constgold": int(len(constgold))}
    analysis_selection = None
    if args.analysis_mag_max is not None:
        analysis_selection = f"measured MAG_AUTO<{args.analysis_mag_max:g}"
        mock = mock.loc[mock["mag"] < args.analysis_mag_max].reset_index(drop=True)
        constgold_mask = constgold["mag"].to_numpy(float) < args.analysis_mag_max
        constgold = constgold.loc[constgold_mask].reset_index(drop=True)
        constgold_identity = constgold_identity.loc[constgold_mask].reset_index(drop=True)
        if len(mock) == 0 or len(constgold) == 0:
            raise ValueError("analysis magnitude filter emptied one comparison sample")
    args.output.mkdir(parents=True)
    if args.analysis_mag_max is None:
        mag_hi = args.mag_max
        corner_title = None
        tail_title = None
    else:
        mag_hi = args.analysis_mag_max
        corner_title = (
            rf"Bright measurement distributions at $g=(0.02, 0)$, $m<{mag_hi:g}$"
            "\n"
            f"Fixed 100k parents: mock n={len(mock):,}; ConstGold n={len(constgold):,}"
        )
        tail_title = (
            rf"Measured-size tail for $m<{mag_hi:g}$ — "
            f"mock n={len(mock):,}; ConstGold n={len(constgold):,}"
        )
    _draw_corner(
        mock,
        constgold,
        args.output,
        output_tag=args.output_tag,
        mag_hi=mag_hi,
        title=corner_title,
    )
    _draw_radius_tail(
        mock,
        constgold,
        args.output,
        output_tag=args.output_tag,
        title=tail_title,
    )
    constgold_sample = constgold_identity.copy()
    constgold_sample["measured_ngmix_g1"] = constgold["g1"]
    constgold_sample["measured_ngmix_g2"] = constgold["g2"]
    constgold_sample["measured_mag_auto"] = constgold["mag"]
    constgold_sample["measured_log_flux_radius"] = np.log(
        constgold["radius"] / args.pixel_size
    )
    constgold_sample.to_parquet(
        args.output / f"constgold_plus_selected_sample_100k{args.output_tag}.parquet",
        index=False,
    )
    summary = {
        "comparison": "selected complete-likelihood mock vs selected ConstGold plus leg",
        "shear": [0.02, 0.0],
        "sample_size_each": int(args.sample_size),
        "parent_sample_size_each": int(args.sample_size),
        "analysis_sample_sizes": {"mock": int(len(mock)), "constgold": int(len(constgold))},
        "constgold_sample_seed": int(args.seed),
        "inputs": {
            "mock": str(args.mock),
            "mock_manifest": str(args.mock_manifest),
            "constgold": str(args.constgold),
            "constgold_sample_cache": (
                None
                if args.constgold_sample_cache is None
                else str(args.constgold_sample_cache)
            ),
            "mock_measurements_sha256": manifest["output_sha256"]["measurements.parquet"],
        },
        "population": {
            "constgold_case_range": [int(args.min_case), 140],
            "source_selection": (
                "18<r_input_p<28; 0.1<Re_input_p<1.5; "
                "(0<distance<5 or not neighbored)"
            ),
            "registered_true_domain": "18<r_input_p<25.8; 0.5<Re_input_p<1.5",
            "measured_selection": (
                f"hypot(g1,g2)<{args.shape_max}; MAG_AUTO<{args.mag_max}; "
                f"flux_radius>={args.radius_min} arcsec"
            ),
            "analysis_selection": analysis_selection,
            "parent_sample_counts": parent_counts,
            "constgold_counts": constgold_counts,
            "mock_rows_passing_recorded_selection": int(len(mock)),
            "unmatched_rows": 0,
            "unmatched_note": "No catalogue join is used for this measurement-only comparison.",
        },
        "plot": {
            "properties": PROPERTY_NAMES,
            "contour_enclosed_masses": list(CONTOUR_MASSES),
            "axis_limits": _axis_limits(mock, constgold, mag_hi=mag_hi),
            "radius_conversion": (
                "ConstGold: 0.2*measured_flux_radius_plus; "
                "mock: 0.2*exp(measured_log_flux_radius)"
            ),
        },
        "mock": _distribution_summary(mock),
        "constgold": _distribution_summary(constgold),
        "differences": _comparison_summary(mock, constgold),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
