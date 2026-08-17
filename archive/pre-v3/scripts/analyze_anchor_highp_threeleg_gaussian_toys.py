"""Analyze the controlled high-prediction three-leg Gaussian toy experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


KEY = ["template_id", "case", "input_index", "multiplicity", "nominal_snr"]
RESPONSE_COLUMNS = [
    "R_trace_coherent",
    "R_trace_context_sum",
    "R_trace_pair_sum",
]
CONTRAST_COLUMNS = [
    "delta_additivity_trace",
    "delta_context_trace",
    "delta_total_trace",
]
MEASURE_COLUMNS = RESPONSE_COLUMNS + CONTRAST_COLUMNS
LABELS = {
    "R_trace_coherent": "All neighbours coherent",
    "R_trace_context_sum": "One at a time; others present",
    "R_trace_pair_sum": "Isolated-pair sum",
    "delta_additivity_trace": r"$R_{\rm coh}-R_{\rm context}$",
    "delta_context_trace": r"$R_{\rm context}-R_{\rm pair}$",
    "delta_total_trace": r"$R_{\rm coh}-R_{\rm pair}$",
}
COLORS = {
    "R_trace_coherent": "#0072B2",
    "R_trace_context_sum": "#E69F00",
    "R_trace_pair_sum": "#009E73",
}


def load_experiment(
    input_dir: str,
    template_ids: list[int],
    nreal: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load all shards and enforce their expected matched-block coverage."""
    root = Path(input_dir)
    draw_parts = []
    pair_parts = []
    audits = []
    for template_id in template_ids:
        draw_path = root / f"draws_template{template_id:02d}.feather"
        pair_path = root / f"pairs_template{template_id:02d}.feather"
        audit_path = root / f"audit_template{template_id:02d}.json"
        for path in (draw_path, pair_path, audit_path):
            if not path.exists():
                raise FileNotFoundError(path)
        draws = pd.read_feather(draw_path)
        pairs = pd.read_feather(pair_path)
        with open(audit_path, encoding="utf-8") as handle:
            audit = json.load(handle)
        if draws.template_id.nunique() != 1 or int(draws.template_id.iloc[0]) != template_id:
            raise RuntimeError(f"template {template_id} draw identity mismatch")
        expected = len(audit["nominal_snr"]) * (int(nreal) + 1)
        if len(draws) != expected:
            raise RuntimeError(
                f"template {template_id}: {len(draws)} rows != expected {expected}"
            )
        if int(audit["nreal"]) != int(nreal):
            raise RuntimeError(f"template {template_id} nreal mismatch")
        draw_parts.append(draws)
        pair_parts.append(pairs)
        audits.append(audit)
    return (
        pd.concat(draw_parts, ignore_index=True),
        pd.concat(pair_parts, ignore_index=True),
        audits,
    )


def validate_decompositions(draws: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Validate the scene identity and both explicit pair sums."""
    good = draws.loc[draws.success.astype(bool)].copy()
    if good.empty:
        raise RuntimeError("experiment has no successful matched blocks")
    scene_replay = np.abs(
        good.delta_total_trace.to_numpy(float)
        - good.delta_additivity_trace.to_numpy(float)
        - good.delta_context_trace.to_numpy(float)
    )
    pair_sums = pairs.groupby(
        ["template_id", "nominal_snr", "realization", "noise_mode"],
        sort=False,
    ).agg(
        pair_context=("R_trace_context", "sum"),
        pair_only=("R_trace_pair_only", "sum"),
        n_pair_rows=("source_rank", "size"),
    ).reset_index()
    merged = good.merge(
        pair_sums,
        on=["template_id", "nominal_snr", "realization", "noise_mode"],
        how="left",
        validate="one_to_one",
    )
    if merged[["pair_context", "pair_only"]].isna().any().any():
        raise RuntimeError("successful draw lacks explicit pair rows")
    expected_pairs = merged.multiplicity.to_numpy(int)
    if not np.array_equal(merged.n_pair_rows.to_numpy(int), expected_pairs):
        raise RuntimeError("explicit pair-row count differs from multiplicity")
    context_error = np.abs(
        merged.R_trace_context_sum.to_numpy(float)
        - merged.pair_context.to_numpy(float)
    )
    pair_error = np.abs(
        merged.R_trace_pair_sum.to_numpy(float)
        - merged.pair_only.to_numpy(float)
    )
    metrics = {
        "max_scene_decomposition_replay_abs": float(scene_replay.max()),
        "max_context_pair_sum_replay_abs": float(context_error.max()),
        "max_isolated_pair_sum_replay_abs": float(pair_error.max()),
    }
    if max(metrics.values()) > 5.0e-11:
        raise RuntimeError(f"response decomposition failed: {metrics}")
    return metrics


def build_cell_means(draws: pd.DataFrame, min_success_fraction: float) -> pd.DataFrame:
    """Reduce technical noise draws to one independent value per template cell."""
    noisy = draws.loc[draws.noise_mode == "noisy"].copy()
    noiseless = draws.loc[
        (draws.noise_mode == "noiseless") & draws.success.astype(bool)
    ].copy()
    rows = []
    for key, local in noisy.groupby(KEY, sort=True):
        success = local.success.astype(bool)
        fraction = float(success.mean())
        if fraction < float(min_success_fraction):
            raise RuntimeError(f"cell {key} success fraction {fraction:.3f} is too low")
        good = local.loc[success]
        row = dict(zip(KEY, key))
        row.update({
            "n_requested": int(len(local)),
            "n_success": int(success.sum()),
            "success_fraction": fraction,
        })
        for column in MEASURE_COLUMNS:
            values = good[column].to_numpy(float)
            row[column] = float(values.mean())
            row[f"{column}_technical_sd"] = float(values.std(ddof=1))
            row[f"{column}_technical_sem"] = float(
                values.std(ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    cells = pd.DataFrame(rows).sort_values(KEY, kind="mergesort")
    noiseless_keep = noiseless[KEY + MEASURE_COLUMNS].rename(columns={
        column: f"{column}_noiseless" for column in MEASURE_COLUMNS
    })
    cells = cells.merge(noiseless_keep, on=KEY, how="left", validate="one_to_one")
    if cells[[f"{column}_noiseless" for column in MEASURE_COLUMNS]].isna().any().any():
        raise RuntimeError("a noisy cell lacks its successful noiseless control")
    replay = np.abs(
        cells.delta_total_trace.to_numpy(float)
        - cells.delta_additivity_trace.to_numpy(float)
        - cells.delta_context_trace.to_numpy(float)
    )
    if replay.max() > 5.0e-12:
        raise RuntimeError("cell-mean response identity failed")
    return cells


def vector_stat(
    values: np.ndarray,
    seed: int,
    n_boot: int = 20000,
) -> dict:
    """Estimate a geometry-level mean with parametric and bootstrap diagnostics."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        raise ValueError("geometry-level statistic requires at least three values")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = float(sd / np.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    rng = np.random.RandomState(int(seed))
    # Chunk the bootstrap to avoid a large temporary array.
    boot = np.empty(int(n_boot), dtype=float)
    chunk = 2000
    for start in range(0, int(n_boot), chunk):
        stop = min(start + chunk, int(n_boot))
        indices = rng.randint(0, n, size=(stop - start, n))
        boot[start:stop] = values[indices].mean(axis=1)
    boot_low, boot_high = np.percentile(boot, [2.5, 97.5])
    shapiro = stats.shapiro(values) if n <= 5000 else None
    ttest = stats.ttest_1samp(values, 0.0)
    try:
        wilcoxon = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    return {
        "n_templates": int(n),
        "mean": mean,
        "sd_between_templates": sd,
        "sem_between_templates": sem,
        "t_ci95_low": float(mean - critical * sem),
        "t_ci95_high": float(mean + critical * sem),
        "bootstrap_ci95_low": float(boot_low),
        "bootstrap_ci95_high": float(boot_high),
        "median": float(np.median(values)),
        "q16": float(np.percentile(values, 16)),
        "q84": float(np.percentile(values, 84)),
        "cohen_dz": float(mean / sd) if sd > 0 else 0.0,
        "one_sample_t": float(ttest.statistic),
        "one_sample_t_p": float(ttest.pvalue),
        "shapiro_w": float(shapiro.statistic) if shapiro is not None else None,
        "shapiro_p": float(shapiro.pvalue) if shapiro is not None else None,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
    }


def build_summary(cells: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Summarize each S/N x multiplicity cell plus equal-template all-K rows."""
    records = []
    payload: dict[str, dict] = {}
    counter = 0
    for snr in sorted(cells.nominal_snr.unique()):
        groups = [(f"N={int(k)}", local) for k, local in cells.loc[
            cells.nominal_snr == snr
        ].groupby("multiplicity", sort=True)]
        groups.append(("all", cells.loc[cells.nominal_snr == snr]))
        for group_name, local in groups:
            key = f"snr{snr:g}_{group_name.replace('=', '').replace(' ', '_')}"
            payload[key] = {}
            for column in MEASURE_COLUMNS:
                stat = vector_stat(
                    local[column].to_numpy(float),
                    seed + counter,
                    n_boot=n_boot,
                )
                counter += 1
                payload[key][column] = stat
                records.append({
                    "nominal_snr": float(snr),
                    "group": group_name,
                    "multiplicity": (
                        int(local.multiplicity.iloc[0])
                        if group_name != "all" else 0
                    ),
                    "quantity": column,
                    **stat,
                })
    return pd.DataFrame(records), payload


def summarize_gcheck(
    main_cells: pd.DataFrame,
    gcheck_cells: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Paired g=0.01 minus g=0.02 sensitivity on the frozen subset."""
    columns = ["template_id", "multiplicity", "nominal_snr", *MEASURE_COLUMNS]
    merged = gcheck_cells[columns].merge(
        main_cells[columns],
        on=["template_id", "multiplicity", "nominal_snr"],
        suffixes=("_g001", "_g002"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gcheck_cells):
        raise RuntimeError("g-check cells did not close one-to-one against main cells")
    records = []
    payload = {}
    counter = 0
    for snr, local in merged.groupby("nominal_snr", sort=True):
        key = f"snr{snr:g}"
        payload[key] = {}
        for column in MEASURE_COLUMNS:
            difference = (
                local[f"{column}_g001"].to_numpy(float)
                - local[f"{column}_g002"].to_numpy(float)
            )
            stat = vector_stat(difference, seed + counter, n_boot=n_boot)
            counter += 1
            payload[key][column] = stat
            records.append({
                "nominal_snr": float(snr),
                "quantity": column,
                "comparison": "g0.01 minus g0.02",
                **stat,
            })
    return pd.DataFrame(records), payload


def plot_main(cells: pd.DataFrame, output_png: str, output_pdf: str) -> None:
    """Make the response and three planned contrast panels."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    multiplicities = sorted(cells.multiplicity.unique())
    rng = np.random.RandomState(20260814)

    ax = axes[0, 0]
    mid_snr = 20.0
    local_mid = cells.loc[np.isclose(cells.nominal_snr, mid_snr)]
    for offset, column in zip((-0.13, 0.0, 0.13), RESPONSE_COLUMNS):
        color = COLORS[column]
        xs, means, sems = [], [], []
        for multiplicity in multiplicities:
            values = local_mid.loc[
                local_mid.multiplicity == multiplicity, column
            ].to_numpy(float)
            x = float(multiplicity + offset)
            xs.append(x)
            means.append(float(values.mean()))
            sems.append(float(values.std(ddof=1) / np.sqrt(len(values))))
            jitter = rng.uniform(-0.035, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=12,
                color=color,
                alpha=0.22,
                linewidths=0,
            )
        ax.errorbar(
            xs, means, yerr=sems, color=color, marker="o", ms=5,
            linewidth=1.5, capsize=3, label=LABELS[column],
        )
    ax.set_xlabel("Selected neighbour count")
    ax.set_ylabel(r"Response trace $\frac{1}{2}\,\mathrm{tr}(R)$")
    ax.set_title("Responses at nominal primary S/N = 20")
    ax.set_xticks(multiplicities)
    ax.legend(frameon=False, loc="best")

    snr_levels = sorted(cells.nominal_snr.unique())
    snr_colors = ["#0072B2", "#E69F00", "#009E73"]
    snr_markers = ["o", "s", "^"]
    for ax, column, title in zip(
        axes.flat[1:],
        CONTRAST_COLUMNS,
        (
            "Simultaneous-shear additivity",
            "Other-neighbour context",
            "Total isolated-pair discrepancy",
        ),
    ):
        for snr_index, (snr, color, marker) in enumerate(
            zip(snr_levels, snr_colors, snr_markers)
        ):
            offset = (snr_index - 1) * 0.13
            xs, means, sems = [], [], []
            for multiplicity in multiplicities:
                values = cells.loc[
                    np.isclose(cells.nominal_snr, snr)
                    & (cells.multiplicity == multiplicity),
                    column,
                ].to_numpy(float)
                x = float(multiplicity + offset)
                xs.append(x)
                means.append(float(values.mean()))
                sems.append(float(values.std(ddof=1) / np.sqrt(len(values))))
                jitter = rng.uniform(-0.03, 0.03, size=len(values))
                ax.scatter(
                    np.full(len(values), x) + jitter,
                    values,
                    s=10,
                    color=color,
                    alpha=0.18,
                    linewidths=0,
                )
            ax.errorbar(
                xs, means, yerr=sems, color=color, marker=marker, ms=5,
                linewidth=1.4, capsize=3, label=f"S/N = {snr:g}",
            )
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Selected neighbour count")
        ax.set_ylabel(LABELS[column])
        ax.set_title(title)
        ax.set_xticks(multiplicities)
        ax.legend(frameon=False, loc="best")

    for label, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.12, 1.04, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top",
        )
    fig.suptitle(
        "Controlled three-leg neighbour-shear toy\n"
        "Points are independent anchor geometries; error bars are one geometry SEM",
        fontsize=13,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_report(payload: dict, summary: pd.DataFrame) -> str:
    coverage = payload["coverage"]
    design = payload["design"]
    lines = [
        "# Controlled high-response-anchor three-leg Gaussian toy",
        "",
        "The primary is never sheared. The three measurements are: all selected "
        "neighbours coherently sheared; one neighbour at a time with every other "
        "selected neighbour still present; and one isolated primary-neighbour pair "
        "at a time. Profiles are round Gaussians, while catalogue flux ratios, "
        "circularized sizes, separations, and angles are retained.",
        "",
        f"- Templates: `{design['n_templates']}` independent catalogue cases; "
        f"multiplicities `{design['multiplicities']}`.",
        f"- Nominal primary S/N levels: `{design['nominal_snr']}`; "
        f"`{design['nreal']}` common-noise blocks plus one noiseless control per cell.",
        f"- Successful noisy cells: `{coverage['n_success_noisy']:,}` / "
        f"`{coverage['n_requested_noisy']:,}` "
        f"(`{coverage['success_fraction_noisy']:.2%}`).",
        "- Statistical unit: anchor geometry. Noise repetitions are technical "
        "replicates and are averaged before between-template uncertainty is formed.",
        "",
        "## Planned contrasts",
        "",
        "Positive values mean the response on the left is larger.",
        "",
        "| S/N | group | n | coherent − context | context − pair | coherent − pair |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for snr in sorted(summary.nominal_snr.unique()):
        local = summary.loc[
            (summary.nominal_snr == snr) & summary.quantity.isin(CONTRAST_COLUMNS)
        ]
        for group in [*map(lambda x: f"N={int(x)}", design["multiplicities"]), "all"]:
            row = local.loc[local.group == group].set_index("quantity")
            if row.empty:
                continue
            values = []
            for column in CONTRAST_COLUMNS:
                stat = row.loc[column]
                values.append(
                    f"{stat['mean']:+.5f} ± {stat['sem_between_templates']:.5f}"
                )
            lines.append(
                f"| {snr:g} | {group} | {int(row.iloc[0].n_templates)} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend([
        "",
        "Error terms in the table are one SEM across independent anchor geometries. "
        "The machine-readable summary also stores t intervals, template-bootstrap "
        "95% intervals, Shapiro diagnostics, paired one-sample t tests, Wilcoxon "
        "sensitivity tests, and standardized paired effects.",
        "",
    ])
    if payload.get("finite_shear_check"):
        lines.extend([
            "## Finite-shear check",
            "",
            "The frozen 12-template subset was repeated at g=0.01 with identical "
            "noise and fit seeds. The JSON and sensitivity CSV report g=0.01 minus "
            "g=0.02 for every response and contrast.",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--nreal", type=int, default=200)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--gcheck-dir")
    parser.add_argument(
        "--gcheck-template-ids", nargs="+", type=int,
        default=[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35],
    )
    parser.add_argument("--gcheck-g", type=float, default=0.01)
    parser.add_argument("--min-success-fraction", type=float, default=0.8)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--output-cells", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-sensitivity", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()
    outputs = (
        args.output_cells, args.output_pairs, args.output_summary,
        args.output_sensitivity, args.output_json, args.output_md,
        args.output_png, args.output_pdf,
    )
    for output in outputs:
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    templates = pd.read_feather(args.templates).sort_values("template_id")
    template_ids = templates.template_id.astype(int).tolist()
    draws, pairs, audits = load_experiment(args.input_dir, template_ids, args.nreal)
    if not np.allclose([float(audit["g"]) for audit in audits], args.g):
        raise RuntimeError("main shard shear amplitude differs from analysis g")
    replay = validate_decompositions(draws, pairs)
    cells = build_cell_means(draws, args.min_success_fraction)
    expected_cells = len(templates) * len(audits[0]["nominal_snr"])
    if len(cells) != expected_cells:
        raise RuntimeError(f"{len(cells)} cells != expected {expected_cells}")
    if cells.template_id.nunique() != len(templates):
        raise RuntimeError("cell means do not cover every template")
    summary, summary_payload = build_summary(
        cells, args.n_bootstrap, args.bootstrap_seed
    )

    sensitivity = pd.DataFrame()
    sensitivity_payload = None
    gcheck_coverage = None
    if args.gcheck_dir:
        gdraws, gpairs, gaudits = load_experiment(
            args.gcheck_dir, args.gcheck_template_ids, args.nreal
        )
        if not np.allclose([float(audit["g"]) for audit in gaudits], args.gcheck_g):
            raise RuntimeError("g-check shard amplitude differs from requested value")
        greplay = validate_decompositions(gdraws, gpairs)
        gcells = build_cell_means(gdraws, args.min_success_fraction)
        sensitivity, sensitivity_payload = summarize_gcheck(
            cells, gcells, args.n_bootstrap, args.bootstrap_seed + 1_000_000
        )
        gcheck_coverage = {
            "n_templates": int(gcells.template_id.nunique()),
            "n_cells": int(len(gcells)),
            "n_requested_noisy": int((gdraws.noise_mode == "noisy").sum()),
            "n_success_noisy": int((
                (gdraws.noise_mode == "noisy") & gdraws.success.astype(bool)
            ).sum()),
            "replay": greplay,
        }

    noisy_mask = draws.noise_mode == "noisy"
    n_requested_noisy = int(noisy_mask.sum())
    n_success_noisy = int((noisy_mask & draws.success.astype(bool)).sum())
    payload = {
        "design": {
            "description": (
                "catalogue-informed round-Gaussian three-leg neighbour-shear toy"
            ),
            "n_templates": int(len(templates)),
            "all_cases_distinct": bool(~templates.case.duplicated().any()),
            "multiplicities": sorted(templates.multiplicity.astype(int).unique().tolist()),
            "nominal_snr": sorted(cells.nominal_snr.astype(float).unique().tolist()),
            "nreal": int(args.nreal),
            "g": float(args.g),
            "statistical_unit": "independent anchor template / catalogue case",
            "technical_unit": "matched common-noise realization",
            "error_bars": "one SEM across template cell means",
            "confidence_intervals": "95% template bootstrap and t intervals",
        },
        "coverage": {
            "n_draw_rows": int(len(draws)),
            "n_pair_rows": int(len(pairs)),
            "n_requested_noisy": n_requested_noisy,
            "n_success_noisy": n_success_noisy,
            "success_fraction_noisy": float(n_success_noisy / n_requested_noisy),
            "n_success_noiseless": int((
                (draws.noise_mode == "noiseless") & draws.success.astype(bool)
            ).sum()),
            "replay": replay,
        },
        "summary": summary_payload,
        "finite_shear_check": (
            {
                "gcheck_g": float(args.gcheck_g),
                "main_g": float(args.g),
                "template_ids": list(map(int, args.gcheck_template_ids)),
                "coverage": gcheck_coverage,
                "summary_gcheck_minus_main": sensitivity_payload,
            }
            if sensitivity_payload is not None else None
        ),
        "inputs": {
            "templates": os.path.abspath(args.templates),
            "input_dir": os.path.abspath(args.input_dir),
            "gcheck_dir": os.path.abspath(args.gcheck_dir) if args.gcheck_dir else None,
        },
        "artifacts": {
            "cells": os.path.abspath(args.output_cells),
            "pairs": os.path.abspath(args.output_pairs),
            "summary": os.path.abspath(args.output_summary),
            "sensitivity": os.path.abspath(args.output_sensitivity),
            "figure_png": os.path.abspath(args.output_png),
            "figure_pdf": os.path.abspath(args.output_pdf),
        },
    }

    Path(args.output_cells).parent.mkdir(parents=True, exist_ok=True)
    cells.to_feather(args.output_cells)
    # Pair rows remain technical; store their per-template/S/N means for reuse.
    pair_success = pairs.loc[pairs.noise_mode == "noisy"].groupby(
        [
            "template_id", "case", "input_index", "multiplicity",
            "nominal_snr", "source_rank", "secondary_index",
            "distance_arcsec", "R_model_pair",
        ],
        sort=True,
    )[["R_trace_context", "R_trace_pair_only", "delta_context_pair_trace"]].mean().reset_index()
    pair_success.to_feather(args.output_pairs)
    summary.to_csv(args.output_summary, index=False)
    sensitivity.to_csv(args.output_sensitivity, index=False)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown_report(payload, summary))
    plot_main(cells, args.output_png, args.output_pdf)
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))
    print("ANCHOR_HIGHP_THREELEG_GAUSSIAN_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
