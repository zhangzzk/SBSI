"""Compare physical neighbour-flux/separation proxies with anchor V2.2 P_s.

The scene population and model coordinate are frozen: coherent anchors c400--899
and the exact deployed renderer-input V2.2 pair manifest.  For each anchor this
script forms

    Q_abs = sum_j F_s,j / d_j^2
    Q_rel = sum_j (F_s,j / F_p) / d_j^2,

where d is in arcsec and F uses the simulation r-band zero point of 30.  It
checks that the sum of stored pair responses replays the scene prediction in
the existing anchor feature table, then writes a compact table, a density plot,
and correlation statistics.  No response truth is read or used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ZERO_POINT = 30.0
Y_SCALE = 0.01
BLOCK_SIZE = 100


def refuse(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")


def base_for_case(root: Path, case: int) -> Path:
    start = (case // BLOCK_SIZE) * BLOCK_SIZE
    stop = start + BLOCK_SIZE - 1
    return root / f"lsst_sims_fs2_25876_anchorblend_g002_c{start}-{stop}"


def input_path(base: Path, case: int, shear: float) -> Path:
    return (
        base / f"case{case}_{str(float(shear))}" / "real0" / "catalogues"
        / "input" / "gals_info_tile180.0_-0.5.feather"
    )


def summarize_case(case: int, base_root: Path, pair_prefix: str,
                   shear: float, expected: pd.DataFrame) -> pd.DataFrame:
    base = base_for_case(base_root, case)
    pair_path = base / f"{pair_prefix}_case{case}.feather"
    catalogue_path = input_path(base, case, shear)
    if not pair_path.exists() or not catalogue_path.exists():
        raise FileNotFoundError(
            f"case {case}: pair={pair_path.exists()} catalogue={catalogue_path.exists()}"
        )
    pairs = pd.read_feather(
        pair_path,
        columns=["anchor_index", "secondary_index", "distance", "response"],
    )
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError(f"case {case}: duplicate deployed pair")
    distance = pairs.distance.to_numpy(float)
    if not np.isfinite(distance).all() or np.any(distance <= 0.0):
        raise RuntimeError(f"case {case}: invalid pair distance")

    catalogue = pd.read_feather(
        catalogue_path, columns=["index_input", "r_input"]
    ).rename(columns={"index_input": "index"})
    if catalogue["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate input index")
    magnitude = catalogue.set_index("index")["r_input"]
    secondary_mag = magnitude.reindex(pairs.secondary_index).to_numpy(float)
    primary_mag = magnitude.reindex(pairs.anchor_index).to_numpy(float)
    if not np.isfinite(secondary_mag).all() or not np.isfinite(primary_mag).all():
        raise RuntimeError(f"case {case}: pair ID missing from rendered catalogue")

    inv_d2 = np.reciprocal(np.square(distance))
    flux_secondary = np.power(10.0, -0.4 * (secondary_mag - ZERO_POINT))
    flux_ratio = np.power(10.0, -0.4 * (secondary_mag - primary_mag))
    work = pd.DataFrame({
        "input_index": pairs.anchor_index.to_numpy(np.int64),
        "q_abs_term": flux_secondary * inv_d2,
        "q_rel_term": flux_ratio * inv_d2,
        "pair_response": pairs.response.to_numpy(float),
    })
    scene = work.groupby("input_index", sort=False).agg(
        flux_over_d2=("q_abs_term", "sum"),
        flux_ratio_over_d2=("q_rel_term", "sum"),
        pair_response_sum=("pair_response", "sum"),
        n_pairs=("pair_response", "size"),
    ).reset_index()
    scene["case"] = case
    merged = expected.merge(
        scene, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    no_pair = merged.flux_over_d2.isna()
    if no_pair.any():
        merged.loc[no_pair, [
            "flux_over_d2", "flux_ratio_over_d2", "pair_response_sum", "n_pairs",
        ]] = 0.0
    merged["n_pairs"] = merged.n_pairs.astype(np.int64)
    replay = np.max(np.abs(
        merged.scene_prediction.to_numpy(float)
        - merged.pair_response_sum.to_numpy(float)
    ))
    if replay > 5e-6:
        raise RuntimeError(f"case {case}: V2.2 replay mismatch {replay:.3e}")
    print(
        f"case {case}: anchors={len(merged):,} pairs={len(pairs):,} "
        f"zero_pair={int(no_pair.sum())} replay={replay:.2e}", flush=True,
    )
    return merged


def transformed_response(value: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(value, float) / Y_SCALE)


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def case_correlation(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = []
    for _, part in frame.groupby("case", sort=True):
        values.append(stats.spearmanr(
            part[column].to_numpy(float),
            part.scene_prediction.to_numpy(float),
        ).statistic)
    values = np.asarray(values, float)
    return {
        "mean_spearman_rho": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 40):
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for bin_index in range(len(edges) - 1):
        keep = index == bin_index
        if not keep.any():
            continue
        rows.append((
            float(np.median(x[keep])),
            float(np.median(y[keep])),
            float(np.quantile(y[keep], 0.16)),
            float(np.quantile(y[keep], 0.84)),
            int(keep.sum()),
        ))
    return np.asarray(rows, float)


def plot_panel(ax, frame: pd.DataFrame, column: str, label: str,
               statistics: dict[str, float]):
    x = frame[column].to_numpy(float)
    y_raw = frame.scene_prediction.to_numpy(float)
    y = transformed_response(y_raw)
    density = ax.hexbin(
        x, y, gridsize=(90, 80), bins="log", mincnt=1,
        cmap="viridis", linewidths=0.0,
    )
    curve = binned_curve(x, y_raw)
    ax.fill_between(
        curve[:, 0], transformed_response(curve[:, 2]),
        transformed_response(curve[:, 3]), color="#E69F00", alpha=0.22,
        linewidth=0.0, label="16--84%",
    )
    ax.plot(
        curve[:, 0], transformed_response(curve[:, 1]), color="#D55E00",
        linewidth=2.0, label="Median in 40 equal-count bins",
    )
    ticks = np.asarray([-0.3, -0.1, -0.03, -0.01, 0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticks(transformed_response(ticks))
    ax.set_yticklabels([f"{value:g}" for value in ticks])
    ax.set_xlabel(label)
    ax.text(
        0.03, 0.97,
        f"Spearman $\\rho$ = {statistics['spearman_rho']:+.3f}\n"
        f"Pearson $r$ = {statistics['pearson_r']:+.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
    )
    ax.spines[["top", "right"]].set_visible(False)
    return density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--figure-output", required=True)
    parser.add_argument("--pdf-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    table_output = Path(args.table_output)
    figure_output = Path(args.figure_output)
    pdf_output = Path(args.pdf_output)
    summary_output = Path(args.summary_output)
    refuse([table_output, figure_output, pdf_output, summary_output])
    table_output.parent.mkdir(parents=True, exist_ok=True)
    figure_output.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_feather(
        args.features,
        columns=["case", "input_index", "scene_prediction", "primary_mag"],
    )
    features = features[
        features.case.between(args.case_min, args.case_max)
    ].copy()
    if features.duplicated(["case", "input_index"]).any():
        raise RuntimeError("anchor feature keys are not unique")
    cases = np.arange(args.case_min, args.case_max + 1)
    if not np.array_equal(np.sort(features.case.unique()), cases):
        raise RuntimeError("anchor feature table does not cover requested cases")

    parts = []
    base_root = Path(args.base_root)
    for case in cases:
        part = features[features.case == case].copy()
        parts.append(summarize_case(
            int(case), base_root, args.pair_prefix, args.shear, part
        ))
    frame = pd.concat(parts, ignore_index=True)
    proxy_floors = {}
    for source, target in (
        ("flux_over_d2", "log10_flux_over_d2"),
        ("flux_ratio_over_d2", "log10_flux_ratio_over_d2"),
    ):
        value = frame[source].to_numpy(float)
        positive = value[value > 0.0]
        if not len(positive):
            raise RuntimeError(f"physical proxy {source} has no positive values")
        floor = float(0.5 * positive.min())
        proxy_floors[source] = floor
        frame[target] = np.log10(np.maximum(value, floor))
    keep_columns = [
        "case", "input_index", "primary_mag", "scene_prediction", "n_pairs",
        "flux_over_d2", "flux_ratio_over_d2", "log10_flux_over_d2",
        "log10_flux_ratio_over_d2",
    ]
    frame[keep_columns].to_feather(table_output)

    columns = {
        "log10_flux_over_d2": (
            r"$\log_{10}\!\left[\sum_j F_{s,j}/d_j^2\right]$ "
            r"(r-band flux arcsec$^{-2}$)"
        ),
        "log10_flux_ratio_over_d2": (
            r"$\log_{10}\!\left[\sum_j (F_{s,j}/F_p)/d_j^2\right]$ "
            r"(arcsec$^{-2}$)"
        ),
    }
    summary = {
        "dataset": f"coherent_anchor_c{args.case_min}-{args.case_max}",
        "n_anchors": int(len(frame)),
        "n_cases": int(len(cases)),
        "scene_prediction": "sum of exact deployed renderer-input V2.2 pair responses",
        "distance_unit": "arcsec",
        "flux_zero_point": ZERO_POINT,
        "n_zero_neighbour_anchors": int((frame.n_pairs == 0).sum()),
        "zero_proxy_plot_floors": proxy_floors,
        "proxies": {},
    }
    for column in columns:
        summary["proxies"][column] = {
            "global_signed_scene_prediction": correlation(
                frame[column].to_numpy(float),
                frame.scene_prediction.to_numpy(float),
            ),
            "global_absolute_scene_prediction": correlation(
                frame[column].to_numpy(float),
                np.abs(frame.scene_prediction.to_numpy(float)),
            ),
            "per_case_signed_scene_prediction": case_correlation(frame, column),
        }

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    density = None
    for index, (ax, (column, label)) in enumerate(zip(axes, columns.items())):
        stats_signed = summary["proxies"][column]["global_signed_scene_prediction"]
        density = plot_panel(ax, frame, column, label, stats_signed)
        ax.set_title("Absolute secondary flux" if index == 0 else "Primary-normalized flux")
        ax.text(-0.10, 1.04, chr(ord("A") + index), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
    axes[0].set_ylabel(r"Frozen full-neighbour V2.2 scene prediction $P_s$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    colorbar = fig.colorbar(density, ax=axes, pad=0.015, fraction=0.035)
    colorbar.set_label("logarithmic anchor count per hexagon")
    fig.suptitle(
        "Physical neighbour-flux / separation proxies versus coherent-anchor scene prediction\n"
        f"Exact same {len(frame):,} anchors from cases {args.case_min}--{args.case_max}; "
        "response axis shown with an asinh stretch",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.15, top=0.82, wspace=0.12)
    fig.savefig(figure_output, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    with summary_output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_SCENE_FLUX_DISTANCE_PROXY_DONE", flush=True)


if __name__ == "__main__":
    main()
