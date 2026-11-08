#!/usr/bin/env python
"""Plot V3.2/V3.3 conditional shape likelihoods for 36 ConstGold galaxies."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from sbsi import sample_measurement
from sbsi.scene_prior import ScenePrior
from scripts.evaluate_constgold_complete_forward_response import (
    CONDITIONS,
    CROWD_COLUMNS,
    _raw_catalogue_root,
    _read_truth,
)


STRATA = (
    ("bright", "Bright", lambda mag: mag < 22.0, r"$\hat m<22$"),
    (
        "typical",
        "Typical",
        lambda mag: (mag >= 23.5) & (mag < 24.3),
        r"$23.5\leq\hat m<24.3$",
    ),
    ("faint", "Faint", lambda mag: mag >= 25.0, r"$\hat m\geq25$"),
)
MASS_LEVELS = (0.50, 0.80, 0.95)
COLORS = {"v3.2": "#0072B2", "v3.3": "#D55E00"}
HISTOGRAM_BINS = 64
SMOOTHING_SIGMA_BINS = 2.0


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _representative_cases(sample: pd.DataFrame, count: int = 4) -> list[int]:
    cases = np.sort(sample["case"].unique())
    if len(cases) < count:
        raise ValueError(f"need at least {count} cases, found {len(cases)}")
    locations = np.linspace(0, len(cases) - 1, count + 2)[1:-1]
    return cases[np.rint(locations).astype(int)].astype(int).tolist()


def _three_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 3:
        raise ValueError("stratum/case contains fewer than three galaxies")
    ordered = frame.sort_values(
        ["measured_mag_auto", "input_index"], kind="stable"
    ).reset_index(drop=True)
    positions = np.rint(np.linspace(0, len(ordered) - 1, 7)[1::2]).astype(int)
    return ordered.iloc[positions].copy()


def select_gallery_rows(frame: pd.DataFrame, cases: list[int]) -> pd.DataFrame:
    """Select three magnitude-spaced rows per stratum from each of four cases."""

    selected = []
    for key, label, predicate, interval in STRATA:
        for case in cases:
            subset = frame.loc[
                (frame["case"] == case)
                & predicate(frame["measured_mag_auto"].to_numpy(float))
            ]
            rows = _three_quantiles(subset)
            rows["stratum"] = key
            rows["stratum_label"] = label
            rows["stratum_interval"] = interval
            selected.append(rows)
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 36 or result[["case", "input_index"]].duplicated().any():
        raise RuntimeError("gallery selection is not 36 unique galaxies")
    return result


def _build_case_frame(
    sample: pd.DataFrame,
    *,
    case: int,
    constgold_root: Path,
    crowd: pd.DataFrame,
    shear: float,
) -> pd.DataFrame:
    observations = sample.loc[sample["case"] == case].copy()
    truth_path = (
        _raw_catalogue_root(constgold_root, case, shear)
        / "input"
        / "gals_info_tile180.0_-0.5.feather"
    )
    truth = _read_truth(truth_path)
    position = pd.Series(np.arange(len(truth), dtype=np.int64), index=truth["index"])
    rows = position.reindex(observations["input_index"]).to_numpy()
    if pd.isna(rows).any():
        raise RuntimeError(f"case {case} sample identities are absent from truth")
    primary_indices = rows.astype(np.int64)
    prior = ScenePrior.from_catalogue(
        truth,
        guard_radius_arcsec=7.0,
        duplicate_policy="error",
    ).shear(shear, 0.0)
    frame = prior.object_view(
        neighbour_radius_arcsec=7.0,
        conditions=CONDITIONS,
        include_crowding=False,
        primary_indices=primary_indices,
    )
    if not np.array_equal(
        frame["input_index"].to_numpy(np.int64),
        observations["input_index"].to_numpy(np.int64),
    ):
        raise RuntimeError(f"case {case} truth/context order changed")
    frame.insert(0, "case", case)
    matched_crowd = observations[["case", "input_index"]].merge(
        crowd,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if matched_crowd[list(CROWD_COLUMNS)].isna().any().any():
        raise RuntimeError(f"case {case} has unmatched crowding rows")
    for name in CROWD_COLUMNS:
        frame[name] = matched_crowd[name].to_numpy(float)
    for name in (
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    ):
        frame[name] = observations[name].to_numpy(float)
    return frame


def _density_thresholds(counts: np.ndarray) -> list[float]:
    ranked = np.sort(counts.ravel())[::-1]
    total = ranked.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empty likelihood histogram")
    cumulative = np.cumsum(ranked) / total
    thresholds = [ranked[np.searchsorted(cumulative, mass)] for mass in MASS_LEVELS]
    return np.unique(np.sort(thresholds)).tolist()


def _contour(ax, x, y, *, color, linestyle, extent):
    edges = np.linspace(-extent, extent, HISTOGRAM_BINS + 1)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=(edges, edges))
    density = gaussian_filter(counts.T, sigma=SMOOTHING_SIGMA_BINS, mode="constant")
    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    levels = _density_thresholds(density)
    return ax.contour(
        centers,
        centers,
        density,
        levels=levels,
        colors=[color],
        linestyles=[linestyle],
        linewidths=np.linspace(0.65, 1.25, len(levels)),
        alpha=0.95,
    )


def _render(selected, samples32, samples33, output, *, draws, extent):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        6,
        6,
        figsize=(12.2, 12.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for index, ax in enumerate(axes.flat):
        row = selected.iloc[index]
        _contour(
            ax,
            samples32[index, :, 0],
            samples32[index, :, 1],
            color=COLORS["v3.2"],
            linestyle="solid",
            extent=extent,
        )
        _contour(
            ax,
            samples33[index, :, 0],
            samples33[index, :, 1],
            color=COLORS["v3.3"],
            linestyle="dashed",
            extent=extent,
        )
        ax.scatter(
            row["measured_ngmix_g1"],
            row["measured_ngmix_g2"],
            marker="x",
            s=22,
            linewidth=1.2,
            color="black",
            zorder=5,
        )
        ax.set_title(
            rf"$\hat m={row['measured_mag_auto']:.2f}$, $m_{{true}}={row['r_input_p']:.2f}$",
            pad=2.5,
        )
        ax.text(
            0.03,
            0.03,
            f"c{int(row['case'])} / {int(row['input_index'])}",
            transform=ax.transAxes,
            fontsize=5.4,
            color="0.35",
        )
        ax.axhline(0, color="0.88", lw=0.45, zorder=0)
        ax.axvline(0, color="0.88", lw=0.45, zorder=0)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="out", length=2.2, width=0.55)
        ax.spines[["top", "right"]].set_visible(False)

    for row in range(6):
        axes[row, 0].set_ylabel(r"measured $\hat e_2$")
    for col in range(6):
        axes[-1, col].set_xlabel(r"measured $\hat e_1$")

    for start, (_, label, _, interval) in zip((0, 2, 4), STRATA):
        y = 1.0 - (start + 1.0) / 6.0
        fig.text(
            0.009,
            y,
            f"{label}\n{interval}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            weight="bold",
        )
        if start:
            fig.add_artist(
                Line2D([0.045, 0.995], [1.0 - start / 6.0] * 2,
                       transform=fig.transFigure, color="0.78", lw=0.8)
            )

    handles = [
        Line2D([0], [0], color=COLORS["v3.2"], lw=1.7, label="V3.2 flow"),
        Line2D(
            [0], [0], color=COLORS["v3.3"], lw=1.7, ls="--", label="V3.3 flow"
        ),
        Line2D(
            [0], [0], color="black", marker="x", lw=0, markersize=6,
            label="ConstGold measurement"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.993), fontsize=9)
    fig.suptitle(
        "Per-galaxy conditional measurement likelihood: V3.2 versus V3.3",
        y=0.999,
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.52,
        0.007,
        f"{draws:,} common-seed draws per flow and galaxy; contours enclose "
        "50%, 80%, and 95%. Flow term only: no detection, measured cuts, or "
        r"$R_{\rm blend}$.",
        ha="center",
        fontsize=8,
        color="0.25",
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.045, top=0.965,
                        hspace=0.26, wspace=0.12)
    fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sample", type=Path, required=True)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument("--v32-model", type=Path, required=True)
    parser.add_argument("--v33-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output prefix; writes PNG, PDF, JSON, parquet, and NPZ")
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--extent", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    outputs = [args.output.with_suffix(ext) for ext in (".png", ".pdf", ".json", ".parquet", ".npz")]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")
    if args.draws <= 0 or args.extent <= 0:
        raise ValueError("draws and extent must be positive")

    sample = pd.read_parquet(args.source_sample)
    cases = _representative_cases(sample, count=4)
    crowd = pd.read_feather(
        args.crowd_lookup,
        columns=["case", "input_index", *CROWD_COLUMNS],
    )
    crowd = crowd.loc[crowd["case"].isin(cases)].copy()
    frames = [
        _build_case_frame(
            sample,
            case=case,
            constgold_root=args.constgold_root,
            crowd=crowd,
            shear=args.shear,
        )
        for case in cases
    ]
    candidates = pd.concat(frames, ignore_index=True)
    selected = select_gallery_rows(candidates, cases)

    # The same seed pairs the base random draws across the two architectures.
    draw32 = sample_measurement(
        [args.v32_model],
        selected,
        n_samples=args.draws,
        random_seed=args.seed,
        device=args.device,
    )
    draw33 = sample_measurement(
        [args.v33_model],
        selected,
        n_samples=args.draws,
        random_seed=args.seed,
        device=args.device,
    )
    samples32 = np.asarray(draw32.samples, dtype=np.float32)
    samples33 = np.asarray(draw33.samples, dtype=np.float32)
    if samples32.shape != samples33.shape or samples32.shape != (36, args.draws, 4):
        raise RuntimeError(f"unexpected sample shapes {samples32.shape}, {samples33.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output.with_suffix(".parquet"), index=False)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        v32=samples32,
        v33=samples33,
    )
    _render(selected, samples32, samples33, args.output,
            draws=args.draws, extent=args.extent)

    report = {
        "purpose": "V3.2/V3.3 per-galaxy conditional shape-likelihood gallery",
        "projection": ["measured_ngmix_g1", "measured_ngmix_g2"],
        "likelihood_terms_excluded": [
            "detection probability", "measured selection", "R_blend"
        ],
        "source_sample": str(args.source_sample.resolve()),
        "source_sample_sha256": _sha256(args.source_sample),
        "constgold_cases": cases,
        "selection": {
            "rule": "three measured-magnitude quantiles per stratum and case",
            "strata": {key: interval for key, _, _, interval in STRATA},
            "n_per_stratum": 12,
        },
        "shear": [args.shear, 0.0],
        "draws_per_model_per_galaxy": args.draws,
        "common_random_seed": args.seed,
        "contour_mass_levels": list(MASS_LEVELS),
        "histogram_bins_per_axis": HISTOGRAM_BINS,
        "gaussian_smoothing_sigma_bins": SMOOTHING_SIGMA_BINS,
        "axis_extent": [-args.extent, args.extent],
        "models": {
            "v3.2": {"path": str(args.v32_model.resolve()), "sha256": _sha256(args.v32_model)},
            "v3.3": {"path": str(args.v33_model.resolve()), "sha256": _sha256(args.v33_model)},
        },
        "outputs": {
            path.suffix.lstrip("."): {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in outputs
            if path.suffix != ".json"
        },
        "unmatched_rows": 0,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
