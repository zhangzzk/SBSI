"""Combine azimuth-averaged half-shear pair toys and plot R_blend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats


QUANTILE_PROBABILITIES = np.asarray(
    [0.0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5,
     0.75, 0.95, 0.99, 0.995, 0.999, 1.0],
    dtype=float,
)
NEGATIVE_THRESHOLDS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1)


def wilson_interval(count: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= count <= total:
        raise ValueError("invalid binomial count")
    p = count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return [float(center - half), float(center + half)]


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("distribution needs at least two finite values")
    quantiles = np.quantile(values, QUANTILE_PROBABILITIES)
    negative = int(np.count_nonzero(values < 0.0))
    thresholded = {}
    for threshold in NEGATIVE_THRESHOLDS:
        mask = values < -threshold
        thresholded[f"{threshold:g}"] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_signed_contribution": float(np.where(mask, values, 0.0).mean()),
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "pair_sampling_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        "negative_count": negative,
        "negative_fraction": float(negative / values.size),
        "negative_fraction_wilson95": wilson_interval(negative, len(values)),
        "negative_thresholds": thresholded,
        "negative_part_mean_contribution": float(np.minimum(values, 0.0).mean()),
        "positive_part_mean_contribution": float(np.maximum(values, 0.0).mean()),
        "mean_after_zero_floor": float(np.maximum(values, 0.0).mean()),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "quantiles": {
            f"q{probability:.3f}": float(value)
            for probability, value in zip(
                QUANTILE_PROBABILITIES, quantiles
            )
        },
    }


def case_mean_stat(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("case statistic needs at least two finite case means")
    return {
        "mean": float(values.mean()),
        "case_sd": float(values.std(ddof=1)),
        "case_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        "n_cases": int(values.size),
        "case_values": values.tolist(),
    }


def sign_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    toy = frame.R_blend_toy.to_numpy(float)
    emulator = frame.R_emulator_v22.to_numpy(float)
    toy_negative = toy < 0.0
    emulator_negative = emulator < 0.0
    counts = {
        "toy_negative_emulator_negative": int(np.sum(toy_negative & emulator_negative)),
        "toy_negative_emulator_nonnegative": int(np.sum(toy_negative & ~emulator_negative)),
        "toy_nonnegative_emulator_negative": int(np.sum(~toy_negative & emulator_negative)),
        "toy_nonnegative_emulator_nonnegative": int(np.sum(~toy_negative & ~emulator_negative)),
    }
    return {
        "counts": counts,
        "sign_agreement_fraction": float(np.mean(toy_negative == emulator_negative)),
        "toy_negative_given_emulator_negative": float(
            np.mean(toy_negative[emulator_negative])
        ) if np.any(emulator_negative) else None,
        "emulator_negative_given_toy_negative": float(
            np.mean(emulator_negative[toy_negative])
        ) if np.any(toy_negative) else None,
        "mean_toy_when_emulator_negative": float(toy[emulator_negative].mean())
        if np.any(emulator_negative) else None,
        "mean_toy_when_emulator_nonnegative": float(toy[~emulator_negative].mean())
        if np.any(~emulator_negative) else None,
        "spearman_toy_emulator": float(stats.spearmanr(toy, emulator).statistic),
        "spearman_toy_original_forward_label": float(
            stats.spearmanr(toy, frame.R_label_forward.to_numpy(float)).statistic
        ),
    }


def plot_histogram(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    bins: int,
    tail_probability: float,
    output_prefix: str,
) -> None:
    toy = frame.R_blend_toy.to_numpy(float)
    emulator = frame.R_emulator_v22.to_numpy(float)
    lower, upper = np.quantile(toy, [tail_probability, 1.0 - tail_probability])
    limit = float(max(abs(lower), abs(upper)))
    if not np.isfinite(limit) or limit <= 0.0:
        raise RuntimeError("invalid histogram display interval")
    edges = np.linspace(-limit, limit, bins + 1)
    toy_counts = np.histogram(toy, bins=edges)[0]
    emulator_counts = np.histogram(emulator, bins=edges)[0]
    toy_fraction = toy_counts / len(toy)
    emulator_fraction = emulator_counts / len(emulator)
    toy_sem = np.sqrt(toy_fraction * (1.0 - toy_fraction) / len(toy))
    center = 0.5 * (edges[:-1] + edges[1:])
    positive = toy_fraction[toy_fraction > 0.0]
    floor = max(float(positive.min()) * 0.2 if len(positive) else 1.0e-7, 1.0e-7)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axis = plt.subplots(figsize=(9.2, 4.45))
    axis.axvspan(-limit, 0.0, color="0.94", zorder=0)
    axis.stairs(
        toy_fraction, edges, color="#0072B2", fill=True, alpha=0.30,
        linewidth=1.3,
        label=r"Noiseless toy: 8-azimuth mean",
    )
    axis.fill_between(
        center,
        np.maximum(toy_fraction - 1.96 * toy_sem, floor),
        toy_fraction + 1.96 * toy_sem,
        step="mid", color="#0072B2", alpha=0.16, linewidth=0.0,
        label="Pair-sampling 95% interval",
    )
    axis.stairs(
        emulator_fraction, edges, color="#D55E00", linewidth=1.15,
        label="Frozen V2.2 prediction",
    )
    toy_summary = summary["distributions"]["R_blend_toy"]
    mean = toy_summary["mean"]
    median = toy_summary["quantiles"]["q0.500"]
    axis.axvline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axis.axvline(
        mean, color="#009E73", linestyle="--", linewidth=1.3,
        label=fr"Toy mean $={mean:+.4f}$",
    )
    axis.axvline(
        median, color="0.15", linestyle="-.", linewidth=1.0,
        label=fr"Toy median $={median:+.4f}$",
    )
    negative = toy_summary["negative_fraction"]
    ci = toy_summary["negative_fraction_wilson95"]
    below_milli = toy_summary["negative_thresholds"]["0.001"]["fraction"]
    below_centi = toy_summary["negative_thresholds"]["0.01"]["fraction"]
    axis.text(
        0.025, 0.965,
        fr"Toy $P(R_{{\rm blend}}<0)={100*negative:.2f}\%$ "
        f"(95% CI {100*ci[0]:.2f}–{100*ci[1]:.2f}%)\n"
        fr"$P(R<-10^{{-3}})={100*below_milli:.2f}\%$; "
        fr"$P(R<-10^{{-2}})={100*below_centi:.2f}\%$",
        transform=axis.transAxes, ha="left", va="top", fontsize=9.0,
    )
    axis.set_yscale("log")
    axis.set_ylim(bottom=floor)
    axis.set_xlim(-limit, limit)
    axis.set_xlabel(
        r"Azimuth-averaged pair response "
        r"$R_{\rm blend}=\frac{1}{2}\,\mathrm{Tr}(\langle R\rangle_\phi)$"
    )
    axis.set_ylabel("Fraction of sampled pairs per bin")
    displayed = np.mean((toy >= -limit) & (toy <= limit))
    axis.set_title(
        "Random V2.2 half-shear pairs: noiseless two-object toys, "
        "8 positions at 45° spacing\n"
        f"n={len(frame):,}; central {100*displayed:.2f}% of toy responses displayed"
    )
    axis.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.90", linewidth=0.55)
    fig.subplots_adjust(left=0.095, right=0.70, bottom=0.145, top=0.84)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    summary["plot"] = {
        "bins": int(bins),
        "display_tail_probability_each_side": float(tail_probability),
        "display_limits": [-limit, limit],
        "displayed_toy_fraction": float(displayed),
    }


def markdown(summary: dict[str, Any]) -> str:
    toy = summary["distributions"]["R_blend_toy"]
    emulator = summary["distributions"]["R_emulator_v22"]
    case = summary["case_balanced_means"]["R_blend_toy"]
    signs = summary["sign_comparison"]
    ci = toy["negative_fraction_wilson95"]
    below_milli = toy["negative_thresholds"]["0.001"]
    below_centi = toy["negative_thresholds"]["0.01"]
    native = summary["azimuth_diagnostics"]["native_angle_R_blend"]
    azimuth_sd = summary["azimuth_diagnostics"]["within_pair_azimuth_sd"]
    return "\n".join([
        "# Azimuth-averaged noiseless single-neighbour response",
        "",
        f"A uniform sample of `{summary['coverage']['n_success']:,}` successful pair rows "
        "from the exact V2.2 half-shear training support was re-rendered as isolated "
        "primary+secondary stamps. For each pair, the secondary position was rotated "
        "around the primary in eight 45-degree steps, keeping both source position "
        "angles fixed. The primary is unsheared; only the secondary is sheared "
        "antithetically along g1 and g2. All statistics below use the response matrix "
        "averaged over those eight positions.",
        "",
        f"- Toy mean: `{toy['mean']:+.6f}` (pair-sampling SEM "
        f"`{toy['pair_sampling_sem']:.6f}`); median: "
        f"`{toy['quantiles']['q0.500']:+.6f}`.",
        f"- Case-balanced toy mean: `{case['mean']:+.6f} +- {case['case_sem']:.6f}` "
        f"across `{case['n_cases']}` cases.",
        f"- For comparison, the native-position-only toy mean on these same pairs is "
        f"`{native['mean']:+.6f}`; the median within-pair standard deviation over "
        f"azimuth is `{azimuth_sd['quantiles']['q0.500']:.6f}`.",
        f"- Negative toy responses: `{100*toy['negative_fraction']:.2f}%` "
        f"(Wilson 95% CI `{100*ci[0]:.2f}--{100*ci[1]:.2f}%`).",
        f"- Material negative thresholds: `{100*below_milli['fraction']:.2f}%` "
        f"fall below `-0.001`, and `{100*below_centi['fraction']:.2f}%` fall "
        "below `-0.01`.",
        f"- Negative and positive signed contributions to the toy mean are "
        f"`{toy['negative_part_mean_contribution']:+.6f}` and "
        f"`{toy['positive_part_mean_contribution']:+.6f}`.",
        f"- Toy 1--99% interval: `[{toy['quantiles']['q0.010']:+.6f}, "
        f"{toy['quantiles']['q0.990']:+.6f}]`.",
        f"- Frozen V2.2 negative predictions on the same pairs: "
        f"`{100*emulator['negative_fraction']:.2f}%`.",
        f"- Among emulator-negative pairs, `{100*signs['toy_negative_given_emulator_negative']:.2f}%` "
        "are also negative in the clean toy.",
        f"- Toy/emulator Spearman correlation: `{signs['spearman_toy_emulator']:+.4f}`.",
        "",
        "The histogram shows the eight-position-averaged toy distribution and the "
        "frozen V2.2 prediction on "
        "the exact same sampled pairs. Its shaded band is the pair-sampling 95% "
        "interval for each bin.",
        "",
        "Scope: this is a two-object, noiseless, true-centred, central-g=0.05 "
        "measurement. It intentionally removes the original full scene, pixel noise, "
        "detection/matching, centroid motion, and the forward-g=0.2 label convention. "
        "It tests whether the noiseless measured isolated-pair response can be "
        "negative; it is not a replacement label for the original half-shear "
        "simulation. Rare ngmix fit-branch tails remain measurement behavior, not "
        "proof of a smooth physical response kernel.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--n-shards", type=int, default=40)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--tail-probability", type=float, default=0.001)
    parser.add_argument("--min-success-fraction", type=float, default=0.98)
    args = parser.parse_args()
    if args.bins < 40 or not 0.0 < args.tail_probability < 0.05:
        raise ValueError("invalid histogram binning")
    outputs = [
        f"{args.output_prefix}.{suffix}"
        for suffix in ("feather", "rotations.feather", "json", "md", "pdf", "png")
    ]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    input_dir = Path(args.input_dir)
    parts = []
    rotation_parts = []
    audits = []
    for shard in range(args.n_shards):
        feather = input_dir / f"pairs_shard{shard:03d}.feather"
        rotations_feather = input_dir / f"rotations_shard{shard:03d}.feather"
        audit = input_dir / f"audit_shard{shard:03d}.json"
        if not feather.exists() or not rotations_feather.exists() or not audit.exists():
            raise FileNotFoundError(
                f"missing shard outputs {feather} / {rotations_feather} / {audit}"
            )
        parts.append(pd.read_feather(feather))
        rotation_parts.append(pd.read_feather(rotations_feather))
        with open(audit, encoding="utf-8") as handle:
            audits.append(json.load(handle))
    frame = pd.concat(parts, ignore_index=True).sort_values(
        "sample_id", kind="mergesort"
    ).reset_index(drop=True)
    rotations = pd.concat(rotation_parts, ignore_index=True).sort_values(
        ["sample_id", "rotation_index"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = pd.read_feather(args.manifest).sort_values(
        "sample_id", kind="mergesort"
    ).reset_index(drop=True)
    if frame.sample_id.duplicated().any() or len(frame) != len(manifest):
        raise RuntimeError("shards do not provide one row per manifest sample")
    if not frame[["sample_id", "catalogue_row"]].equals(
        manifest[["sample_id", "catalogue_row"]]
    ):
        raise RuntimeError("shard keys do not replay the frozen manifest")
    for column in ("R_emulator_v22", "R_label_forward", "R_label_null"):
        if not np.array_equal(
            frame[column].to_numpy(float), manifest[column].to_numpy(float)
        ):
            raise RuntimeError(f"shard column {column} drifted from manifest")
    designs = {
        (item["g"], item["stamp"], item["fit_seed"], item["n_azimuths"])
        for item in audits
    }
    if len(designs) != 1:
        raise RuntimeError("toy shard designs differ")

    success = frame.success.astype(bool).to_numpy()
    success_fraction = float(success.mean())
    if success_fraction < args.min_success_fraction:
        raise RuntimeError(
            f"success fraction {success_fraction:.2%} below "
            f"{args.min_success_fraction:.2%}"
        )
    complete = frame.loc[success].copy()
    numeric = [
        "R_blend_toy", "R11_toy", "R12_toy", "R21_toy", "R22_toy",
        "R_emulator_v22", "R_label_forward", "R_label_null",
        "latent_distance_arcsec", "catalogue_minus_latent_distance_arcsec",
    ]
    if not np.isfinite(complete[numeric].to_numpy(float)).all():
        raise RuntimeError("successful toy rows contain non-finite measurements")
    if not np.allclose(
        complete.R_blend_toy,
        0.5 * (complete.R11_toy + complete.R22_toy),
        rtol=0.0, atol=2.0e-14,
    ):
        raise RuntimeError("stored toy trace does not replay response matrix")

    n_azimuths = int(audits[0]["n_azimuths"])
    successful_ids = complete.sample_id.to_numpy(np.int64)
    if rotations[["sample_id", "rotation_index"]].duplicated().any():
        raise RuntimeError("duplicate pair/rotation keys")
    if len(rotations) != len(complete) * n_azimuths:
        raise RuntimeError("rotation count does not replay successful pair count")
    counts = rotations.groupby("sample_id", sort=False).size()
    if not np.array_equal(np.sort(counts.index.to_numpy(np.int64)), successful_ids):
        raise RuntimeError("rotation pair keys differ from successful pair keys")
    if not np.all(counts.to_numpy(np.int64) == n_azimuths):
        raise RuntimeError("a successful pair does not have every azimuth")
    rotation_numeric = [
        "R_blend_toy", "R11_toy", "R12_toy", "R21_toy", "R22_toy",
        "azimuth_offset_deg", "rotated_dx_arcsec", "rotated_dy_arcsec",
    ]
    if not np.isfinite(rotations[rotation_numeric].to_numpy(float)).all():
        raise RuntimeError("rotation rows contain non-finite measurements")
    expected_offsets = np.arange(n_azimuths, dtype=float) * (360.0 / n_azimuths)
    for _, group in rotations.groupby("sample_id", sort=False):
        if not np.array_equal(group.rotation_index.to_numpy(int), np.arange(n_azimuths)):
            raise RuntimeError("rotation indices are incomplete or out of order")
        if not np.allclose(group.azimuth_offset_deg, expected_offsets, atol=1.0e-12):
            raise RuntimeError("rotation offsets do not match the frozen grid")
    if not np.allclose(
        rotations.R_blend_toy,
        0.5 * (rotations.R11_toy + rotations.R22_toy),
        rtol=0.0, atol=2.0e-14,
    ):
        raise RuntimeError("stored rotation trace does not replay response matrix")
    rotation_means = rotations.groupby("sample_id", sort=True)[
        ["R11_toy", "R12_toy", "R21_toy", "R22_toy", "R_blend_toy"]
    ].mean()
    pair_means = complete.set_index("sample_id").sort_index()[rotation_means.columns]
    if not np.allclose(rotation_means, pair_means, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("pair response does not replay the mean over azimuths")
    native = rotations.loc[rotations.rotation_index == 0].set_index(
        "sample_id"
    ).sort_index().R_blend_toy
    if not np.allclose(
        native, complete.set_index("sample_id").sort_index().R_blend_toy_native,
        rtol=0.0, atol=2.0e-14,
    ):
        raise RuntimeError("stored native-angle response does not replay rotation zero")

    g, stamp, fit_seed, design_n_azimuths = next(iter(designs))
    summary: dict[str, Any] = {
        "design": {
            "description": audits[0]["design"],
            "g": float(g), "stamp": int(stamp), "fit_seed": int(fit_seed),
            "n_azimuths": int(design_n_azimuths),
            "azimuth_offsets_deg": expected_offsets.tolist(),
            "position_rotation": audits[0]["position_rotation"],
            "pixel_noise": 0.0, "primary_sheared": False,
            "positions_sheared": False,
            "response_scalar": "0.5 * trace of central g1/g2 response matrix",
            "scene_members": "sampled primary plus sampled secondary only",
        },
        "sampling": {
            "manifest": os.path.abspath(args.manifest),
            "population": "exact V2.2 half-shear training support, cases 40--199",
            "scheme": "uniform iid-priority sample over eligible catalogue pair rows",
            "worst_case_negative_fraction_95_halfwidth": float(
                1.96 * 0.5 / np.sqrt(len(complete))
            ),
        },
        "coverage": {
            "n_sampled": int(len(frame)),
            "n_success": int(success.sum()),
            "n_failure": int((~success).sum()),
            "success_fraction": success_fraction,
            "n_cases": int(complete.case.nunique()),
            "failure_types": {
                str(name): int(count) for name, count in
                frame.loc[~success, "error_type"].value_counts().items()
            },
        },
        "distributions": {
            column: distribution(complete[column].to_numpy(float))
            for column in ("R_blend_toy", "R_emulator_v22", "R_label_forward", "R_label_null")
        },
        "azimuth_diagnostics": {
            "native_angle_R_blend": distribution(
                complete.R_blend_toy_native.to_numpy(float)
            ),
            "within_pair_azimuth_sd": distribution(
                complete.R_blend_toy_azimuth_sd.to_numpy(float)
            ),
            "n_rotation_rows": int(len(rotations)),
        },
        "case_balanced_means": {
            column: case_mean_stat(complete, column)
            for column in ("R_blend_toy", "R_emulator_v22", "R_label_forward")
        },
        "sign_comparison": sign_comparison(complete),
        "geometry": {
            "latent_distance_arcsec": distribution(
                complete.latent_distance_arcsec.to_numpy(float)
            ),
            "catalogue_minus_latent_distance_arcsec": distribution(
                complete.catalogue_minus_latent_distance_arcsec.to_numpy(float)
            ),
        },
        "artifacts": {
            "combined_feather": os.path.abspath(f"{args.output_prefix}.feather"),
            "combined_rotations_feather": os.path.abspath(
                f"{args.output_prefix}.rotations.feather"
            ),
            "figure_pdf": os.path.abspath(f"{args.output_prefix}.pdf"),
            "figure_png": os.path.abspath(f"{args.output_prefix}.png"),
        },
    }
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    plot_histogram(
        complete, summary, args.bins, args.tail_probability, args.output_prefix
    )
    complete.reset_index(drop=True).to_feather(f"{args.output_prefix}.feather")
    rotations.reset_index(drop=True).to_feather(
        f"{args.output_prefix}.rotations.feather"
    )
    report = markdown(summary)
    Path(f"{args.output_prefix}.md").write_text(report, encoding="utf-8")
    with open(f"{args.output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(report, flush=True)
    print("HALFSHEAR_NOISELESS_PAIR_TOY_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
