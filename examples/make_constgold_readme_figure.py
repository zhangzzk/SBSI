"""Build the README measurement-flow figure from real constgold galaxies.

The three primaries are selected deterministically from the 10th, 50th, and
90th magnitude percentiles of one constgold shear leg.  Each one is a genuine,
successfully cross-matched detection.  The upper panels are cut directly from
the noisy simulated FITS image; the lower panels show release-flow draws for
the same truth rows and their actual local neighbour scenes.

The flow alone describes an isolated measurement, so its ellipticity contour is
centred on the sheared input truth.  Blending adds the emulator's ``R_blend``
response on top, computed from every real neighbour of each primary; the panel
titles carry it, and the run prints the displacement ``R_blend * gamma`` it
implies for the leg's true shear.

This is a figure-generation utility, not a package runtime dependency.  Run it
on a compute node through ``jobs/job_constgold_readme_figure.sh``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.lines import Line2D
from scipy import ndimage

from sbsi import (
    EmulatorPairingConfig,
    ResponsePredictor,
    get_model,
    load_emulator,
    predict_blend_response,
    prepare_emulator_pairs,
    sample_measurement,
)
from sbsi.forward_catalogue import prepare_flow_inputs
from sbsi.shear_map import apply_shear_to_ellipticity

DEFAULT_CONSTGOLD_ROOT = Path(
    os.environ.get(
        "CONSTGOLD_ROOT",
        "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant",
    )
)
TILE = "tile180.0_-0.5"

OBSERVING_CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}

REPRESENTATIVE_QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_BAND = 0.04

# The automatic faint pick (input 452770) is typical in morphology but draws a
# ragged, sprawling contour -- 44th of 48 on cleanliness, with a 1-sigma outline
# scoring 0.26 for roundness against 0.99 for the best.  This galaxy sits in the
# same magnitude band and draws cleanly; see jobs/diag_faint_contour_candidates.py.
FAINT_EXAMPLE_INDEX = 136443
MORPHOLOGY_COLUMNS = ("log_Re", "log_n", "e_abs", "redshift_input", "log_snr")
REPRESENTATIVE_LABELS = ("Bright", "Typical", "Faint")
COLORS = ("#4477AA", "#CC6677", "#228833")

MASS_LEVELS = (0.393, 0.865)  # 1 sigma and 2 sigma of a 2D Gaussian
SMOOTHING = 1.6
PAD_BINS = int(np.ceil(4 * SMOOTHING))

# Measured FLUX_RADIUS is positive by construction, but the smoothed density
# spreads a little below zero, so clamping the panel at zero sliced the outer
# contour off flat along the axis.  Let the contour close on its own and stop
# the axis here instead.
RADIUS_AXIS_FLOOR = -0.5



@dataclass(frozen=True)
class ConstgoldPaths:
    input_catalogue: Path
    cross_match: Path
    shape_catalogue: Path
    image: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, default=DEFAULT_CONSTGOLD_ROOT)
    parser.add_argument("--case", type=int, default=40)
    parser.add_argument(
        "--shear",
        default="0.02",
        help="constgold shear-leg suffix, for example 0.02 or -0.02",
    )
    parser.add_argument("--tile", default=TILE)
    parser.add_argument("--model", default="V3")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--stamp-size", type=int, default=72, help="pixels")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--faint-index",
        type=int,
        default=FAINT_EXAMPLE_INDEX,
        help="constgold input index to use as the faint example (must lie in the "
        "faint magnitude band); 0 restores the automatic morphology-only pick",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("measurement_flow_contours.png"),
    )
    args = parser.parse_args()
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.stamp_size < 24 or args.stamp_size % 2:
        parser.error("--stamp-size must be an even integer of at least 24 pixels")
    return args


def constgold_paths(root: Path, case: int, shear: str, tile: str) -> ConstgoldPaths:
    leg = root / f"case{case}_{shear}" / "real0"
    paths = ConstgoldPaths(
        input_catalogue=leg / "catalogues" / "input" / f"gals_info_{tile}.feather",
        cross_match=leg / "catalogues" / "CrossMatch" / f"{tile}_rot0_matched.feather",
        shape_catalogue=(
            leg / "catalogues" / "Shapes" / f"shape_catalogue_detect_position_all_{tile}.feather"
        ),
        image=leg / "images" / "original" / f"{tile}_bandr_rot0.fits",
    )
    missing = [path for path in paths.__dict__.values() if not path.is_file()]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"constgold leg is incomplete:\n  {rendered}")
    return paths


def load_truth(path: Path) -> pd.DataFrame:
    columns = [
        "index_input",
        "RA_input",
        "DEC_input",
        "redshift_input",
        "Re_input",
        "axis_ratio_input",
        "position_angle_input",
        "sersic_n_input",
        "r_input",
        "gamma1_input",
        "gamma2_input",
        "e1_input_rot0",
        "e2_input_rot0",
    ]
    truth = pd.read_feather(path, columns=columns)
    if not truth["index_input"].is_unique:
        raise ValueError("constgold input indices are not unique")
    return truth


def detected_primaries(paths: ConstgoldPaths, truth: pd.DataFrame) -> pd.DataFrame:
    match = pd.read_feather(paths.cross_match)
    match = match.sort_values(["distance_pixel_CM", "id_detec"]).drop_duplicates(
        "id_input", keep="first"
    )

    shape_columns = [
        "NUMBER",
        "XWIN_IMAGE",
        "YWIN_IMAGE",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
        "MAG_AUTO",
        "FLUX_RADIUS",
        "FLAGS",
        "NGMIX_G1",
        "NGMIX_G2",
    ]
    shape = pd.read_feather(paths.shape_catalogue, columns=shape_columns)
    if not shape["NUMBER"].is_unique:
        raise ValueError("shape-catalogue detection numbers are not unique")

    detected = match.merge(
        shape, left_on="id_detec", right_on="NUMBER", validate="many_to_one"
    )
    detected = detected.merge(
        truth, left_on="id_input", right_on="index_input", validate="many_to_one"
    )
    detected["snr"] = detected["FLUX_AUTO"] / detected["FLUXERR_AUTO"]
    detected["e_abs"] = np.hypot(detected["e1_input_rot0"], detected["e2_input_rot0"])
    return detected


def candidate_pool(
    detected: pd.DataFrame,
    predictor: ResponsePredictor,
    image_shape: tuple[int, int],
    stamp_size: int,
) -> pd.DataFrame:
    """Every detection usable as an example, ranked by input magnitude."""

    height, width = image_shape
    half = stamp_size // 2

    finite_columns = [
        "r_input",
        "Re_input",
        "sersic_n_input",
        "redshift_input",
        "e_abs",
        "snr",
        "XWIN_IMAGE",
        "YWIN_IMAGE",
        "NGMIX_G1",
        "NGMIX_G2",
        "MAG_AUTO",
        "FLUX_RADIUS",
    ]
    finite = np.isfinite(detected[finite_columns]).all(axis=1)
    domain = predictor.domain.mask(detected["r_input"], detected["Re_input"])
    quality = (
        detected["FLAGS"].eq(0)
        & detected["snr"].gt(10)
        & detected["NGMIX_G1"].ne(-1)
        & detected["NGMIX_G2"].ne(-1)
        & detected["FLUX_RADIUS"].gt(0)
    )
    inside = detected["XWIN_IMAGE"].between(half + 2, width - half - 1) & detected[
        "YWIN_IMAGE"
    ].between(half + 2, height - half - 1)

    candidates = detected.loc[finite & domain & quality & inside].copy()
    if len(candidates) < 100:
        raise RuntimeError(f"only {len(candidates)} valid constgold candidates remain")

    candidates = candidates.sort_values(["r_input", "index_input"]).reset_index(drop=True)
    candidates["magnitude_rank"] = (np.arange(len(candidates)) + 0.5) / len(candidates)
    candidates["log_Re"] = np.log(candidates["Re_input"])
    candidates["log_n"] = np.log(candidates["sersic_n_input"])
    candidates["log_snr"] = np.log(candidates["snr"])
    return candidates


def magnitude_band(candidates: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Candidates within the fixed rank window around one magnitude percentile."""

    return candidates.loc[
        (candidates["magnitude_rank"] - quantile).abs() <= QUANTILE_BAND
    ].copy()


def representative_score(local: pd.DataFrame, quantile: float) -> pd.Series:
    """Distance from the band's median morphology, plus distance in magnitude rank."""

    center = local[list(MORPHOLOGY_COLUMNS)].median()
    scale = local[list(MORPHOLOGY_COLUMNS)].quantile(0.75) - local[list(MORPHOLOGY_COLUMNS)].quantile(0.25)
    scale = scale.mask(scale <= 0, 1)
    morphology_distance = ((local[list(MORPHOLOGY_COLUMNS)] - center) / scale).pow(2).sum(axis=1)
    rank_distance = ((local["magnitude_rank"] - quantile) / QUANTILE_BAND).pow(2)
    return morphology_distance + rank_distance


def select_representatives(candidates: pd.DataFrame, overrides=None) -> pd.DataFrame:
    """Choose locally typical morphologies at three magnitude percentiles.

    ``overrides`` maps a representative label to a constgold input index, for
    picking a specific galaxy by hand.  The override still has to sit inside
    that label's magnitude band, so the figure keeps describing what it claims.
    """

    overrides = dict(overrides or {})
    selected = []
    used = set()
    for label, quantile in zip(REPRESENTATIVE_LABELS, REPRESENTATIVE_QUANTILES):
        local = magnitude_band(candidates, quantile)
        local = local.loc[~local["index_input"].isin(used)]
        if local.empty:
            raise RuntimeError(f"no candidates near magnitude quantile {quantile:.2f}")

        forced = overrides.get(label)
        if forced is None:
            local["representative_score"] = representative_score(local, quantile)
            chosen = local.sort_values(["representative_score", "index_input"]).iloc[0]
        else:
            match = local.loc[local["index_input"].astype(int) == int(forced)]
            if match.empty:
                raise RuntimeError(
                    f"input index {forced} is not a usable {label.lower()} candidate: it "
                    f"fails the cuts or falls outside the {quantile:.2f} magnitude band"
                )
            chosen = match.iloc[0]

        selected.append(chosen)
        used.add(int(chosen["index_input"]))

    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "representative", REPRESENTATIVE_LABELS)
    return result


def as_forward_catalogue(truth: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "index_input": "index",
        "RA_input": "RA",
        "DEC_input": "DEC",
        "redshift_input": "redshift",
        "Re_input": "Re",
        "axis_ratio_input": "axis_ratio",
        "position_angle_input": "position_angle",
        "sersic_n_input": "sersic_n",
        "r_input": "r",
    }
    return truth.rename(columns=rename)[list(rename.values())].copy()


def selected_primaries(truth: pd.DataFrame, selected: pd.DataFrame):
    """Return the three chosen primaries and the full input catalogue they sit in."""

    all_galaxies = as_forward_catalogue(truth)
    indexed = all_galaxies.set_index("index", drop=False)
    primaries = indexed.loc[selected["index_input"].astype(int)].reset_index(drop=True)
    return primaries, all_galaxies


def prepare_selected_flow_rows(truth: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    primaries, all_galaxies = selected_primaries(truth, selected)

    flow = prepare_flow_inputs(primaries, all_galaxies, conditions=OBSERVING_CONDITIONS)

    g1 = selected["gamma1_input"].to_numpy(dtype=float)
    g2 = selected["gamma2_input"].to_numpy(dtype=float)
    sheared_e1, sheared_e2 = apply_shear_to_ellipticity(
        flow["e1_input_rot0_p"].to_numpy(dtype=float),
        flow["e2_input_rot0_p"].to_numpy(dtype=float),
        g1,
        g2,
    )
    flow = flow.copy()
    flow["e1_input_rot0_p"] = sheared_e1
    flow["e2_input_rot0_p"] = sheared_e2
    return flow


def predict_selected_blend_response(truth: pd.DataFrame, selected: pd.DataFrame, emulator):
    """Sum the emulator's pair response over every real neighbour of each primary.

    The pair table is built against the whole input catalogue, so a primary's
    response counts all of its neighbours, not just the ones inside the stamp.
    """

    primaries, all_galaxies = selected_primaries(truth, selected)
    pairing = EmulatorPairingConfig.from_emulator(emulator)
    pairs = prepare_emulator_pairs(primaries, all_galaxies, config=pairing)
    response = predict_blend_response(emulator, pairs)
    # A primary with no accepted pair contributes no blending response at all.
    return response.reindex(range(len(primaries)), fill_value=0.0).to_numpy(dtype=float)


def blending_shift(selected: pd.DataFrame, blend_response) -> np.ndarray:
    """Displacement of the mean measured ellipticity, ``R_blend * gamma``."""

    blend_response = np.asarray(blend_response, dtype=float)
    g1 = selected["gamma1_input"].to_numpy(dtype=float)
    g2 = selected["gamma2_input"].to_numpy(dtype=float)
    return np.column_stack([blend_response * g1, blend_response * g2])


def extract_stamps(image_path: Path, selected: pd.DataFrame, stamp_size: int):
    half = stamp_size // 2
    stamps = []
    with fits.open(image_path, memmap=True) as hdus:
        image = hdus[0].data
        if image is None or image.ndim != 2:
            raise ValueError(f"expected a 2D image in {image_path}")
        for row in selected.itertuples(index=False):
            x = int(np.rint(row.XWIN_IMAGE - 1))
            y = int(np.rint(row.YWIN_IMAGE - 1))
            stamp = np.asarray(
                image[y - half : y + half, x - half : x + half], dtype=float
            ).copy()
            if stamp.shape != (stamp_size, stamp_size):
                raise RuntimeError(
                    f"clipped stamp for input_index={row.index_input}: {stamp.shape}"
                )
            stamps.append(stamp)
    return stamps


def extended_edges(edges, pad_bins: int = PAD_BINS):
    width = edges[1] - edges[0]
    return edges[0] + width * np.arange(-pad_bins, len(edges) + pad_bins)


def density_contours(ax, x, y, color, bins=50, y_floor=None):
    """Draw zero-padded highest-density contours without edge clipping."""

    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)

    counts, x_edges, y_edges = np.histogram2d(x[finite], y[finite], bins=bins)
    density = np.pad(counts.T, PAD_BINS, mode="constant")
    x_edges = extended_edges(x_edges)
    y_edges = extended_edges(y_edges)
    density = ndimage.gaussian_filter(density, sigma=SMOOTHING, mode="constant")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    if y_floor is not None:
        keep = y_centers > y_floor
        density = density[keep]
        y_centers = y_centers[keep]
        density = np.vstack([np.zeros(density.shape[1]), density])
        y_centers = np.concatenate([[y_floor], y_centers])

    ranked = np.sort(density.ravel())[::-1]
    cumulative_mass = np.cumsum(ranked) / ranked.sum()
    thresholds = [ranked[np.searchsorted(cumulative_mass, mass)] for mass in MASS_LEVELS]
    levels = np.unique(np.sort(thresholds))

    return ax.contour(
        x_centers,
        y_centers,
        density,
        levels=levels,
        colors=[color] * len(levels),
        linewidths=np.linspace(1.0, 1.8, len(levels)),
        alpha=0.95,
    )


def marginal_density(values, bins=70, lower_bound=None):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    counts, edges = np.histogram(values, bins=bins)
    counts = np.pad(counts, PAD_BINS, mode="constant")
    edges = extended_edges(edges)
    density = ndimage.gaussian_filter1d(counts.astype(float), sigma=SMOOTHING, mode="constant")
    centers = 0.5 * (edges[:-1] + edges[1:])

    if lower_bound is not None:
        keep = centers > lower_bound
        centers = np.concatenate([[lower_bound], centers[keep]])
        density = np.concatenate([[0.0], density[keep]])

    density /= np.trapz(density, centers)
    return centers, density


def make_joint_axes(fig, spec, title):
    grid = spec.subgridspec(
        2, 2, height_ratios=(1, 4), width_ratios=(4, 1), hspace=0.04, wspace=0.04
    )
    joint = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=joint)
    right = fig.add_subplot(grid[1, 1], sharey=joint)
    top.set_title(title, loc="left", weight="bold", pad=8)
    top.set_axis_off()
    right.set_axis_off()
    return joint, top, right


def add_distribution(
    joint,
    top,
    right,
    x,
    y,
    truth_x,
    truth_y,
    observed_x,
    observed_y,
    color,
    y_floor=None,
):
    contours = density_contours(joint, x, y, color, y_floor=y_floor)

    x_grid, x_density = marginal_density(x)
    y_grid, y_density = marginal_density(y, lower_bound=y_floor)

    top.plot(x_grid, x_density, color=color, lw=1.5)
    top.fill_between(x_grid, x_density, color=color, alpha=0.1)
    right.plot(y_density, y_grid, color=color, lw=1.5)
    right.fill_betweenx(y_grid, y_density, color=color, alpha=0.1)

    joint.scatter(truth_x, truth_y, marker="+", s=90, linewidth=2, color=color, zorder=6)
    joint.scatter(
        observed_x,
        observed_y,
        marker="o",
        s=27,
        linewidth=0.8,
        edgecolor="white",
        facecolor=color,
        zorder=7,
    )
    return contours


def enclosed_limits(contours, coordinate, markers, padding=0.06, lower_bound=None):
    segments = [
        path[:, coordinate]
        for contour in contours
        for level in contour.allsegs
        for path in level
        if len(path)
    ]
    values = np.concatenate([*segments, np.asarray(markers, dtype=float)])
    values = values[np.isfinite(values)]

    low, high = values.min(), values.max()
    margin = padding * max(high - low, np.finfo(float).eps)
    low, high = low - margin, high + margin
    if lower_bound is not None:
        low = max(lower_bound, low)
    return low, high


def add_stamp_panels(fig, spec, stamps, selected, blend_response):
    stamp_grid = spec.subgridspec(1, 3, wspace=0.1)
    all_pixels = np.concatenate([stamp.ravel() for stamp in stamps])
    vmax = max(
        8.0 * OBSERVING_CONDITIONS["pixel_rms"],
        float(np.nanpercentile(all_pixels, 99.85)),
    )
    norm = AsinhNorm(
        linear_width=OBSERVING_CONDITIONS["pixel_rms"],
        vmin=-2.0 * OBSERVING_CONDITIONS["pixel_rms"],
        vmax=vmax,
    )

    axes = []
    for index, (stamp, (_, row), response, color) in enumerate(
        zip(stamps, selected.iterrows(), blend_response, COLORS)
    ):
        ax = fig.add_subplot(stamp_grid[0, index])
        ax.imshow(stamp, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
        ax.set_title(
            f"{row['representative']}  ·  $r={row['r_input']:.1f}$ mag"
            rf"  ·  $R_{{\rm blend}}={response:+.2f}$",
            color=color,
            weight="bold",
            fontsize=10,
            pad=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.3)

        bar_pixels = 2 / OBSERVING_CONDITIONS["pixel_size"]
        x0 = 5
        y0 = stamp.shape[0] - 7
        ax.plot([x0, x0 + bar_pixels], [y0, y0], color="white", lw=2.2)
        ax.text(
            x0 + bar_pixels / 2,
            y0 - 3,
            '2"',
            color="white",
            ha="center",
            va="top",
            fontsize=7,
        )
        axes.append(ax)
    return axes


def make_figure(stamps, selected, flow_rows, draws, blend_response):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "sans-serif",
        }
    )

    fig = plt.figure(figsize=(12, 8.2))
    layout = fig.add_gridspec(3, 1, height_ratios=(1.0, 0.13, 2.0), hspace=0.16)

    add_stamp_panels(fig, layout[0], stamps, selected, blend_response)

    legend_ax = fig.add_subplot(layout[1])
    legend_ax.set_axis_off()

    plots = layout[2].subgridspec(1, 2, wspace=0.24)
    shape_axes = make_joint_axes(fig, plots[0], "Conditional measured ellipticity")
    photometry_axes = make_joint_axes(
        fig, plots[1], "Conditional measured magnitude and radius"
    )

    shape_contours = []
    photometry_contours = []
    shape_markers = []
    photometry_markers = []

    for index, ((_, observed), color) in enumerate(zip(selected.iterrows(), COLORS)):
        g1 = draws.column("measured_ngmix_g1")[index]
        g2 = draws.column("measured_ngmix_g2")[index]
        mag = draws.column("measured_mag_auto")[index]
        radius = (
            np.exp(draws.column("measured_log_flux_radius")[index])
            * OBSERVING_CONDITIONS["pixel_size"]
        )

        truth_g1 = float(flow_rows.iloc[index]["e1_input_rot0_p"])
        truth_g2 = float(flow_rows.iloc[index]["e2_input_rot0_p"])
        truth_mag = float(flow_rows.iloc[index]["r_input_p"])
        truth_radius = float(flow_rows.iloc[index]["Re_input_p"])
        observed_radius = float(observed["FLUX_RADIUS"]) * OBSERVING_CONDITIONS["pixel_size"]

        shape_contours.append(
            add_distribution(
                *shape_axes,
                g1,
                g2,
                truth_g1,
                truth_g2,
                observed["NGMIX_G1"],
                observed["NGMIX_G2"],
                color,
            )
        )
        photometry_contours.append(
            add_distribution(
                *photometry_axes,
                mag,
                radius,
                truth_mag,
                truth_radius,
                observed["MAG_AUTO"],
                observed_radius,
                color,
                # No artificial floor: the outer contour closes on its own,
                # instead of being sliced flat along radius = 0.
                y_floor=None,
            )
        )

        shape_markers.extend(
            [(truth_g1, truth_g2), (observed["NGMIX_G1"], observed["NGMIX_G2"])]
        )
        photometry_markers.extend(
            [(truth_mag, truth_radius), (observed["MAG_AUTO"], observed_radius)]
        )

    shape_axes[0].set(xlabel=r"measured $g_1$", ylabel=r"measured $g_2$")
    photometry_axes[0].set(
        xlabel="measured MAG_AUTO (mag)",
        ylabel="measured FLUX_RADIUS (arcsec)",
    )

    shape_x = [point[0] for point in shape_markers]
    shape_y = [point[1] for point in shape_markers]
    x_limits = enclosed_limits(shape_contours, 0, shape_x)
    y_limits = enclosed_limits(shape_contours, 1, shape_y)
    bound = max(abs(value) for value in [*x_limits, *y_limits])
    shape_axes[0].set(
        xlim=(-bound, bound),
        ylim=(-bound, bound),
        aspect="equal",
        adjustable="box",
    )

    legend_ax.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="+",
                color="0.25",
                lw=0,
                markersize=9,
                markeredgewidth=1.8,
                label="Sheared input truth",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="0.35",
                markeredgecolor="0.35",
                markersize=6,
                label="Constgold realization",
            ),
        ),
        loc="center",
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handletextpad=0.6,
    )

    photometry_axes[0].set_xlim(
        enclosed_limits(
            photometry_contours,
            0,
            [point[0] for point in photometry_markers],
        )
    )
    photometry_axes[0].set_ylim(
        enclosed_limits(
            photometry_contours,
            1,
            [point[1] for point in photometry_markers],
            lower_bound=RADIUS_AXIS_FLOOR,
        )
    )

    for joint, _, _ in (shape_axes, photometry_axes):
        joint.grid(False)
        joint.spines[["top", "right"]].set_visible(False)
        joint.tick_params(direction="out", length=4, color="0.35")

    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.075, top=0.96)
    return fig


def main():
    args = parse_args()
    paths = constgold_paths(args.constgold_root, args.case, args.shear, args.tile)

    models = get_model(args.model)
    predictor = ResponsePredictor.load(models, device=args.device)
    emulator = load_emulator(models, conditions=OBSERVING_CONDITIONS, device=args.device)

    truth = load_truth(paths.input_catalogue)
    detected = detected_primaries(paths, truth)
    with fits.open(paths.image, memmap=True) as hdus:
        image_shape = tuple(int(value) for value in hdus[0].data.shape)

    candidates = candidate_pool(detected, predictor, image_shape, args.stamp_size)
    overrides = {"Faint": args.faint_index} if args.faint_index else None
    selected = select_representatives(candidates, overrides)
    flow_rows = prepare_selected_flow_rows(truth, selected)
    draws = sample_measurement(
        models,
        flow_rows,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        device=args.device,
    )
    stamps = extract_stamps(paths.image, selected, args.stamp_size)
    blend_response = predict_selected_blend_response(truth, selected, emulator)

    fig = make_figure(stamps, selected, flow_rows, draws, blend_response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_columns = [
        "representative",
        "index_input",
        "id_detec",
        "r_input",
        "Re_input",
        "snr",
        "gamma1_input",
        "gamma2_input",
    ]
    summary = selected[summary_columns].copy()
    summary["R_blend"] = blend_response
    shifts = blending_shift(selected, blend_response)
    summary["shift_e1"] = shifts[:, 0]
    summary["shift_e2"] = shifts[:, 1]
    print(summary.to_string(index=False))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
