#!/usr/bin/env python3
"""Show representative rendered stamps for high-response coherent anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import mad_std
from astropy.visualization import AsinhStretch, ImageNormalize, ManualInterval
from astropy.wcs import WCS
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


BLENDEMU_ROOT = Path("/home/z/Zekang.Zhang/blendemu")
if str(BLENDEMU_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDEMU_ROOT))

from blendemu import utils  # noqa: E402


TILE = "tile180.0_-0.5"
KEY = ["case", "input_index"]
FEATURE_COLUMNS = [
    *KEY,
    "scene_prediction",
    "R_blend_truth",
    "bias_truth_minus_model",
    "primary_mag",
    "log10_primary_size",
    "log10_primary_sersic_n",
    "log1p_n_pairs",
]


def select_quantile_case_diverse(
    anchors: pd.DataFrame,
    n_select: int,
    threshold: float,
) -> pd.DataFrame:
    """Select response-quantile representatives with distinct rendered cases."""
    required = set(FEATURE_COLUMNS)
    missing = required.difference(anchors.columns)
    if missing:
        raise KeyError(f"anchor table lacks columns: {sorted(missing)}")
    high = anchors.loc[anchors.scene_prediction > threshold].copy()
    if high.duplicated(KEY).any():
        raise ValueError("duplicate high-response anchor key")
    numeric = high[FEATURE_COLUMNS[2:]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("high-response anchor table contains non-finite values")
    if n_select <= 0 or high.case.nunique() < n_select:
        raise ValueError("not enough distinct cases for the requested sample")

    high = high.sort_values(
        ["scene_prediction", "case", "input_index"], kind="mergesort"
    ).reset_index(drop=True)
    response = high.scene_prediction.to_numpy(float)
    quantiles = (np.arange(n_select, dtype=float) + 0.5) / n_select
    targets = np.quantile(response, quantiles)
    used_cases: set[int] = set()
    chosen_indices: list[int] = []
    for target in targets:
        right = int(np.searchsorted(response, target, side="left"))
        left = right - 1
        chosen = None
        while left >= 0 or right < len(high):
            use_left = (
                right >= len(high)
                or (
                    left >= 0
                    and abs(response[left] - target)
                    <= abs(response[right] - target)
                )
            )
            candidate = left if use_left else right
            if use_left:
                left -= 1
            else:
                right += 1
            case = int(high.iloc[candidate].case)
            if case not in used_cases:
                chosen = candidate
                used_cases.add(case)
                break
        if chosen is None:
            raise RuntimeError("failed to find a case-diverse quantile representative")
        chosen_indices.append(chosen)

    selected = high.iloc[chosen_indices].copy().reset_index(drop=True)
    selected["target_quantile"] = quantiles
    selected["target_scene_prediction"] = targets
    lower = np.searchsorted(response, selected.scene_prediction, side="left")
    upper = np.searchsorted(response, selected.scene_prediction, side="right")
    selected["high_population_percentile"] = (lower + upper) / (2.0 * len(high))
    selected["n_pairs"] = np.rint(
        np.expm1(selected.log1p_n_pairs.to_numpy(float))
    ).astype(int)
    selected["primary_size"] = np.power(
        10.0, selected.log10_primary_size.to_numpy(float)
    )
    selected["primary_sersic_n"] = np.power(
        10.0, selected.log10_primary_sersic_n.to_numpy(float)
    )
    if selected.case.nunique() != n_select:
        raise RuntimeError("selected sample does not have one case per anchor")
    if not selected.scene_prediction.gt(threshold).all():
        raise RuntimeError("selected sample violates the response threshold")
    return selected


def locate_case_base(case: int, bases: list[Path]) -> Path:
    matches = [base for base in bases if (base / f"anchors_case{case}.feather").is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"case {case}: expected exactly one base directory, found {matches}"
        )
    return matches[0]


def square_grid_size(n_select: int) -> int:
    """Return the side length for a non-empty square gallery."""
    if n_select <= 0:
        raise ValueError("number of selected stamps must be positive")
    side = int(round(np.sqrt(n_select)))
    if side * side != n_select:
        raise ValueError("number of selected stamps must be a perfect square")
    return side


def matched_detection(
    base: Path,
    case: int,
    shear_leg: float,
    target_ids: np.ndarray,
) -> pd.DataFrame:
    """Replay the exact retained-detection match used for response measurement."""
    catalogue = (
        base / f"case{case}_{str(float(shear_leg))}" / "real0" / "catalogues"
    )
    shape_path = (
        catalogue / "Shapes" /
        f"shape_catalogue_detect_position_all_{TILE}.feather"
    )
    match_path = catalogue / "CrossMatch" / f"{TILE}_rot0_matched.feather"
    for path in (shape_path, match_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    shape_catalogue = pd.read_feather(
        shape_path,
        columns=["X_WORLD", "Y_WORLD", "FLUX_AUTO", "X_IMAGE", "Y_IMAGE"],
    )
    match = pd.read_feather(
        match_path,
        columns=["id_detec", "id_input", "distance_pixel_CM"],
    )
    rejected = utils.remove_detection_w_bright_neighbour(
        shape_catalogue.X_WORLD.array,
        shape_catalogue.Y_WORLD.array,
        shape_catalogue.FLUX_AUTO.array,
        ratio_max=5,
        r_min=0,
        r_max=3 / 3600,
    )
    retained_shape = shape_catalogue.drop(rejected)
    _, retained_positions, _ = np.intersect1d(
        match.id_detec.to_numpy(int) - 1,
        retained_shape.index.to_numpy(int),
        return_indices=True,
    )
    match = match.iloc[retained_positions].reset_index(drop=True)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    common, match_positions, _ = np.intersect1d(
        match.id_input.to_numpy(np.int64), target_ids, return_indices=True
    )
    selected_match = match.iloc[match_positions]
    detection_index = selected_match.id_detec.to_numpy(int) - 1
    detection = shape_catalogue.loc[detection_index]
    output = pd.DataFrame({
        "input_index": common,
        "x_detect_1indexed": detection.X_IMAGE.to_numpy(float),
        "y_detect_1indexed": detection.Y_IMAGE.to_numpy(float),
        "crossmatch_distance_pixels": (
            selected_match.distance_pixel_CM.to_numpy(float)
        ),
    })
    if (
        len(output) != len(target_ids)
        or output.input_index.duplicated().any()
        or not np.array_equal(np.sort(target_ids), output.input_index.to_numpy())
    ):
        raise RuntimeError(
            f"case {case} leg {shear_leg}: matched "
            f"{len(output)}/{len(target_ids)} retained response anchors"
        )
    return output


def exact_cutout(
    image: np.ndarray,
    x_1indexed: float,
    y_1indexed: float,
    stamp_size: int,
) -> tuple[np.ndarray, float, float]:
    """Replay BlendEMU's 1-indexed stamp convention and return local position."""
    if stamp_size <= 0 or stamp_size % 2:
        raise ValueError("stamp size must be a positive even integer")
    x0 = int(x_1indexed - stamp_size / 2.0) - 1
    x1 = int(x_1indexed + stamp_size / 2.0) - 1
    y0 = int(y_1indexed - stamp_size / 2.0) - 1
    y1 = int(y_1indexed + stamp_size / 2.0) - 1
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise RuntimeError("anchor cutout reaches beyond the rendered image")
    stamp = np.asarray(image[y0:y1, x0:x1], dtype=np.float32).copy()
    if stamp.shape != (stamp_size, stamp_size):
        raise RuntimeError(f"unexpected cutout shape {stamp.shape}")
    local_x = float(x_1indexed - 1.0 - x0)
    local_y = float(y_1indexed - 1.0 - y0)
    if not (0.0 <= local_x < stamp_size and 0.0 <= local_y < stamp_size):
        raise RuntimeError("truth position lies outside its cutout")
    return stamp, local_x, local_y


def local_detected_centroid(
    local_x_truth: float,
    local_y_truth: float,
    x_truth_1indexed: float,
    y_truth_1indexed: float,
    x_detect_1indexed: float,
    y_detect_1indexed: float,
    stamp_size: int,
) -> tuple[float, float]:
    """Transform a global detected centroid into a truth-centered cutout."""
    local_x = local_x_truth + x_detect_1indexed - x_truth_1indexed
    local_y = local_y_truth + y_detect_1indexed - y_truth_1indexed
    if not (0.0 <= local_x < stamp_size and 0.0 <= local_y < stamp_size):
        raise RuntimeError("detected centroid lies outside the truth-centered cutout")
    return float(local_x), float(local_y)


def border_values(stamp: np.ndarray, width: int = 5) -> np.ndarray:
    if stamp.ndim != 2 or min(stamp.shape) <= 2 * width:
        raise ValueError("stamp is too small for the requested border")
    mask = np.zeros(stamp.shape, dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return stamp[mask]


def load_stamps(
    selected: pd.DataFrame,
    bases: list[Path],
    shear_leg: float,
    stamp_size: int,
    pixel_scale: float,
    show_detected_centroid: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    stamps = []
    rows = []
    for sample_index, anchor in selected.iterrows():
        case = int(anchor.case)
        input_index = int(anchor.input_index)
        base = locate_case_base(case, bases)
        manifest_path = base / f"anchors_case{case}.feather"
        manifest = pd.read_feather(
            manifest_path, columns=["index", "RA", "DEC", "r", "Re"]
        )
        match = manifest.loc[manifest["index"].eq(input_index)]
        if len(match) != 1:
            raise RuntimeError(
                f"case {case} anchor {input_index}: manifest match count {len(match)}"
            )
        truth = match.iloc[0]
        detected = None
        if show_detected_centroid:
            detected_matches = matched_detection(
                base,
                case,
                shear_leg,
                np.asarray([input_index], dtype=np.int64),
            )
            detected = detected_matches.iloc[0]
        leg_label = str(float(shear_leg))
        image_path = (
            base / f"case{case}_{leg_label}" / "real0" / "images" /
            "original" / f"{TILE}_bandr_rot0.fits"
        )
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with fits.open(image_path, memmap=True, lazy_load_hdus=True) as hdus:
            if hdus[0].data is None or hdus[0].data.ndim != 2:
                raise RuntimeError(f"case {case}: science FITS is not a 2D image")
            world = WCS(hdus[0].header).celestial
            x, y = world.wcs_world2pix(
                np.asarray([float(truth.RA)]),
                np.asarray([float(truth.DEC)]),
                1,
            )
            x = float(x[0])
            y = float(y[0])
            stamp, local_x, local_y = exact_cutout(
                hdus[0].data, x, y, stamp_size
            )
            roundtrip_ra, roundtrip_dec = world.wcs_pix2world(
                np.asarray([x]), np.asarray([y]), 1
            )
        input_coord = SkyCoord(float(truth.RA) * u.deg, float(truth.DEC) * u.deg)
        roundtrip_coord = SkyCoord(roundtrip_ra[0] * u.deg, roundtrip_dec[0] * u.deg)
        separation_mas = float(input_coord.separation(roundtrip_coord).to_value(u.mas))
        if separation_mas > 1.0e-3:
            raise RuntimeError(
                f"case {case} anchor {input_index}: WCS roundtrip is {separation_mas} mas"
            )
        border = border_values(stamp)
        stamp_median = float(np.median(border))
        stamp_sigma = float(mad_std(border, ignore_nan=True))
        if not np.isfinite(stamp).all() or not np.isfinite(stamp_sigma) or stamp_sigma <= 0.0:
            raise RuntimeError(f"case {case} anchor {input_index}: invalid stamp")
        stamps.append(stamp)
        row = anchor.to_dict()
        row.update({
            "sample_index": int(sample_index),
            "RA": float(truth.RA),
            "DEC": float(truth.DEC),
            "x_truth_1indexed": x,
            "y_truth_1indexed": y,
            "local_x_truth_0indexed": local_x,
            "local_y_truth_0indexed": local_y,
            "wcs_roundtrip_mas": separation_mas,
            "stamp_border_median": stamp_median,
            "stamp_border_mad_sigma": stamp_sigma,
            "image_path": str(image_path.resolve()),
        })
        if detected is not None:
            x_detect = float(detected.x_detect_1indexed)
            y_detect = float(detected.y_detect_1indexed)
            local_x_detect, local_y_detect = local_detected_centroid(
                local_x,
                local_y,
                x,
                y,
                x_detect,
                y_detect,
                stamp_size,
            )
            offset_pixels = float(np.hypot(x_detect - x, y_detect - y))
            stored_offset = float(detected.crossmatch_distance_pixels)
            offset_difference = abs(offset_pixels - stored_offset)
            if offset_difference > 0.02:
                raise RuntimeError(
                    f"case {case} anchor {input_index}: WCS/detection offset "
                    f"{offset_pixels:.4f} differs from crossmatch distance "
                    f"{stored_offset:.4f} pixels"
                )
            row.update({
                "x_detect_1indexed": x_detect,
                "y_detect_1indexed": y_detect,
                "local_x_detect_0indexed": local_x_detect,
                "local_y_detect_0indexed": local_y_detect,
                "centroid_offset_pixels": offset_pixels,
                "centroid_offset_arcsec": offset_pixels * pixel_scale,
                "crossmatch_distance_pixels": stored_offset,
                "centroid_vs_crossmatch_difference_pixels": offset_difference,
            })
        rows.append(row)
        print(
            f"loaded {sample_index + 1}/{len(selected)}: "
            f"case={case} anchor={input_index} "
            f"R_scene={anchor.scene_prediction:.6f}",
            flush=True,
        )
    return np.stack(stamps), pd.DataFrame(rows)


def common_display(
    stamps: np.ndarray,
    sample: pd.DataFrame,
) -> tuple[np.ndarray, float, float, float]:
    medians = sample.stamp_border_median.to_numpy(float)
    centered = stamps.astype(np.float64) - medians[:, None, None]
    noise = float(np.median(sample.stamp_border_mad_sigma.to_numpy(float)))
    positive_tail = float(np.quantile(centered, 0.998))
    vmin = -2.5 * noise
    vmax = max(12.0 * noise, min(positive_tail, 60.0 * noise))
    if not np.isfinite(centered).all() or not vmin < vmax:
        raise RuntimeError("invalid common display scale")
    return centered.astype(np.float32), noise, vmin, vmax


def plot_grid(
    centered: np.ndarray,
    sample: pd.DataFrame,
    pixel_scale: float,
    noise: float,
    vmin: float,
    vmax: float,
    output_prefix: Path,
    show_detected_centroid: bool,
) -> None:
    if centered.shape[0] != len(sample):
        raise ValueError("stamp array and sample table have different lengths")
    grid_size = square_grid_size(len(sample))
    if show_detected_centroid:
        required = {
            "local_x_detect_0indexed",
            "local_y_detect_0indexed",
            "centroid_offset_arcsec",
        }
        if not required.issubset(sample.columns):
            raise ValueError("detected-centroid columns are missing")
    panel_title_size = 6.2 if grid_size <= 6 else 5.6
    figure_width = 1.92 * grid_size
    figure_height = 2.10 * grid_size
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.0,
        "axes.titlesize": panel_title_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    norm = ImageNormalize(
        interval=ManualInterval(vmin=vmin, vmax=vmax),
        stretch=AsinhStretch(a=0.035),
        clip=True,
    )
    figure, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    for index, (axis, stamp) in enumerate(zip(axes.flat, centered)):
        row = sample.iloc[index]
        axis.imshow(
            stamp,
            origin="lower",
            cmap="gray",
            norm=norm,
            interpolation="nearest",
        )
        axis.plot(
            row.local_x_truth_0indexed,
            row.local_y_truth_0indexed,
            marker="+",
            color="#00BFC4",
            markersize=5.0,
            markeredgewidth=0.85,
            linestyle="none",
            zorder=5,
        )
        if show_detected_centroid:
            axis.plot(
                row.local_x_detect_0indexed,
                row.local_y_detect_0indexed,
                marker="x",
                color="#D55E00",
                markersize=5.2,
                markeredgewidth=0.95,
                linestyle="none",
                zorder=6,
            )
            offset_text = rf'  $\Delta$={row.centroid_offset_arcsec:.2f}"'
        else:
            offset_text = ""
        axis.set_title(
            f"c{int(row.case)}  i{int(row.input_index)}\n"
            f"p={row.scene_prediction:.3f}  t={row.R_blend_truth:+.2f}\n"
            f"r={row.primary_mag:.1f}  N={int(row.n_pairs)}{offset_text}",
            pad=1.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("0.25")
        if index == 0:
            bar_pixels = 2.0 / pixel_scale
            x0 = 2.5
            y0 = 3.0
            axis.plot(
                [x0, x0 + bar_pixels], [y0, y0], color="white",
                linewidth=1.5, solid_capstyle="butt",
            )
            axis.text(
                x0 + bar_pixels / 2.0, y0 + 1.5, '2"', color="white",
                ha="center", va="bottom", fontsize=5.5,
            )
    figure.suptitle(
        r"Representative coherent anchors with raw V2.2 $R_{\rm scene}>0.1$",
        fontsize=12.0,
        y=0.994,
    )
    figure.text(
        0.5,
        0.972,
        (
            rf"{len(sample)} response-quantile representatives (one case each), sorted by "
            r"$p=R_{\rm scene}$; "
            r"$t=R_{\rm blend}^{\rm coherent}$, $r$=primary magnitude, $N$=deployed neighbours"
        ),
        ha="center",
        va="top",
        fontsize=7.3,
    )
    marker_text = "cyan +=input truth"
    if show_detected_centroid:
        marker_text += r'; orange $\times$=retained SExtractor centroid; $\Delta$=separation'
    figure.text(
        0.5,
        0.012,
        (
            f'Truth-centered +0.02 coherent renders; {centered.shape[1]}×'
            f'{centered.shape[2]} pixels = {centered.shape[1] * pixel_scale:.1f}"; '
            f"{marker_text}; shared asinh display [{vmin/noise:.1f}, "
            f"{vmax/noise:.1f}]×median border noise"
        ),
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.988,
        bottom=0.037,
        top=0.935,
        wspace=0.055,
        hspace=0.43,
    )
    figure.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    figure.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    rows = payload["sample"]
    lines = [
        "# High-response coherent-anchor stamp gallery",
        "",
        f"Selected `{len(rows)}` anchors from the `{payload['population']['n_high']:,}` "
        f"held-out c700--899 anchors satisfying raw V2.2 scene prediction "
        f"`> {payload['selection']['threshold']}`. The sample is deterministic: "
        "one nearest anchor at each equally spaced response quantile, requiring "
        "a distinct rendered case for every stamp.",
        "",
        f"Stamps are {payload['stamp']['stamp_size_pixels']}x"
        f"{payload['stamp']['stamp_size_pixels']}-pixel "
        f"({payload['stamp']['width_arcsec']:.1f}-arcsec) truth-centered cutouts "
        "from the real +0.02 coherent simulation images. The cyan plus marks the "
        "input truth centroid. "
        + (
            "The orange cross marks the retained SExtractor centroid used by the "
            "coherent-response measurement; `Delta` is their angular separation. "
            if payload['stamp']['detected_centroid_overlay'] else ""
        )
        + "Every panel uses the same background-subtracted asinh display scale.",
        "",
        f"Selected scene predictions span `{payload['selection']['selected_min']:.6f}` "
        f"to `{payload['selection']['selected_max']:.6f}`; their primary magnitudes "
        f"span `{payload['selection']['primary_mag_min']:.3f}` to "
        f"`{payload['selection']['primary_mag_max']:.3f}`.",
        "",
        "Per-panel notation: `p` is raw V2.2 scene prediction, `t` is the noisy "
        "per-anchor coherent response label, `r` is primary magnitude, and `N` "
        "is the number of deployed neighbours.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--case-min", type=int, default=700)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--n-select", type=int, default=36)
    parser.add_argument("--shear-leg", type=float, default=0.02)
    parser.add_argument("--stamp-size", type=int, default=48)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    parser.add_argument("--show-detected-centroid", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if (
        args.case_min > args.case_max
        or args.n_select <= 0
        or args.threshold < 0.0
        or args.stamp_size <= 0
        or args.pixel_scale <= 0.0
    ):
        raise ValueError("invalid case window, selection, stamp, or pixel scale")
    square_grid_size(args.n_select)
    features_path = Path(args.features)
    bases = [Path(value) for value in args.base]
    for path in [features_path, *bases]:
        if not path.exists():
            raise FileNotFoundError(path)
    output_prefix = Path(args.output_prefix)
    suffixes = ("csv", "json", "md", "npz", "pdf", "png")
    existing = [
        str(Path(f"{output_prefix}.{suffix}")) for suffix in suffixes
        if Path(f"{output_prefix}.{suffix}").exists()
    ]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    anchors = pd.read_feather(features_path, columns=FEATURE_COLUMNS)
    anchors = anchors.loc[anchors.case.between(args.case_min, args.case_max)].copy()
    selected = select_quantile_case_diverse(
        anchors, args.n_select, args.threshold
    )
    stamps, sample = load_stamps(
        selected,
        bases,
        args.shear_leg,
        args.stamp_size,
        args.pixel_scale,
        args.show_detected_centroid,
    )
    centered, noise, vmin, vmax = common_display(stamps, sample)
    plot_grid(
        centered,
        sample,
        args.pixel_scale,
        noise,
        vmin,
        vmax,
        output_prefix,
        args.show_detected_centroid,
    )
    sample.to_csv(f"{output_prefix}.csv", index=False)
    array_payload = {
        "raw_stamps": stamps,
        "centered_stamps": centered,
        "local_x_truth": sample.local_x_truth_0indexed.to_numpy(float),
        "local_y_truth": sample.local_y_truth_0indexed.to_numpy(float),
    }
    if args.show_detected_centroid:
        array_payload.update({
            "local_x_detect": sample.local_x_detect_0indexed.to_numpy(float),
            "local_y_detect": sample.local_y_detect_0indexed.to_numpy(float),
            "centroid_offset_arcsec": sample.centroid_offset_arcsec.to_numpy(float),
        })
    np.savez_compressed(f"{output_prefix}.npz", **array_payload)
    high = anchors.loc[anchors.scene_prediction > args.threshold]
    payload = {
        "title": "Representative high-response coherent-anchor stamps",
        "sources": {
            "features": str(features_path.resolve()),
            "render_directories": [str(path.resolve()) for path in bases],
            "shear_leg": float(args.shear_leg),
            "constgold_opened": False,
        },
        "population": {
            "case_window": [int(args.case_min), int(args.case_max)],
            "n_cases": int(anchors.case.nunique()),
            "n_anchors": int(len(anchors)),
            "n_high": int(len(high)),
        },
        "selection": {
            "feature": "scene_prediction",
            "operator": ">",
            "threshold": float(args.threshold),
            "method": (
                "equally spaced response quantiles; nearest anchor at each "
                "quantile with one distinct rendered case per stamp"
            ),
            "n_selected": int(len(sample)),
            "selected_min": float(sample.scene_prediction.min()),
            "selected_max": float(sample.scene_prediction.max()),
            "primary_mag_min": float(sample.primary_mag.min()),
            "primary_mag_max": float(sample.primary_mag.max()),
        },
        "stamp": {
            "centering": "input truth WCS position",
            "stamp_size_pixels": int(args.stamp_size),
            "pixel_scale_arcsec": float(args.pixel_scale),
            "width_arcsec": float(args.stamp_size * args.pixel_scale),
            "display": "per-stamp border median subtraction; one shared asinh scale",
            "median_border_mad_sigma": float(noise),
            "display_vmin": float(vmin),
            "display_vmax": float(vmax),
            "display_vmin_sigma": float(vmin / noise),
            "display_vmax_sigma": float(vmax / noise),
            "maximum_wcs_roundtrip_mas": float(sample.wcs_roundtrip_mas.max()),
            "detected_centroid_overlay": bool(args.show_detected_centroid),
        },
        "sample": json_clean(sample.to_dict(orient="records")),
    }
    if args.show_detected_centroid:
        offsets = sample.centroid_offset_arcsec.to_numpy(float)
        payload["stamp"]["detected_centroid_definition"] = (
            "SExtractor X_IMAGE/Y_IMAGE after the exact response-measurement "
            "bright-neighbour rejection and input crossmatch"
        )
        payload["stamp"]["centroid_offset_arcsec"] = {
            "minimum": float(np.min(offsets)),
            "median": float(np.median(offsets)),
            "p84": float(np.quantile(offsets, 0.84)),
            "p95": float(np.quantile(offsets, 0.95)),
            "maximum": float(np.max(offsets)),
        }
        payload["stamp"]["maximum_centroid_vs_crossmatch_difference_pixels"] = (
            float(sample.centroid_vs_crossmatch_difference_pixels.max())
        )
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    print(f"saved coherent-anchor stamp gallery to {output_prefix}.*", flush=True)
    print("ANCHOR_HIGH_RESPONSE_STAMPS_DONE", flush=True)


if __name__ == "__main__":
    main()
