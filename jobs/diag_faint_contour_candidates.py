"""Rank faint constgold candidates by how legible their flow contour is.

The README figure's faint example is picked for typical morphology, which says
nothing about whether its measured-ellipticity contour draws cleanly.  A very
low signal-to-noise galaxy has a broad, ragged conditional distribution, so the
highest-density contour breaks into lobes and wanders.

This scores every candidate in the faint magnitude band on the same density
estimate the figure uses: how many closed pieces the outer contour has, how
round it is, and how large it is.  It writes a contact sheet of the best ones so
the final choice is still made by eye.

    sbatch jobs/job_faint_contour_candidates.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_figure_module():
    """Import the figure script by path; examples/ is not an importable package."""

    path = REPOSITORY_ROOT / "examples" / "make_constgold_readme_figure.py"
    spec = importlib.util.spec_from_file_location("readme_figure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["readme_figure"] = module
    spec.loader.exec_module(module)
    return module


figure = load_figure_module()

import matplotlib.pyplot as plt  # noqa: E402  (after the module sets the Agg backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, default=figure.DEFAULT_CONSTGOLD_ROOT)
    parser.add_argument("--case", type=int, default=40)
    parser.add_argument("--shear", default="0.02")
    parser.add_argument("--tile", default=figure.TILE)
    parser.add_argument("--model", default="V3")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--stamp-size", type=int, default=72)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--n-candidates", type=int, default=48, help="most typical faint galaxies to sample"
    )
    parser.add_argument("--n-show", type=int, default=8, help="panels on the contact sheet")
    parser.add_argument(
        "--output",
        type=Path,
        # jobs/ is dev-only; examples/ is published to the public master tree.
        default=REPOSITORY_ROOT / "jobs" / "faint_contour_candidates.png",
    )
    return parser.parse_args()


def polygon_area(segment: np.ndarray) -> float:
    x, y = segment[:, 0], segment[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_perimeter(segment: np.ndarray) -> float:
    steps = np.diff(np.vstack([segment, segment[:1]]), axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def contour_metrics(g1: np.ndarray, g2: np.ndarray) -> dict:
    """Measure the figure's own contour: how many pieces, how round, how large.

    ``isoperimetric`` is 4*pi*area / perimeter^2, which is 1 for a circle and
    falls towards 0 as the outline grows ragged or stringy.
    """

    scratch = plt.figure()
    try:
        axis = scratch.add_subplot()
        contours = figure.density_contours(axis, g1, g2, "black")
        levels = contours.allsegs
        outer = [np.asarray(seg) for seg in levels[0] if len(seg) > 2]
        inner = [np.asarray(seg) for seg in levels[-1] if len(seg) > 2]
    finally:
        plt.close(scratch)

    if not outer or not inner:
        return {"pieces_outer": 0, "pieces_inner": 0, "isoperimetric": 0.0, "area": np.inf}

    area = sum(polygon_area(seg) for seg in outer)
    perimeter = sum(polygon_perimeter(seg) for seg in outer)
    inner_area = sum(polygon_area(seg) for seg in inner)
    inner_perimeter = sum(polygon_perimeter(seg) for seg in inner)
    return {
        "pieces_outer": len(outer),
        "pieces_inner": len(inner),
        "isoperimetric": 4 * np.pi * area / max(perimeter**2, 1e-12),
        "isoperimetric_inner": 4 * np.pi * inner_area / max(inner_perimeter**2, 1e-12),
        "area": area,
    }


def main():
    args = parse_args()
    paths = figure.constgold_paths(args.constgold_root, args.case, args.shear, args.tile)

    models = figure.get_model(args.model)
    predictor = figure.ResponsePredictor.load(models, device=args.device)
    truth = figure.load_truth(paths.input_catalogue)
    detected = figure.detected_primaries(paths, truth)

    with figure.fits.open(paths.image, memmap=True) as hdus:
        image_shape = tuple(int(value) for value in hdus[0].data.shape)

    emulator = figure.load_emulator(
        models, conditions=figure.OBSERVING_CONDITIONS, device=args.device
    )

    candidates = figure.candidate_pool(detected, predictor, image_shape, args.stamp_size)
    faint_quantile = figure.REPRESENTATIVE_QUANTILES[-1]
    band = figure.magnitude_band(candidates, faint_quantile)
    band["representative_score"] = figure.representative_score(band, faint_quantile)
    band = band.sort_values(["representative_score", "index_input"]).head(args.n_candidates)
    band = band.reset_index(drop=True)
    # The automatic pick is the top of this ordering, so it is always row 0.
    automatic_index = int(band.loc[0, "index_input"])
    print(f"faint band: {len(band)} candidates sampled, r = "
          f"{band['r_input'].min():.2f} to {band['r_input'].max():.2f}")

    flow_rows = figure.prepare_selected_flow_rows(truth, band)
    draws = figure.sample_measurement(
        models,
        flow_rows,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        device=args.device,
    )

    blend_response = figure.predict_selected_blend_response(truth, band, emulator)

    g1_all = draws.column("measured_ngmix_g1")
    g2_all = draws.column("measured_ngmix_g2")
    records = []
    for index in range(len(band)):
        metrics = contour_metrics(g1_all[index], g2_all[index])
        metrics["row"] = index
        metrics["index_input"] = int(band.loc[index, "index_input"])
        metrics["r_input"] = float(band.loc[index, "r_input"])
        metrics["snr"] = float(band.loc[index, "snr"])
        metrics["Re_input"] = float(band.loc[index, "Re_input"])
        metrics["spread"] = float(np.hypot(np.std(g1_all[index]), np.std(g2_all[index])))
        metrics["R_blend"] = float(blend_response[index])
        metrics["automatic"] = int(band.loc[index, "index_input"]) == automatic_index
        records.append(metrics)

    table = pd.DataFrame(records)
    # One closed, round, compact piece per level is what reads cleanly on the page.
    table["clean"] = (
        table["isoperimetric"]
        * table["isoperimetric_inner"]
        / (1 + 0.6 * (table["pieces_outer"] - 1 + table["pieces_inner"] - 1))
        / (1 + table["area"])
    )
    table = table.sort_values("clean", ascending=False).reset_index(drop=True)

    columns = [
        "index_input", "r_input", "snr", "Re_input", "spread", "R_blend",
        "pieces_outer", "pieces_inner", "isoperimetric", "isoperimetric_inner",
        "area", "clean", "automatic",
    ]
    table["rank"] = np.arange(1, len(table) + 1)
    print(table[["rank"] + columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    current = table.loc[table["automatic"]]
    if not current.empty:
        row = current.iloc[0]
        print(f"\ncurrent automatic pick {int(row['index_input'])} ranks "
              f"{int(row['rank'])} of {len(table)} on contour cleanliness "
              f"(area {row['area']:.3f} vs best {table['area'].min():.3f})")

    show = table.head(args.n_show)
    stamps = figure.extract_stamps(
        paths.image, band.loc[show["row"].to_numpy()], args.stamp_size
    )

    n = len(show)
    fig = plt.figure(figsize=(2.6 * n, 6.0))
    grid = fig.add_gridspec(2, n, height_ratios=(1, 1.15), hspace=0.18, wspace=0.16)
    for column, (_, row) in enumerate(show.iterrows()):
        index = int(row["row"])
        stamp_axis = fig.add_subplot(grid[0, column])
        stamp_axis.imshow(stamps[column], origin="lower", cmap="gray",
                          vmin=-2 * figure.OBSERVING_CONDITIONS["pixel_rms"],
                          vmax=8 * figure.OBSERVING_CONDITIONS["pixel_rms"],
                          interpolation="nearest")
        stamp_axis.set_xticks([])
        stamp_axis.set_yticks([])
        stamp_axis.set_title(
            f"input {int(row['index_input'])}\n"
            f"r={row['r_input']:.2f}  S/N={row['snr']:.1f}",
            fontsize=8,
        )

        contour_axis = fig.add_subplot(grid[1, column])
        figure.density_contours(contour_axis, g1_all[index], g2_all[index], "#228833")
        contour_axis.scatter(
            float(flow_rows.iloc[index]["e1_input_rot0_p"]),
            float(flow_rows.iloc[index]["e2_input_rot0_p"]),
            marker="+", s=80, linewidth=1.8, color="0.25", zorder=5,
        )
        contour_axis.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
        contour_axis.set_title(
            f"pieces {int(row['pieces_outer'])}/{int(row['pieces_inner'])}  "
            f"Q={row['isoperimetric']:.2f}  $R_b$={row['R_blend']:+.2f}",
            fontsize=8,
        )
        contour_axis.tick_params(labelsize=6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
