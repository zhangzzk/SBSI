"""Compute an Eq.-17-style noiseless image-overlap purity for q3 anchors.

For anchor ``i`` the metric is

    rho_i = sum_p s_ip**2 / sum_p (s_ip * sum_j s_jp),

where ``s`` is the unsheared, PSF-convolved intrinsic r-band model on the
simulation pixel grid.  The Eq. 17 mask retains anchor pixels above five per
cent of the sky RMS.  Every retained local-scene source is included in the
denominator; the anchor itself is included as required by the definition.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import galsim
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FIELD_COLUMNS = [
    "index", "RA", "DEC", "position_angle", "Re", "axis_ratio", "sersic_n", "r",
]
ZERO_POINT = 30.0
N_MIN, N_MAX = 0.3, 6.2
Q_MIN, Q_MAX = 0.05, 1.0


def galaxy_profile(row, psf: galsim.GSObject) -> galsim.GSObject:
    """Reproduce the single-Sersic branch used by MultiBand_ImSim."""
    n = float(np.clip(row["sersic_n"], N_MIN, N_MAX))
    q = float(np.clip(row["axis_ratio"], Q_MIN, Q_MAX))
    circularized_re = float(row["Re"]) * np.sqrt(q)
    flux = 10.0 ** (-0.4 * (float(row["r"]) - ZERO_POINT))
    galaxy = galsim.Sersic(
        n=n, half_light_radius=circularized_re, flux=flux, trunc=0,
    ).shear(q=q, beta=float(row["position_angle"]) * galsim.degrees)
    return galsim.Convolve(galaxy, psf)


def draw_profile(profile: galsim.GSObject, stamp_size: int, pixel_scale: float,
                 dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    image = galsim.ImageF(stamp_size, stamp_size, scale=pixel_scale)
    profile.shift(dx, dy).drawImage(image=image, method="auto")
    array = image.array.astype(np.float64, copy=False)
    # True surface brightness is non-negative.  FFT drawing can leave ringing
    # at ~1e-7 of the peak in distant wings, which otherwise permits rho>1 for
    # almost isolated sources.  Removing it is a physical positivity constraint.
    np.maximum(array, 0.0, out=array)
    return array


def overlap_purity(anchor_row, neighbour_rows: pd.DataFrame, *, stamp_size: int,
                   pixel_scale: float, psf: galsim.GSObject,
                   threshold: float) -> dict:
    anchor_profile = galaxy_profile(anchor_row, psf)
    anchor = draw_profile(anchor_profile, stamp_size, pixel_scale)
    total = anchor.copy()
    cosdec = np.cos(np.deg2rad(float(anchor_row["DEC"])))
    for _, neighbour in neighbour_rows.iterrows():
        dx = (float(neighbour["RA"]) - float(anchor_row["RA"])) * cosdec * 3600.0
        dy = (float(neighbour["DEC"]) - float(anchor_row["DEC"])) * 3600.0
        total += draw_profile(
            galaxy_profile(neighbour, psf), stamp_size, pixel_scale, dx=dx, dy=dy,
        )

    mask = anchor >= threshold
    if not mask.any():
        raise RuntimeError("anchor has no pixels above the Eq. 17 threshold")
    numerator = float(np.sum(anchor[mask] ** 2, dtype=np.float64))
    denominator = float(np.sum(anchor[mask] * total[mask], dtype=np.float64))
    full_numerator = float(np.sum(anchor ** 2, dtype=np.float64))
    full_denominator = float(np.sum(anchor * total, dtype=np.float64))
    purity = numerator / denominator
    purity_full = full_numerator / full_denominator
    tolerance = 1.0e-10
    if not (0.0 < purity <= 1.0 + tolerance) or not (0.0 < purity_full <= 1.0 + tolerance):
        raise RuntimeError(f"invalid purity rho={purity}, unthresholded={purity_full}")
    purity = min(purity, 1.0)
    purity_full = min(purity_full, 1.0)
    edge = np.zeros_like(mask)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {
        "purity_eq17": purity,
        "true_blendedness_eq17": 1.0 - purity,
        "purity_unthresholded": purity_full,
        "true_blendedness_unthresholded": 1.0 - purity_full,
        "n_mask_pixels": int(mask.sum()),
        "mask_touches_stamp_edge": bool(np.any(mask & edge)),
        "anchor_mask_flux": float(anchor[mask].sum(dtype=np.float64)),
        "n_local_neighbours": int(len(neighbour_rows)),
    }


def process_case(task: tuple) -> tuple[pd.DataFrame, dict]:
    (case, base, anchor_ids, stamp_size, max_stamp_size, pixel_scale, neighbour_radius,
     sky_rms, threshold_fraction, psf_fwhm, psf_beta) = task
    path = os.path.join(base, f"gals{case}_0.02.feather")
    field = pd.read_feather(path, columns=FIELD_COLUMNS)
    if field["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate source index")
    by_id = field.set_index("index", drop=False, verify_integrity=True)
    missing = np.setdiff1d(anchor_ids, by_id.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"case {case}: {len(missing)} truth anchors missing from field")
    dec0 = float(np.median(field["DEC"]))
    xy = np.column_stack([
        field["RA"].to_numpy(float) * np.cos(np.deg2rad(dec0)) * 3600.0,
        field["DEC"].to_numpy(float) * 3600.0,
    ])
    tree = cKDTree(xy)
    row_by_id = pd.Series(np.arange(len(field), dtype=np.int64), index=field["index"])
    psf = galsim.Moffat(
        beta=psf_beta, fwhm=psf_fwhm, trunc=4.5 * psf_fwhm,
    )
    threshold = threshold_fraction * sky_rms
    rows = []
    for anchor_id in anchor_ids:
        anchor_position = int(row_by_id.loc[int(anchor_id)])
        local_positions = np.asarray(
            tree.query_ball_point(xy[anchor_position], r=neighbour_radius), dtype=np.int64,
        )
        local_positions = local_positions[local_positions != anchor_position]
        current_stamp_size = stamp_size
        while True:
            stats = overlap_purity(
                field.iloc[anchor_position], field.iloc[local_positions],
                stamp_size=current_stamp_size, pixel_scale=pixel_scale, psf=psf,
                threshold=threshold,
            )
            if not stats["mask_touches_stamp_edge"]:
                break
            if current_stamp_size >= max_stamp_size:
                raise RuntimeError(
                    f"case {case} anchor {anchor_id}: Eq. 17 mask reaches "
                    f"maximum {max_stamp_size}-pixel stamp edge"
                )
            current_stamp_size = min(2 * current_stamp_size, max_stamp_size)
        stats["stamp_size_used"] = current_stamp_size
        rows.append({"case": case, "input_index": int(anchor_id), **stats})
    out = pd.DataFrame(rows)
    summary = {
        "case": case, "n_field_sources": len(field), "n_anchors": len(out),
        "mean_purity_eq17": float(out["purity_eq17"].mean()),
        "median_purity_eq17": float(out["purity_eq17"].median()),
        "edge_touch_count": int(out["mask_touches_stamp_edge"].sum()),
        "mean_local_neighbours": float(out["n_local_neighbours"].mean()),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pilot-base", required=True)
    parser.add_argument("--main-base", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-anchors-per-case", type=int, default=0)
    parser.add_argument("--stamp-size", type=int, default=96)
    parser.add_argument("--max-stamp-size", type=int, default=384)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    parser.add_argument("--neighbour-radius", type=float, default=15.0)
    parser.add_argument("--sky-rms", type=float, default=0.311519)
    parser.add_argument("--threshold-fraction", type=float, default=0.05)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--psf-beta", type=float, default=2.224068)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.stamp_size % 2 or args.max_stamp_size % 2:
        raise ValueError("stamp sizes must be even")
    if args.max_stamp_size < args.stamp_size:
        raise ValueError("maximum stamp size is smaller than initial stamp size")
    truth = pd.read_feather(args.truth, columns=["case", "input_index"])
    truth = truth.loc[truth["case"].isin(args.cases)].copy()
    if truth.empty or truth["case"].nunique() != len(set(args.cases)):
        raise RuntimeError("truth does not cover every requested case")
    if truth.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate truth key")
    tasks = []
    for case in args.cases:
        anchor_ids = truth.loc[truth["case"] == case, "input_index"].to_numpy(np.int64)
        if args.max_anchors_per_case:
            anchor_ids = anchor_ids[:args.max_anchors_per_case]
        base = args.pilot_base if case < 48 else args.main_base
        tasks.append((
            case, base, anchor_ids, args.stamp_size, args.max_stamp_size, args.pixel_scale,
            args.neighbour_radius, args.sky_rms, args.threshold_fraction,
            args.psf_fwhm, args.psf_beta,
        ))
    parts, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_case, task): task[0] for task in tasks}
        for future in as_completed(futures):
            case = futures[future]
            part, summary = future.result()
            parts.append(part)
            summaries.append(summary)
            print(
                f"case {case}: anchors={len(part):,}, mean rho={summary['mean_purity_eq17']:.5f}, "
                f"edge masks={summary['edge_touch_count']}", flush=True,
            )
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["case", "input_index"], kind="mergesort",
    ).reset_index(drop=True)
    if out.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate output key")
    expected = sum(len(task[2]) for task in tasks)
    if len(out) != expected:
        raise RuntimeError(f"expected {expected} output rows, got {len(out)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "x", encoding="utf-8") as handle:
        json.dump({
            "definition": "Nourbakhsh et al. 2022 Eq. 17 operationalized on SBSI scenes",
            "formula": "sum_p(s_i^2) / sum_p(s_i * sum_j(s_j))",
            "profile": "unsheared single-Sersic intrinsic model convolved with simulation Moffat PSF",
            "render_positivity": "negative FFT ringing clipped to zero per source profile",
            "mask": "anchor noiseless pixel signal >= threshold_fraction * sky_rms",
            "truth": os.path.abspath(args.truth), "cases": sorted(args.cases),
            "n_rows": len(out), "n_cases": int(out["case"].nunique()),
            "initial_stamp_size": args.stamp_size,
            "max_stamp_size": args.max_stamp_size, "pixel_scale": args.pixel_scale,
            "neighbour_radius": args.neighbour_radius, "sky_rms": args.sky_rms,
            "threshold_fraction": args.threshold_fraction,
            "psf_fwhm": args.psf_fwhm, "psf_beta": args.psf_beta,
            "case_summary": sorted(summaries, key=lambda item: item["case"]),
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} and {sidecar}")
    print("V22_Q3_PURITY_DONE", flush=True)


if __name__ == "__main__":
    main()
