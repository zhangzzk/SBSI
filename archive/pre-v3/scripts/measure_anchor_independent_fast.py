"""Fast anchor-only measurement of independent-neighbour response scenes.

This uses the full SExtractor catalogues for the exact bright-neighbour and
crossmatch selection, but runs deterministic production ngmix fits only for
anchors detected in both antithetic legs.  It is an early cross-check; the
full all-source shape pipeline remains the authoritative measurement.
"""
from __future__ import annotations

import argparse
import os
import sys

from astropy.io import fits
from mpi4py import MPI
import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import shape, utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402
from build_anchorblend_independent_response import COND, input_frame  # noqa: E402
from measure_anchor_fixed_positions import (  # noqa: E402
    deterministic_ngmix, image_paths, subpixel_centre,
)


TILE = "tile180.0_-0.5"


def detected_anchors(base: str, case: int, sign: float,
                     target_ids: np.ndarray) -> pd.DataFrame:
    root = os.path.join(base, f"case{case}_{str(float(sign))}", "real0", "catalogues")
    detection = pd.read_feather(
        os.path.join(root, "SExtractor", f"{TILE}_bandr_rot0.feather")
    )
    match = pd.read_feather(os.path.join(root, "CrossMatch", f"{TILE}_rot0_matched.feather"))
    rejected = utils.remove_detection_w_bright_neighbour(
        detection.X_WORLD.array, detection.Y_WORLD.array,
        detection.FLUX_AUTO.array, ratio_max=5, r_min=0, r_max=3 / 3600,
    )
    retained = detection.drop(rejected)
    _, match_positions, _ = np.intersect1d(
        match.id_detec.to_numpy(int) - 1,
        retained.index.to_numpy(int), return_indices=True,
    )
    match = match.iloc[match_positions].reset_index(drop=True)
    common, target_positions, _ = np.intersect1d(
        match.id_input.to_numpy(np.int64), target_ids, return_indices=True,
    )
    selected = match.iloc[target_positions]
    det_index = selected.id_detec.to_numpy(int) - 1
    rows = detection.loc[det_index]
    output = pd.DataFrame({
        "input_index": common,
        "x_detect": rows.X_IMAGE.to_numpy(float),
        "y_detect": rows.Y_IMAGE.to_numpy(float),
    })
    if output.input_index.duplicated().any():
        raise RuntimeError(f"case {case} sign {sign}: duplicate matched anchor")
    return output


def measure_leg(base: str, case: int, sign: float, ids: np.ndarray,
                positions: pd.DataFrame, stamp_size: int, pixel_scale: float,
                leg_index: int) -> np.ndarray:
    science_path, psf_path = image_paths(base, case, sign)
    image = fits.getdata(science_path)
    psf_image = fits.getdata(psf_path)
    positions = positions.set_index("input_index", verify_integrity=True).loc[ids]
    measured = np.full((len(ids), 2), np.nan)
    for index, input_id in enumerate(ids):
        x = float(positions.x_detect.iloc[index])
        y = float(positions.y_detect.iloc[index])
        stamp = shape.cutout(image, x, y, stamp_size=stamp_size)
        centre = subpixel_centre(x, y, stamp_size, stamp.shape)
        seed = int((case * 1_000_003 + int(input_id) * 17 + leg_index * 97) % (2**32 - 1))
        try:
            measured[index] = deterministic_ngmix(
                stamp, psf_image, pixel_scale, centre, seed,
            )
        except Exception:
            measured[index] = np.nan
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, default=200)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--stamp-size", type=int, default=48)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    parser.add_argument(
        "--active-only", action="store_true",
        help="score only pairs whose secondary has nonzero applied shear",
    )
    args = parser.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", MPI.COMM_WORLD.Get_rank()))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
    all_ids = np.sort(manifest["index"].to_numpy(np.int64))
    plus_positions = detected_anchors(args.base, case, args.g, all_ids)
    minus_positions = detected_anchors(args.base, case, -args.g, all_ids)
    ids = np.intersect1d(plus_positions.input_index, minus_positions.input_index)
    if len(ids) < 0.8 * len(all_ids):
        raise RuntimeError(f"case {case}: both-leg coverage {len(ids)/len(all_ids):.2%}")
    plus = measure_leg(
        args.base, case, args.g, ids, plus_positions,
        args.stamp_size, args.pixel_scale, 0,
    )
    minus = measure_leg(
        args.base, case, -args.g, ids, minus_positions,
        args.stamp_size, args.pixel_scale, 1,
    )
    finite = np.isfinite(plus).all(axis=1) & np.isfinite(minus).all(axis=1)
    ids, plus, minus = ids[finite], plus[finite], minus[finite]
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = input_frame(args.base, case, args.g)
    pairs = predictor.predict_response(frame, frame)
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_p")
    )
    pairs = pairs[pairs[primary].isin(ids)].copy()
    pairs["u1"] = pairs.gamma1_input_s.to_numpy(float) / args.g
    pairs["u2"] = pairs.gamma2_input_s.to_numpy(float) / args.g
    norm = np.hypot(pairs.u1, pairs.u2)
    if args.active_only:
        pairs = pairs.loc[norm > 0.5].copy()
        norm = np.hypot(pairs.u1, pairs.u2)
    if not np.allclose(norm, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"case {case}: non-unit sheared pair direction")
    pairs["response_u1"] = pairs.response * pairs.u1
    pairs["response_u2"] = pairs.response * pairs.u2
    aggregate = pairs.groupby(primary, sort=False).agg(
        prediction=("response", "sum"), n_pairs=("response", "size"),
        sum_u1=("u1", "sum"), sum_u2=("u2", "sum"),
        prediction_u1=("response_u1", "sum"),
        prediction_u2=("response_u2", "sum"),
    )
    # Pandas can otherwise drop the left index name when the groupby key has a
    # different name; one case then serialized this column as plain ``index``.
    aggregate.index.name = "input_index"
    measured = pd.DataFrame({
        "input_index": ids,
        "measured_e1_plus": plus[:, 0], "measured_e2_plus": plus[:, 1],
        "measured_e1_minus": minus[:, 0], "measured_e2_minus": minus[:, 1],
    }).set_index("input_index")
    joined = measured.join(aggregate, how="inner").reset_index()
    de1 = joined.measured_e1_plus - joined.measured_e1_minus
    de2 = joined.measured_e2_plus - joined.measured_e2_minus
    joined["label_sum"] = (de1 * joined.sum_u1 + de2 * joined.sum_u2) / (2 * args.g)
    joined["null_sum"] = (-de1 * joined.sum_u2 + de2 * joined.sum_u1) / (2 * args.g)
    joined["projected_prediction"] = (
        joined.prediction_u1 * joined.sum_u1 + joined.prediction_u2 * joined.sum_u2
    )
    joined["desired_gap"] = joined.prediction - joined.label_sum
    joined["projected_gap"] = joined.projected_prediction - joined.label_sum
    joined["case"] = case
    joined.to_feather(output)
    print(
        f"case {case}: manifest={len(all_ids):,} both_detected={len(ids):,} "
        f"finite_paired={len(joined):,} pairs={len(pairs):,} "
        f"pred={joined.prediction.mean():+.6f} label={joined.label_sum.mean():+.6f} "
        f"projected={joined.projected_prediction.mean():+.6f}", flush=True,
    )
    print("ANCHOR_INDEPENDENT_FAST_CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
