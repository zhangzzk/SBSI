"""Remeasure coherent anchors at detected and fixed truth positions.

One MPI rank handles one already-rendered case.  For every anchor present in
the stored response population, ngmix is rerun on the same image at (a) the
per-leg SExtractor centroid used in production and (b) the fixed input truth
position.  Both fits receive the same deterministic initialization seed, so
their paired difference isolates position/matching motion rather than fitter
randomness.  No images are rendered and constgold is not read.
"""
from __future__ import annotations

import argparse
import os
import sys

import galsim
from astropy.io import fits
from astropy import wcs
from mpi4py import MPI
import ngmix
import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import shape, utils  # noqa: E402


TILE = "tile180.0_-0.5"


def deterministic_ngmix(stamp: np.ndarray, psf_image: np.ndarray, scale: float,
                        centre: tuple[float, float] | None, seed: int) -> np.ndarray:
    """Exact production Gaussian fit with an explicitly seeded initializer."""
    observation = shape.make_obs(stamp, psf_image, scale, gal_cen=centre)
    rng = np.random.RandomState(seed)
    image = galsim.Image(observation.image, scale=observation.jacobian.scale)
    try:
        hsm_shape = galsim.hsm.FindAdaptiveMom(image, strict=False)
    except Exception:
        hsm_shape = galsim.hsm.FindAdaptiveMom(
            image, guess_sig=10.0,
            hsmparams=galsim.hsm.HSMParams(max_mom2_iter=1000), strict=False,
        )
    t_guess = (
        1.0 if hsm_shape.error_message != ""
        else 2.0 * (hsm_shape.moments_sigma * observation.jacobian.scale) ** 2
    )
    prior = shape._get_prior(rng, scale=observation.jacobian.scale)
    fitter = ngmix.fitting.Fitter(model="gauss", prior=prior)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=t_guess, prior=prior,
    )
    psf_runner = ngmix.runners.PSFRunner(
        fitter=ngmix.em.EMFitter(),
        guesser=ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1), ntry=2,
    )
    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=2)
    result = ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner).go(observation)
    return np.asarray(result["g"], dtype=float)


def subpixel_centre(x: float, y: float, stamp_size: int,
                    stamp_shape: tuple[int, int]) -> tuple[float, float] | None:
    if stamp_shape != (stamp_size, stamp_size):
        return None
    return (
        y - int(y - stamp_size / 2.0),
        x - int(x - stamp_size / 2.0),
    )


def matched_detection(base: str, case: int, sign: float,
                      target_ids: np.ndarray) -> pd.DataFrame:
    catalogue = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues",
    )
    shape_path = os.path.join(
        catalogue, "Shapes", f"shape_catalogue_detect_position_all_{TILE}.feather",
    )
    match_path = os.path.join(catalogue, "CrossMatch", f"{TILE}_rot0_matched.feather")
    shape_catalogue = pd.read_feather(shape_path)
    match = pd.read_feather(match_path)
    rejected = utils.remove_detection_w_bright_neighbour(
        shape_catalogue.X_WORLD.array, shape_catalogue.Y_WORLD.array,
        shape_catalogue.FLUX_AUTO.array, ratio_max=5, r_min=0, r_max=3 / 3600,
    )
    retained_shape = shape_catalogue.drop(rejected)
    _, positions, _ = np.intersect1d(
        match.id_detec.to_numpy(int) - 1,
        retained_shape.index.to_numpy(int), return_indices=True,
    )
    match = match.iloc[positions].reset_index(drop=True)
    common, match_positions, _ = np.intersect1d(
        match.id_input.to_numpy(np.int64), target_ids, return_indices=True,
    )
    selected_match = match.iloc[match_positions]
    detection_index = selected_match.id_detec.to_numpy(int) - 1
    detection = shape_catalogue.loc[detection_index]
    out = pd.DataFrame({
        "input_index": common,
        "x_detect": detection.X_IMAGE.to_numpy(float),
        "y_detect": detection.Y_IMAGE.to_numpy(float),
    })
    if len(out) != len(target_ids) or not np.array_equal(np.sort(target_ids), out.input_index):
        raise RuntimeError(
            f"case {case} sign {sign}: matched {len(out)}/{len(target_ids)} response anchors"
        )
    return out


def image_paths(base: str, case: int, sign: float) -> tuple[str, str]:
    root = os.path.join(base, f"case{case}_{str(float(sign))}", "real0")
    science = os.path.join(root, "images", "original", "-BACKGROUND", f"{TILE}_bandr_rot0.fits")
    if not os.path.isfile(science):
        science = os.path.join(root, "images", "original", f"{TILE}_bandr_rot0.fits")
    psf = os.path.join(root, "images", "original", f"psf_{TILE}_bandr", "psf_ima.fits")
    return science, psf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, default=200)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--stamp-size", type=int, default=48)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    args = parser.parse_args()

    # On this cluster, srun can launch mpi4py as independent singleton worlds
    # (every process then reports rank zero).  SLURM_PROCID is the authoritative
    # task index in that launch mode.  Retain MPI rank for ordinary mpiexec use.
    rank = int(os.environ.get("SLURM_PROCID", MPI.COMM_WORLD.Get_rank()))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    os.makedirs(args.output_dir, exist_ok=True)

    response = pd.read_feather(args.response)
    # One legacy independent-response case serialized its reset-index key in
    # ``index`` and left ``input_index`` null.  Normalize that representation
    # without changing any valid IDs.  Coherent response tables do not need the
    # fallback, but accepting it keeps this measurement routine data-generic.
    if "index" in response:
        if "input_index" not in response:
            response["input_index"] = response["index"]
        else:
            missing = response["input_index"].isna()
            if np.any(missing & response["index"].isna()):
                raise RuntimeError("missing input_index has no legacy index fallback")
            response.loc[missing, "input_index"] = response.loc[missing, "index"]
    if "input_index" not in response:
        raise RuntimeError("response table has no input_index key")
    case_ids = response.loc[response.case == case, "input_index"]
    if case_ids.isna().any():
        raise RuntimeError(f"case {case}: response contains null input_index")
    ids = np.sort(case_ids.to_numpy(np.int64))
    if not len(ids):
        raise RuntimeError(f"case {case}: no stored response anchors")
    manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
    truth = manifest.set_index("index", verify_integrity=True).loc[ids, ["RA", "DEC"]]

    output_frame = pd.DataFrame({"case": case, "input_index": ids})
    for leg_index, sign in enumerate((args.g, -args.g)):
        label = "plus" if sign > 0 else "minus"
        detection = matched_detection(args.base, case, sign, ids).set_index(
            "input_index", verify_integrity=True,
        ).loc[ids]
        science_path, psf_path = image_paths(args.base, case, sign)
        image, header = fits.getdata(science_path, header=True)
        psf_image = fits.getdata(psf_path)
        world = wcs.WCS(header)
        x_truth, y_truth = world.wcs_world2pix(
            truth.RA.to_numpy(float), truth.DEC.to_numpy(float), 1,
        )
        detected_shape = np.full((len(ids), 2), np.nan)
        truth_shape = np.full((len(ids), 2), np.nan)
        for index, input_id in enumerate(ids):
            seed = int((case * 1_000_003 + int(input_id) * 17 + leg_index * 97) % (2**32 - 1))
            for mode, x, y, destination in (
                ("detected", detection.x_detect.iloc[index], detection.y_detect.iloc[index], detected_shape),
                ("truth", x_truth[index], y_truth[index], truth_shape),
            ):
                stamp = shape.cutout(image, x, y, stamp_size=args.stamp_size)
                centre = subpixel_centre(x, y, args.stamp_size, stamp.shape)
                try:
                    destination[index] = deterministic_ngmix(
                        stamp, psf_image, args.pixel_scale, centre, seed,
                    )
                except Exception:
                    # Failure is retained explicitly and handled by one common finite mask.
                    destination[index] = np.nan
            if (index + 1) % 500 == 0:
                print(
                    f"case {case} {label}: {index+1}/{len(ids)} anchors measured",
                    flush=True,
                )
        output_frame[f"g1_detected_{label}"] = detected_shape[:, 0]
        output_frame[f"g2_detected_{label}"] = detected_shape[:, 1]
        output_frame[f"g1_truthpos_{label}"] = truth_shape[:, 0]
        output_frame[f"g2_truthpos_{label}"] = truth_shape[:, 1]
        del image, psf_image

    output_frame.to_feather(output)
    finite = np.isfinite(output_frame.iloc[:, 2:].to_numpy(float)).all(axis=1)
    print(
        f"case {case}: wrote {output}; common finite={finite.sum()}/{len(finite)}",
        flush=True,
    )
    print("ANCHOR_FIXED_POSITION_CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
