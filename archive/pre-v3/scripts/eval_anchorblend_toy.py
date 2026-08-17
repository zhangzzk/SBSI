"""Toy test of coherent neighbour response, pair additivity, and V2.1 BlendEMU.

Every scenario is one complete postage-stamp scene.  The target is unsheared.
We measure its ngmix response when (a) all neighbours receive coherent antithetic
``+/-g1`` and (b) each neighbour is sheared individually while every other source
remains present and unsheared.  Thus

    coherent truth - sum(individual-neighbour truths)

tests finite-shear multi-neighbour interaction without an emulator.  The V2.1
emulator is then evaluated on the same intrinsic scene and its native per-pair
predictions are summed for a separate model-versus-truth comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import galsim
import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from archive.toy_blend_linearity import BETA, PIX, PSF_FWHM, _PSF, measure

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)
from blendemu.inference import BlendingPredictor  # noqa: E402


ZERO_POINT = 30.0
PIXEL_RMS = 0.312
COND = dict(pixel_size=PIX, zero_point=ZERO_POINT, psf_fwhm=PSF_FWHM,
            moffat_beta=BETA, pixel_rms=PIXEL_RMS)


def source(mag, re_arcsec, n, x, y):
    return dict(mag=float(mag), re=float(re_arcsec), n=float(n), x=float(x), y=float(y))


def polar_source(mag, re_arcsec, n, distance, angle_deg):
    angle = np.deg2rad(angle_deg)
    return source(mag, re_arcsec, n, distance * np.cos(angle), distance * np.sin(angle))


def scenarios():
    target = source(24.5, 0.70, 1.0, 0.0, 0.0)
    return {
        "one_close_equal": (
            target,
            [polar_source(24.5, 0.70, 1.0, 0.5, 35)],
        ),
        "one_typical_fainter": (
            target,
            [polar_source(25.5, 0.50, 1.0, 1.2, 35)],
        ),
        "four_mixed": (
            target,
            [polar_source(24.0, 0.80, 1.5, 0.7, 20),
             polar_source(25.0, 0.55, 1.0, 1.0, 110),
             polar_source(26.0, 0.40, 0.8, 1.6, 205),
             polar_source(27.0, 0.30, 1.0, 2.5, 300)],
        ),
        "eight_faint_close": (
            target,
            [polar_source(26.5, 0.35, 1.0, 0.9, angle)
             for angle in np.linspace(0, 360, 8, endpoint=False)],
        ),
        "three_far": (
            target,
            [polar_source(24.5, 0.70, 1.0, distance, angle)
             for distance, angle in [(5.0, 15), (7.0, 145), (9.0, 275)]],
        ),
    }


def render(scene, sheared, g1, stamp):
    image = galsim.ImageF(stamp, stamp, scale=PIX)
    selected = set(sheared)
    for index, gal in enumerate(scene):
        flux = 10.0 ** (-0.4 * (gal["mag"] - ZERO_POINT))
        obj = galsim.Sersic(
            n=gal["n"], half_light_radius=gal["re"], flux=flux,
        )
        x, y = gal["x"], gal["y"]
        if index in selected:
            obj = obj.shear(g1=g1, g2=0.0)
            # BlendEMU shears galaxy profiles while keeping catalogue sky
            # positions fixed.  Moving the neighbour centroid with shear would
            # measure a different, much larger astrometric response.
        galsim.Convolve([obj, _PSF]).drawImage(
            image=image, add_to_image=True, offset=(x / PIX, y / PIX),
        )
    return image.array


def response_draws(scene, sheared, g, nreal, stamp):
    plus = render(scene, sheared, +g, stamp)
    minus = render(scene, sheared, -g, stamp)
    values = []
    for realization in range(nreal):
        noise = np.random.RandomState(realization).normal(0.0, PIXEL_RMS, plus.shape)
        try:
            e_plus = measure(plus + noise, realization)
            e_minus = measure(minus + noise, realization)
        except Exception:
            values.append(np.nan)
            continue
        values.append((float(e_plus[0]) - float(e_minus[0])) / (2.0 * g))
    return np.asarray(values, dtype=float)


def emulator_pairs(predictor, scene):
    dec0 = -0.5
    cosdec = np.cos(np.deg2rad(dec0))
    frame = pd.DataFrame({
        "index": np.arange(len(scene), dtype=np.int64),
        "RA": [180.0 + gal["x"] / (3600.0 * cosdec) for gal in scene],
        "DEC": [dec0 + gal["y"] / 3600.0 for gal in scene],
        "r": [gal["mag"] for gal in scene],
        "Re": [gal["re"] for gal in scene],
        "sersic_n": [gal["n"] for gal in scene],
    })
    pairs = predictor.predict_response(frame, frame)
    primary = next(c for c in pairs if c.startswith("index") and c.endswith("_p"))
    secondary = next(c for c in pairs if c.startswith("index") and c.endswith("_s"))
    target_pairs = pairs.loc[pairs[primary].to_numpy(np.int64) == 0]
    by_secondary = pd.Series(
        target_pairs["response"].to_numpy(float),
        index=target_pairs[secondary].to_numpy(np.int64),
    )
    expected = np.arange(1, len(scene), dtype=np.int64)
    missing = expected[~np.isin(expected, by_secondary.index.to_numpy(np.int64))]
    if len(missing):
        raise RuntimeError(f"V2.1 emulator omitted in-support toy neighbours {missing.tolist()}")
    return by_secondary.reindex(expected).to_numpy(float)


def mean_sem(values):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float("nan"), float("nan"), int(finite.sum())
    kept = values[finite]
    return float(kept.mean()), float(kept.std(ddof=1) / np.sqrt(len(kept))), len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--nreal", type=int, default=400)
    ap.add_argument("--stamp", type=int, default=112)
    ap.add_argument("--scenario", choices=["all", *scenarios().keys()], default="all")
    ap.add_argument("--output-json")
    args = ap.parse_args()

    predictor = BlendingPredictor.load(
        "/home/z/Zekang.Zhang/blendemu/models", tag="lsst_r_extnbr_v21",
        conditions=COND, device="cpu",
    )
    selected = scenarios()
    if args.scenario != "all":
        selected = {args.scenario: selected[args.scenario]}

    output = {"g": args.g, "nreal": args.nreal, "model": "lsst_r_extnbr_v21", "scenarios": {}}
    print("TOY COHERENT-NEIGHBOUR RESPONSE VS INDIVIDUAL PAIRS VS V2.1 EMULATOR")
    for name, (target, neighbours) in selected.items():
        scene = [target, *neighbours]
        coherent = response_draws(scene, range(1, len(scene)), args.g, args.nreal, args.stamp)
        individual = np.vstack([
            response_draws(scene, [j], args.g, args.nreal, args.stamp)
            for j in range(1, len(scene))
        ])
        common = np.isfinite(coherent) & np.isfinite(individual).all(axis=0)
        coherent = coherent[common]
        individual = individual[:, common]
        summed_truth = individual.sum(axis=0)
        excess = coherent - summed_truth
        pred_pairs = emulator_pairs(predictor, scene)

        coh_mean, coh_sem, n_good = mean_sem(coherent)
        sum_mean, sum_sem, _ = mean_sem(summed_truth)
        excess_mean, excess_sem, _ = mean_sem(excess)
        pred_sum = float(pred_pairs.sum())
        model_residual = pred_sum / coh_mean - 1.0
        additivity_residual = sum_mean / coh_mean - 1.0
        print(f"\n{name}: neighbours={len(neighbours)}, successful noise draws={n_good}/{args.nreal}")
        print(f"  coherent neighbour truth      {coh_mean:+.6f} +/- {coh_sem:.6f}")
        print(f"  sum individual truths         {sum_mean:+.6f} +/- {sum_sem:.6f}")
        print(f"  coherent - individual sum     {excess_mean:+.6f} +/- {excess_sem:.6f} "
              f"({additivity_residual:+.2%} sum/coherent - 1)")
        print(f"  V2.1 emulator pair sum        {pred_sum:+.6f} "
              f"({model_residual:+.2%} emulator/coherent - 1)")
        for j, (truth_draws, prediction) in enumerate(zip(individual, pred_pairs), start=1):
            truth_mean, truth_sem, _ = mean_sem(truth_draws)
            print(f"    neighbour {j:>2}: truth={truth_mean:+.6f} +/- {truth_sem:.6f} "
                  f"emulator={prediction:+.6f}")
        output["scenarios"][name] = {
            "n_neighbours": len(neighbours),
            "n_success": n_good,
            "coherent_truth": coh_mean,
            "coherent_truth_sem": coh_sem,
            "individual_truth_sum": sum_mean,
            "individual_truth_sum_sem": sum_sem,
            "coherent_minus_individual_sum": excess_mean,
            "coherent_minus_individual_sum_sem": excess_sem,
            "individual_sum_over_coherent_minus_one": additivity_residual,
            "emulator_sum": pred_sum,
            "emulator_over_coherent_minus_one": model_residual,
            "individual_truth_means": [float(np.nanmean(values)) for values in individual],
            "emulator_pair_predictions": pred_pairs.tolist(),
        }
    print("\n" + json.dumps(output, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("ANCHORBLEND_TOY_DONE", flush=True)


if __name__ == "__main__":
    main()
