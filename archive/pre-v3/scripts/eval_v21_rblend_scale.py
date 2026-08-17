"""One-shot constgold acceptance test of an independent R_blend calibration.

The calibration must come from the anchor-scene global-scale or isotonic
fitters, or from the matched outside-aperture response experiment.  This script
first enforces their held-out validation gate, then opens the frozen constgold
dumps and changes only ``R_blend``.  It never fits, chooses, or adjusts a
parameter from constgold.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pyarrow.feather as pf

from sbs_shear import domain
from scripts.eval_v2_indomain_m import catalogue_true_props


SEEDS = [501, 502, 503, 505, 506, 507, 508, 509,
         510, 511, 512, 513, 514, 515, 516, 517]


def seed_from_path(path: str) -> int:
    match = re.search(r"_s(\d+)\.feather$", os.path.basename(path))
    if match is None:
        raise RuntimeError(f"cannot parse seed from {path}")
    return int(match.group(1))


def validate_calibration(
    cal: dict,
    max_residual: float,
    max_z: float,
    expected_tag: str = "lsst_r_extnbr_v21",
) -> float:
    if cal.get("tag") != expected_tag:
        raise RuntimeError(f"wrong calibration tag: {cal.get('tag')!r}")
    # After the development estimator passes its held-out gate, an unchanged
    # global-scale method may be refit on all independent anchor cases.  Older
    # artifacts contain only dev_scale and remain backward compatible.
    scale = float(cal.get("deployment_scale", cal["dev_scale"]))
    residual = abs(float(cal["test_residual"]))
    residual_se = float(cal["test_residual_se"])
    if not 0.5 < scale < 2.0:
        raise RuntimeError(f"implausible calibration scale {scale}")
    if residual > max_residual:
        raise RuntimeError(
            f"held-out anchor residual {residual:.6f} exceeds {max_residual:.6f}"
        )
    if residual_se <= 0 or residual / residual_se > max_z:
        raise RuntimeError(
            f"held-out anchor residual {residual:.6f} is "
            f"{residual / residual_se:.2f} sigma (limit {max_z:.2f})"
        )
    return scale


def validate_additive_calibration(
    calibration: dict,
    max_residual: float,
    max_z: float,
    expected_tag: str,
) -> float:
    """Validate a pre-registered matched outside-aperture response offset."""
    if calibration.get("tag") != expected_tag:
        raise RuntimeError(f"wrong additive calibration tag: {calibration.get('tag')!r}")
    if calibration.get("method") != "matched_outside10_additive_response":
        raise RuntimeError(f"wrong additive method: {calibration.get('method')!r}")
    if calibration.get("gate_passed") is not True:
        raise RuntimeError("additive aperture calibration failed its held-out gate")
    offset = float(calibration["deployment_offset"])
    residual = abs(float(calibration["test_residual"]))
    residual_se = float(calibration["test_residual_se"])
    if not -0.1 < offset < 0.1:
        raise RuntimeError(f"implausible additive response {offset}")
    if residual > max_residual:
        raise RuntimeError(
            f"held-out anchor residual {residual:.6f} exceeds {max_residual:.6f}"
        )
    if residual_se <= 0 or residual / residual_se > max_z:
        raise RuntimeError(
            f"held-out anchor residual {residual:.6f} is "
            f"{residual / residual_se:.2f} sigma (limit {max_z:.2f})"
        )
    return offset


def isotonic_transform(values, calibration):
    """Apply a saved sklearn-compatible isotonic step function with clipping."""
    x = np.asarray(calibration["x_thresholds"], dtype=float)
    y = np.asarray(calibration["y_thresholds"], dtype=float)
    if len(x) < 2 or len(x) != len(y) or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
        raise RuntimeError("invalid isotonic calibration thresholds")
    return np.interp(np.asarray(values, dtype=float), x, y, left=y[0], right=y[-1])


def summarize(label: str, values: np.ndarray) -> None:
    sd = values.std(ddof=1)
    print(
        f"  {label:<23} {values.mean():+8.3f} +/- {sd / np.sqrt(len(values)):.3f}% "
        f"(seed sd {sd:.3f}%, N={len(values)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    calibration_group = ap.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument("--calibration-json")
    calibration_group.add_argument("--additive-json")
    calibration_group.add_argument("--isotonic-npz")
    ap.add_argument("--isotonic-summary")
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True,
                    help="constgold response catalogue used to replay the dump population")
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--expected-tag", default="lsst_r_extnbr_v21")
    ap.add_argument(
        "--primary-domain", choices=("v21", "rectangular"), default="v21",
        help="truth-only population mask applied to the frozen dumps",
    )
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--check-emulator-coverage", action="store_true")
    ap.add_argument("--output-json")
    ap.add_argument("--max-validation-residual", type=float, default=0.006)
    ap.add_argument("--max-validation-z", type=float, default=2.0)
    args = ap.parse_args()

    if args.calibration_json:
        with open(args.calibration_json, encoding="utf-8") as handle:
            calibration = json.load(handle)
        scale = validate_calibration(
            calibration, args.max_validation_residual, args.max_validation_z,
            expected_tag=args.expected_tag,
        )
        transform = lambda values: scale * values
        label = f"scaled R_blend ({scale:.6f})"
    elif args.additive_json:
        with open(args.additive_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        calibration = payload.get("additive_calibration", payload)
        offset = validate_additive_calibration(
            calibration, args.max_validation_residual, args.max_validation_z,
            expected_tag=args.expected_tag,
        )
        transform = lambda values: values + offset
        label = f"additive R_blend ({offset:+.6f})"
    else:
        if not args.isotonic_summary:
            raise RuntimeError("--isotonic-summary is required with --isotonic-npz")
        with open(args.isotonic_summary, encoding="utf-8") as handle:
            calibration = json.load(handle)
        if calibration.get("tag") != args.expected_tag \
                or calibration.get("method") != "isotonic":
            raise RuntimeError("wrong isotonic calibration metadata")
        residual = abs(float(calibration["test_residual"]))
        residual_se = float(calibration["test_residual_se"])
        if residual > args.max_validation_residual or residual > args.max_validation_z * residual_se:
            raise RuntimeError("isotonic calibration failed its held-out global gate")
        artifact = np.load(args.isotonic_npz)
        if str(artifact["tag"]) != args.expected_tag:
            raise RuntimeError("wrong isotonic artifact tag")
        transform = lambda values: isotonic_transform(values, artifact)
        label = "isotonic R_blend"
    print(
        f"Frozen half-shear calibration={label}; held-out residual="
        f"{calibration['test_residual']:+.6f} +/- {calibration['test_residual_se']:.6f}",
        flush=True,
    )

    paths = sorted(glob.glob(args.dump_glob), key=seed_from_path)
    seeds = [seed_from_path(path) for path in paths]
    if seeds != SEEDS:
        raise RuntimeError(f"expected seeds {SEEDS}, found {seeds}")

    truth_props = catalogue_true_props(args.catalogue, args.min_case, time.time())
    mag = truth_props["r_input_p"].to_numpy(float)
    re_value = truth_props["Re_input_p"].to_numpy(float)
    if args.primary_domain == "v21":
        population = domain.in_domain(mag, re_value)
        population_label = "canonical V2.1"
    else:
        population = (mag < args.mag_max) & (re_value > args.re_min)
        population_label = f"rectangular mag<{args.mag_max}, Re>{args.re_min}"
    print(
        f"{population_label} population: {int(population.sum()):,}/{len(population):,} rows",
        flush=True,
    )
    if args.check_emulator_coverage:
        from scripts.eval_v2_indomain_m import check_emulator_covers_domain
        check_emulator_covers_domain(args.expected_tag, mag, re_value, population)

    baseline = []
    candidate = []
    components = []
    ref_case = ref_index = None
    for seed, path in zip(seeds, paths):
        table = pf.read_table(
            path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
            memory_map=True,
        )
        case = table["case"].to_numpy(zero_copy_only=False)
        index = table["input_index"].to_numpy(zero_copy_only=False)
        if len(table) != len(truth_props) \
                or not np.array_equal(case, truth_props["case"].to_numpy()) \
                or not np.array_equal(index, truth_props["input_index"].to_numpy()):
            raise RuntimeError(f"{path}: catalogue replay does not align exactly")
        if ref_case is None:
            ref_case, ref_index = case.copy(), index.copy()
        elif not np.array_equal(case, ref_case) or not np.array_equal(index, ref_index):
            raise RuntimeError(f"row identity/order changed in {path}")
        rs = table["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        rf = table["R_flow"].to_numpy(zero_copy_only=False).astype(float)
        rb = table["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        finite = np.isfinite(rs) & np.isfinite(rf) & np.isfinite(rb)
        keep = population & finite
        if keep.sum() != population.sum():
            raise RuntimeError(
                f"{path}: {(population & ~finite).sum()} non-finite population rows"
            )
        means = np.array([rs[keep].mean(), rf[keep].mean(), rb[keep].mean()])
        m0 = 100.0 * (means[0] / (means[1] + means[2]) - 1.0)
        rb_calibrated = float(transform(rb[keep]).mean())
        m1 = 100.0 * (means[0] / (means[1] + rb_calibrated) - 1.0)
        baseline.append(m0)
        candidate.append(m1)
        components.append([*means, rb_calibrated])
        print(f"seed {seed}: baseline={m0:+.3f}% candidate={m1:+.3f}%", flush=True)

    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    delta = candidate - baseline
    means = np.asarray(components).mean(axis=0)
    print("\nCONSTGOLD -- ONE-SHOT FROZEN R_BLEND ACCEPTANCE")
    summarize(f"current {args.expected_tag}", baseline)
    summarize(label, candidate)
    summarize("paired change", delta)
    print(
        f"  components: R_sim={means[0]:.6f} R_flow={means[1]:.6f} "
        f"R_blend={means[2]:.6f} -> {means[3]:.6f}"
    )
    if args.output_json:
        payload = {
            "expected_tag": args.expected_tag,
            "primary_domain": args.primary_domain,
            "population_label": population_label,
            "n_population": int(population.sum()),
            "seeds": seeds,
            "calibration": label,
            "calibration_source": (
                args.calibration_json or args.additive_json or args.isotonic_npz
            ),
            "heldout_anchor_residual": float(calibration["test_residual"]),
            "heldout_anchor_residual_se": float(calibration["test_residual_se"]),
            "baseline_m_percent_by_seed": baseline.tolist(),
            "candidate_m_percent_by_seed": candidate.tolist(),
            "paired_change_percent_by_seed": delta.tolist(),
            "baseline_m_percent_mean": float(baseline.mean()),
            "baseline_seed_sem_percent": float(baseline.std(ddof=1) / np.sqrt(len(baseline))),
            "candidate_m_percent_mean": float(candidate.mean()),
            "candidate_seed_sem_percent": float(candidate.std(ddof=1) / np.sqrt(len(candidate))),
            "paired_change_percent_mean": float(delta.mean()),
            "paired_change_seed_sem_percent": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "components": {
                "R_sim": float(means[0]), "R_flow": float(means[1]),
                "R_blend_raw": float(means[2]), "R_blend_calibrated": float(means[3]),
            },
        }
        with open(args.output_json, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(f"wrote {args.output_json}")
    print("V21_RBLEND_CALIBRATION_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
