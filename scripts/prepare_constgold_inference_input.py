#!/usr/bin/env python
"""Freeze actual, independently selected ConstGold plus-leg measurements."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_thresholds(config):
    """Read this adapter's supported magnitude/radius cuts from the likelihood."""
    selection = config["measured_selection"]
    if selection["mode"] != "output_cut_with_population_normalization":
        raise ValueError("input selection requires population-normalized output bounds")
    bounds = {}
    for spec in selection["bounds"]:
        name, lower, upper = spec.split(":")
        if name in bounds or upper or not lower:
            raise ValueError("require unique lower-only radius and flux bounds")
        bounds[name] = float(lower)
    if set(bounds) != {"measured_flux_radius", "measured_flux_from_mag_auto"}:
        raise ValueError("input adapter supports only radius and magnitude selection")
    if not all(np.isfinite(value) and value > 0 for value in bounds.values()):
        raise ValueError("selection bounds must be finite and positive")
    conditions = config["observing_conditions"]
    return (
        bounds["measured_flux_radius"] * conditions["pixel_size"],
        conditions["zero_point"] - 2.5 * np.log10(bounds["measured_flux_from_mag_auto"]),
    )


def read_case(root, case, *, pixel_size, zero_point, primary_mag, primary_re,
              measured_radius_min=0.75, measured_mag_max=25.8):
    base = root / f"case{case}_0.02" / "real0" / "catalogues"
    paths = {
        "truth": base / "input/gals_info_tile180.0_-0.5.feather",
        "crossmatch": base / "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
        "shapes": base / "Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
    }
    truth = pd.read_feather(paths["truth"], columns=[
        "index_input", "r_input", "Re_input", "gamma1_input", "gamma2_input",
    ])
    cross = pd.read_feather(paths["crossmatch"], columns=["id_detec", "id_input"])
    shape = pd.read_feather(paths["shapes"], columns=[
        "NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS",
    ])
    for frame, keys in ((truth, ["index_input"]), (cross, ["id_detec", "id_input"]), (shape, ["NUMBER"])):
        for key in keys:
            if frame[key].isna().any() or frame[key].duplicated().any():
                raise ValueError(f"case {case}: missing or duplicate {key}")
    if not np.allclose(truth[["gamma1_input", "gamma2_input"]], [0.02, 0.0], rtol=0, atol=1e-14):
        raise ValueError(f"case {case}: input is not the plus (0.02, 0) shear")
    joined = cross.merge(shape, left_on="id_detec", right_on="NUMBER", validate="one_to_one")
    matched = joined.merge(truth, left_on="id_input", right_on="index_input", validate="one_to_one")
    domain = (
        (matched.r_input > primary_mag[0]) & (matched.r_input < primary_mag[1])
        & (matched.Re_input > primary_re[0]) & (matched.Re_input < primary_re[1])
    )
    supported = matched.loc[domain].copy()
    values = supported[["NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"]].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    usable = finite & (values[:, 3] > 0) & (values[:, 0] != -1.0) & ~(
        (values[:, 0] == 0.0) & (values[:, 1] == 0.0)
    )
    selected = usable & (values[:, 2] < measured_mag_max) & (
        pixel_size * values[:, 3] >= measured_radius_min
    )
    chosen = supported.loc[selected]
    output = pd.DataFrame({
        "source_case": np.full(len(chosen), case, dtype=np.int64),
        "source_input_index": chosen.id_input.to_numpy(np.int64),
        "source_detection_id": chosen.NUMBER.to_numpy(np.int64),
        "measured_ngmix_g1": chosen.NGMIX_G1.to_numpy(float),
        "measured_ngmix_g2": chosen.NGMIX_G2.to_numpy(float),
        "measured_flux_radius": chosen.FLUX_RADIUS.to_numpy(float),
        "measured_flux_from_mag_auto": 10.0 ** (0.4 * (zero_point - chosen.MAG_AUTO.to_numpy(float))),
        "measured_mag_auto": chosen.MAG_AUTO.to_numpy(float),
    })
    report = {
        "case": case, "truth_rows": len(truth), "crossmatch_rows": len(cross), "shape_rows": len(shape),
        "crossmatch_without_shape": len(cross) - len(joined), "shape_without_crossmatch": len(shape) - len(joined),
        "matched_without_truth": len(joined) - len(matched), "outside_truth_domain": int((~domain).sum()),
        "in_truth_domain": len(supported), "invalid_or_unusable": int((~usable).sum()),
        "usable": int(usable.sum()), "selected": len(output),
        "sources": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()},
    }
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--likelihood-config", type=Path, required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=140, help="exclusive")
    parser.add_argument("--primary-mag", type=float, nargs=2, default=(18.0, 25.8))
    parser.add_argument("--primary-re", type=float, nargs=2, default=(0.5, 1.5))
    args = parser.parse_args(argv)
    if args.output.exists() or args.sample_size <= 0 or args.min_case >= args.max_case:
        raise ValueError("require a new output, positive sample size, and nonempty case range")
    config = json.loads(args.likelihood_config.read_text())
    model_hash = file_hash(args.measurement_model)
    if config["measurement_model"]["sha256"] != model_hash:
        raise ValueError("checkpoint differs from the likelihood configuration")
    targets = config["measurement_model"]["target_names"]
    expected_targets = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto"]
    if targets != expected_targets:
        raise ValueError("this input adapter requires the joint physical four-output likelihood")
    radius_min, mag_max = selection_thresholds(config)
    frames, reports = [], []
    for case in range(args.min_case, args.max_case):
        frame, report = read_case(
            args.constgold_root, case, pixel_size=config["observing_conditions"]["pixel_size"],
            zero_point=config["observing_conditions"]["zero_point"],
            primary_mag=args.primary_mag, primary_re=args.primary_re,
            measured_radius_min=radius_min, measured_mag_max=mag_max,
        )
        frames.append(frame)
        reports.append(report)
        print(f"case {case}: selected={len(frame)} unmatched_cross={report['crossmatch_without_shape']} "
              f"unusable={report['invalid_or_unusable']}", flush=True)
    eligible = pd.concat(frames, ignore_index=True)
    if eligible[["source_case", "source_input_index"]].duplicated().any():
        raise ValueError("duplicate catalogue identities")
    if len(eligible) < args.sample_size:
        raise ValueError(f"only {len(eligible)} eligible rows for {args.sample_size} requested")
    rows = np.random.default_rng(args.seed).choice(len(eligible), args.sample_size, replace=False)
    chosen = eligible.iloc[rows].reset_index(drop=True)
    measurements = chosen[[*targets, "measured_mag_auto"]]
    truth = chosen[["source_case", "source_input_index", "source_detection_id"]].copy()
    truth["mock_kind"] = "image"
    truth["shear_transform"] = config["shear_transform"]
    truth["injected_g1"], truth["injected_g2"] = 0.02, 0.0
    args.output.mkdir(parents=True)
    measurements.to_parquet(args.output / "measurements.parquet", index=False)
    truth.to_parquet(args.output / "truth.parquet", index=False)
    manifest = {
        "mock_kind": "image", "catalogue": "constgold", "leg": "plus", "injected_g1": 0.02, "injected_g2": 0.0,
        "shear_transform": config["shear_transform"], "measurement_model_sha256": model_hash,
        "target_names": targets, "n_objects": len(chosen), "n_eligible": len(eligible), "sample_seed": args.seed,
        "sampling": "uniform without replacement after independent plus-leg usability and selection",
        "case_range": [args.min_case, args.max_case], "primary_mag": args.primary_mag, "primary_re": args.primary_re,
        "truth_domain": "strict primary magnitude and semi-major radius bounds matching the configured finite prior; no neighbour cut",
        "usability": "finite four measurements, positive radius, NGMIX_G1 != -1, shape != (0,0)",
        "measured_selection": f"MAG_AUTO < {mag_max:g}; FLUX_RADIUS >= {radius_min:g} arcsec; no measured-|e| cut",
        "measured_selection_bounds": config["measured_selection"]["bounds"],
        "minus_leg_read": False, "per_case": reports,
        "output_sha256": {name: file_hash(args.output / name) for name in ["measurements.parquet", "truth.parquet"]},
        "implementation_sha256": file_hash(__file__), "likelihood_config_sha256": file_hash(args.likelihood_config),
    }
    (args.output / "image_mock_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"CONSTGOLD_INPUT_COMPLETE n={len(chosen)} eligible={len(eligible)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
