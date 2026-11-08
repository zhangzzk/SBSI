"""Compare forward random-neighbour responses at h=.02, .05 and .2.

Use exactly the same prepared pairs and baseline output cuts, restrict to
primaries usable at every amplitude, and verify identical intrinsic scenes.
No model is trained and no coherent-scene correction is inferred.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_response_label_tensor import directional_labels
from scripts.train_fixed_g0_flow import sha256, write_json

AMPLITUDES = (.02, .05, .2)
CUTS = ("uncut", "mag25.8_radius_gt0.60", "mag25.8_radius_ge0.75")


def measured_shapes(root, case, h):
    cat = root / f"case{case}_{h}/real0/catalogues"
    cross = pd.read_feather(cat / "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
                            columns=["id_input", "id_detec"])
    shape = pd.read_feather(cat / "Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather",
                            columns=["NUMBER", "NGMIX_G1", "NGMIX_G2"])
    if (cross["id_input"].duplicated().any() or cross["id_detec"].duplicated().any()
            or shape["NUMBER"].duplicated().any()):
        raise ValueError("nonunique response measurement identities")
    joined = cross.merge(shape, left_on="id_detec", right_on="NUMBER", validate="one_to_one")
    values = joined[["NGMIX_G1", "NGMIX_G2"]].to_numpy(float)
    valid = np.isfinite(values).all(1) & (np.square(values).sum(1) < 1)
    return joined.loc[valid].set_index("id_input")[["NGMIX_G1", "NGMIX_G2"]], {
        "cross_rows": len(cross), "cross_without_shape": len(cross)-len(joined),
        "invalid_shape": int((~valid).sum()), "usable_rows": int(valid.sum())}


def verify_scenes(root, case):
    reference = pd.read_feather(root / f"gals{case}_0.2.feather")
    if reference["index"].duplicated().any():
        raise ValueError("duplicate truth identities")
    columns = [c for c in reference if c not in ("g1", "g2")]
    if not (reference["shear_component_convention"] == "sky_cos_sin").all():
        raise ValueError("unexpected shear-component convention")
    gamma = reference[["g1", "g2"]].to_numpy(float)
    split = len(reference)//2
    np.testing.assert_allclose(gamma[:split], 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(gamma[split:], axis=1), .2, atol=1e-12)
    hashes = {}
    for h in (0., *AMPLITUDES):
        path = root / f"gals{case}_{h}.feather"
        other = pd.read_feather(path)
        pd.testing.assert_frame_equal(reference[columns], other[columns], check_exact=True)
        np.testing.assert_allclose(other[["g1", "g2"]], gamma*(h/.2), rtol=1e-10, atol=1e-12)
        hashes[str(h)] = sha256(path)
    return hashes


def compute(args):
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    reports = []
    for case in range(args.case_start, args.case_stop):
        hashes = verify_scenes(args.source, case)
        legs = {h: measured_shapes(args.source, case, h) for h in (0., *AMPLITUDES)}
        sums = np.zeros((3, 3, 2, 2))
        pair_counts = np.zeros(3, dtype=np.int64)
        ids = [[] for _ in CUTS]
        dropped = total = 0
        for part in manifest["parts"]:
            if part["case"] != case:
                continue
            path = args.prepared / part["path"]
            if sha256(path) != part["sha256"]:
                raise ValueError("prepared response part changed")
            with np.load(path) as data:
                primary, x, angles = data["primary"], data["x"], data["shear_angle"]
                shapes = {h: leg[0].reindex(primary).to_numpy(float) for h, leg in legs.items()}
                common = np.logical_and.reduce([np.isfinite(v).all(1) for v in shapes.values()])
                total += len(primary)
                dropped += int((~common).sum())
                u = np.column_stack((np.cos(2*np.deg2rad(angles.astype(float))),
                                     np.sin(2*np.deg2rad(angles.astype(float)))))
                original = np.sum((shapes[.2]-shapes[0.])*u, axis=1)/.2
                np.testing.assert_allclose(original[common], data["y"][common], atol=5e-6, rtol=1e-6)
                cuts = (common, common & (x[:, 7] > np.log(3)) & (x[:, 8] < 25.8),
                        common & (x[:, 7] >= np.log(3.75)) & (x[:, 8] < 25.8))
                for ki, keep in enumerate(cuts):
                    pair_counts[ki] += keep.sum()
                    ids[ki].append(primary[keep])
                    for hi, h in enumerate(AMPLITUDES):
                        delta = (shapes[h][keep]-shapes[0.][keep])/h
                        parallel = np.sum(delta*u[keep], axis=1)
                        cross = -delta[:, 0]*u[keep, 1]+delta[:, 1]*u[keep, 0]
                        tensor, _ = directional_labels(parallel, cross, angles[keep])
                        sums[ki, hi] += tensor.sum(0)
        counts = [len(np.unique(np.concatenate(group))) for group in ids]
        report = {"case": case, "tensor_sums": sums, "pair_counts": pair_counts,
                  "represented_primaries": counts, "input_pairs": total, "unmatched_or_invalid_pairs_dropped": dropped,
                  "truth_hashes": hashes, "measurement_counts": {str(h): leg[1] for h, leg in legs.items()}}
        reports.append(report)
        print(f"RESPONSE_AMPLITUDE case={case} pairs={total} dropped={dropped}", flush=True)
    write_json(args.output, {"cases": reports, "manifest_sha256": sha256(args.prepared / "manifest.json"),
                            "script_sha256": sha256(Path(__file__))})


def summarize(args):
    parts = [json.loads(p.read_text()) for p in sorted(args.parts.glob("*.json"))]
    if len({p["manifest_sha256"] for p in parts}) != 1 or len({p["script_sha256"] for p in parts}) != 1:
        raise ValueError("inconsistent amplitude parts")
    cases = sorted([c for p in parts for c in p["cases"]], key=lambda c: c["case"])
    if [c["case"] for c in cases] != list(range(200)):
        raise ValueError("amplitude case coverage is not exactly 0--199")
    summaries = {}
    rng = np.random.default_rng(20260917)
    for split, lo, hi in (("fit", 40, 200), ("development", 0, 40)):
        rows = [c for c in cases if lo <= c["case"] < hi]
        sums = np.array([c["tensor_sums"] for c in rows])
        counts = np.array([c["represented_primaries"] for c in rows])
        index = rng.integers(len(rows), size=(10000, len(rows)))
        output = []
        for ki, cut in enumerate(CUTS):
            means = sums[:, ki].sum(0)/counts[:, ki].sum()
            boot = sums[index, ki].sum(1)/counts[index, ki].sum(1)[:, None, None, None]
            scalar = np.trace(means, axis1=1, axis2=2)/2
            boot_scalar = np.trace(boot, axis1=2, axis2=3)/2
            delta = scalar[:2]-scalar[2]
            boot_delta = boot_scalar[:, :2]-boot_scalar[:, 2, None]
            output.append({"cut": cut, "scalar_means": scalar, "tensor_means": means,
                           "scalar_case_se": boot_scalar.std(0, ddof=1),
                           "small_minus_point2": delta,
                           "small_minus_point2_case_se": boot_delta.std(0, ddof=1),
                           "small_minus_point2_ci95": np.quantile(boot_delta, [.025, .975], axis=0)})
        summaries[split] = output
    write_json(args.output, {"amplitudes": AMPLITUDES, "summary": summaries, "cases": cases,
                            "manifest_sha256": parts[0]["manifest_sha256"],
                            "limitations": ["forward finite differences, not central derivatives",
                                            "common usable primary population across all four legs",
                                            "fixed baseline output cuts and represented pairs only",
                                            "random-direction instrument, not native coherent neighbour response",
                                            "case-bootstrap uncertainty excludes training"]})
    print("RESPONSE_AMPLITUDE_COMPLETE", json.dumps(summaries, default=lambda v: v.tolist()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("compute")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--case-start", type=int, required=True)
    run.add_argument("--case-stop", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--parts", type=Path, required=True)
    for p in (run, summary):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("amplitude catalogue scan requires Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (compute if args.mode == "compute" else summarize)(args)


if __name__ == "__main__":
    main()
