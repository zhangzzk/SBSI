#!/usr/bin/env python3
"""Measure neighbour-induced cut response without a flow or fitted correction.

Use existing primary-unsheared, neighbour +/-0.02 scenes. On common usable
identities, decompose the per-leg selected shape response exactly into the
mean paired shape change and changing selection of the pair-average shape.
The latter term tests the effect omitted by an external emulator that shifts
only ellipticity while leaving joint radius/flux draws unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dump_anchorblend_scene_primaries import (
    H, catalogue_paths, read_truth, simulation_roots, stratum_weight, verify_scene_pair,
)
from scripts.evaluate_constgold_truth_parent import CUTS, output_cuts
from scripts.train_fixed_g0_flow import sha256, write_json


def load_leg(paths, ids):
    cross = pd.read_feather(paths["cross"], columns=["id_detec", "id_input"])
    shape = pd.read_feather(paths["shape"],
                            columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"])
    if (cross["id_input"].duplicated().any() or cross["id_detec"].duplicated().any()
            or shape["NUMBER"].duplicated().any()):
        raise ValueError("nonunique scene measurement identities")
    joined = cross.merge(shape, left_on="id_detec", right_on="NUMBER",
                         how="inner", validate="one_to_one").set_index("id_input")
    rows = joined.reindex(ids)
    values = rows[["NGMIX_G1", "NGMIX_G2", "FLUX_RADIUS", "MAG_AUTO"]].to_numpy(float)
    with np.errstate(over="ignore", invalid="ignore"):
        values[:, 3] = 10 ** (-0.4 * (values[:, 3] - 30))
    usable = np.isfinite(values).all(1) & (np.square(values[:, :2]).sum(1) < 1)
    usable &= (values[:, 2:] > 0).all(1)
    return values, usable


def decomposition_sums(plus, minus, weight=1.0, h=H):
    """Weighted sufficient statistics; input pairs must already be usable."""
    plus, minus = np.asarray(plus), np.asarray(minus)
    if (plus.shape != minus.shape or plus.ndim != 2 or plus.shape[1] != 4
            or not len(plus) or not np.isfinite(plus).all() or not np.isfinite(minus).all()
            or not np.isfinite(weight) or weight <= 0 or not np.isfinite(h) or h <= 0):
        raise ValueError("aligned finite usable pairs and positive weight/shear required")
    midpoint = (plus[:, :2] + minus[:, :2]) / 2
    paired_response = (plus[:, :2] - minus[:, :2]) / (2 * h)
    masks = (output_cuts(plus), output_cuts(minus))
    result = []
    for ki, cut in enumerate(CUTS):
        entry = {"cut": cut, "plus": {}, "minus": {},
                 "enters_plus": int((masks[0][:, ki] & ~masks[1][:, ki]).sum()),
                 "enters_minus": int((masks[1][:, ki] & ~masks[0][:, ki]).sum())}
        for direction, values, keep in zip(("plus", "minus"), (plus, minus), masks):
            use = keep[:, ki]
            entry[direction] = {
                "weight": weight * int(use.sum()),
                "shape_sum": weight * values[use, :2].sum(0),
                "midpoint_sum": weight * midpoint[use].sum(0),
                "paired_response_sum": weight * paired_response[use].sum(0),
            }
        result.append(entry)
    return result


def response_terms(entry, h=H):
    plus, minus = entry["plus"], entry["minus"]
    if plus["weight"] <= 0 or minus["weight"] <= 0:
        raise ValueError("empty selected leg")
    def mean(leg, key):
        return np.asarray(leg[key]) / leg["weight"]
    total = (mean(plus, "shape_sum") - mean(minus, "shape_sum")) / (2 * h)
    boundary = (mean(plus, "midpoint_sum") - mean(minus, "midpoint_sum")) / (2 * h)
    within = (mean(plus, "paired_response_sum") + mean(minus, "paired_response_sum")) / 2
    np.testing.assert_allclose(total, boundary + within, rtol=1e-10, atol=1e-11)
    return {"total": total, "boundary": boundary, "within": within}


def calculate(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("scene scan must run under Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for case in range(args.case_start, args.case_stop):
        combined = None
        counts = []
        for stratum, root in enumerate(simulation_roots(case)):
            anchors = pd.read_feather(root / f"anchors_case{case}.feather")
            if anchors["index"].duplicated().any() or not (anchors["case"] == case).all():
                raise ValueError("invalid anchor identities")
            paths = {sign: catalogue_paths(root, case, sign) for sign in (+H, -H)}
            truth = {sign: read_truth(paths[sign]["truth"]) for sign in paths}
            ids = anchors["index"].to_numpy(np.int64)
            verify_scene_pair(truth[+H], truth[-H], ids)
            rendered = np.intersect1d(ids, truth[+H]["index"].to_numpy(np.int64))
            plus, up = load_leg(paths[+H], rendered)
            minus, um = load_leg(paths[-H], rendered)
            common = up & um
            sums = decomposition_sums(plus[common], minus[common], stratum_weight(case, stratum))
            counts.append({"stratum": stratum, "anchors": len(ids), "rendered": len(rendered),
                           "unrendered_dropped": len(ids) - len(rendered),
                           "common_usable": int(common.sum()), "plus_only_dropped": int((up & ~um).sum()),
                           "minus_only_dropped": int((um & ~up).sum()),
                           "neither_usable_dropped": int((~up & ~um).sum())})
            if combined is None:
                combined = sums
            else:
                for entry, addition in zip(combined, sums):
                    for direction in ("plus", "minus"):
                        for field in entry[direction]:
                            entry[direction][field] += addition[direction][field]
                    entry["enters_plus"] += addition["enters_plus"]
                    entry["enters_minus"] += addition["enters_minus"]
        report = {"case": case, "counts": counts, "sufficient": combined,
                  "response": {entry["cut"]: response_terms(entry) for entry in combined}}
        reports.append(report)
        print(f"NEIGHBOUR_SELECTION case={case} joint_boundary="
              f"{report['response'][CUTS[1]]['boundary'][0]:+.7f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"h": H, "cuts": CUTS, "cases": reports,
                            "script_sha256": sha256(Path(__file__)),
                            "population": "original and complement anchor strata reweighted to their V2 truth parent; both-leg usability, independently reselected measured cuts"})


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    products = [json.loads(path.read_text()) for path in sorted(args.parts.glob("block*.json"))]
    cases = sorted((case for product in products for case in product["cases"]), key=lambda x: x["case"])
    if [case["case"] for case in cases] != list(range(400, 900)):
        raise ValueError("summary requires exact cases400--899 coverage")
    picks = np.random.default_rng(20260917).integers(len(cases), size=(10000, len(cases)))
    summary = {}
    for cut in CUTS:
        summary[cut] = {}
        for name in ("total", "boundary", "within"):
            values = np.array([case["response"][cut][name] for case in cases])
            boot = values[picks].mean(1)
            summary[cut][name] = {"mean": values.mean(0), "case_bootstrap_se": boot.std(0, ddof=1),
                                  "ci95": np.quantile(boot, [0.025, 0.975], axis=0)}
        print(f"NEIGHBOUR_SELECTION_SUMMARY {cut} {summary[cut]}", flush=True)
    write_json(args.output, {"case_count": len(cases), "summary": summary, "cases": cases,
                            "limitations": ["finite-stencil decomposition on common usable identities",
                                            "V2 scene-anchor parent differs from the full ConstGold truth parent",
                                            "selection contribution is measured directly; no emulator or flow fit is used",
                                            "does not establish the transferable numerical contribution to ConstGold bias"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("compute")
    compute.add_argument("--case-start", type=int, required=True)
    compute.add_argument("--case-stop", type=int, required=True)
    compute.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--parts", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (calculate if args.command == "compute" else summarize)(args)


if __name__ == "__main__":
    main()
