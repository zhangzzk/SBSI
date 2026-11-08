#!/usr/bin/env python3
"""Audit the frozen emulator against neighbour-only scenes on identical rows.

No flow enters the comparison. Keep all rendered objects when forming the
neighbour list, then zero pairs whose secondary is an unsheared anchor. This
preserves the emulator's finite-k neighbour convention. Compare its native
per-unit-shear response and the primary-ellipticity proxy used by SBSI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.output_conditioned_response import DiskResponseMoment, OutputConditionedResponse
from sbsi.disk_response_transport import disk_response_transport, infinitesimal_response_matrix
from scripts.audit_neighbour_selection_response import decomposition_sums, load_leg, response_terms
from scripts.dump_anchorblend_scene_primaries import (
    H, catalogue_paths, file_sha256, load_predictor, read_truth, simulation_roots,
    stratum_weight, verify_scene_pair,
)
from scripts.evaluate_constgold_truth_parent import CUTS, output_cuts
from scripts.train_fixed_g0_flow import write_json


RECIPES = ("native_full_scene", "native_sheared_neighbours", "primary_shape_proxy")


def aggregate_predictions(pairs, truth, ids):
    """Align native scalar responses and a diagnostic primary-shape proxy."""
    ids = np.asarray(ids, dtype=np.int64)
    if (len(np.unique(ids)) != len(ids) or truth["index"].duplicated().any()
            or pairs.duplicated(["index_input_p", "index_input_s"]).any()):
        raise ValueError("duplicate scene, primary or pair identities")
    if not np.isin(pairs["index_input_p"], ids).all():
        raise ValueError("unexpected predicted primary")
    by_id = truth.set_index("index")
    secondary = by_id.reindex(pairs["index_input_s"])
    primary = by_id.reindex(ids)
    if secondary["gamma1"].isna().any() or primary["axis_ratio"].isna().any():
        raise ValueError("predicted identity absent from truth")
    sheared = np.isclose(secondary["gamma1"].to_numpy(), H, rtol=0, atol=1e-12)
    stationary = np.isclose(secondary["gamma1"].to_numpy(), 0, rtol=0, atol=1e-12)
    if (not (sheared | stationary).all()
            or not np.allclose(secondary["gamma2"], 0, rtol=0, atol=1e-12)
            or not np.allclose(primary[["gamma1", "gamma2"]], 0, rtol=0, atol=1e-12)):
        raise ValueError("unexpected shear pattern")
    values = pairs["response"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite emulator response")
    table = pd.DataFrame({"id": pairs["index_input_p"].to_numpy(),
                          "full": values, "active": values * sheared})
    grouped = table.groupby("id").sum().reindex(ids, fill_value=0)
    e1, e2 = ellipticity_from_axis_ratio_angle(primary["axis_ratio"].to_numpy(),
                                              primary["position_angle"].to_numpy())
    ep = np.column_stack(apply_shear_to_ellipticity(e1, e2, H, 0))
    em = np.column_stack(apply_shear_to_ellipticity(e1, e2, -H, 0))
    full, active = grouped["full"].to_numpy(), grouped["active"].to_numpy()
    return {"native_full_scene": np.column_stack((full, np.zeros(len(ids)))),
            "native_sheared_neighbours": np.column_stack((active, np.zeros(len(ids)))),
            "primary_shape_proxy": active[:, None] * (ep - em) / (2 * H)}, {
                "pairs": len(pairs), "stationary_secondary_pairs": int(stationary.sum()),
                "isolated_primaries": int((~np.isin(ids, pairs["index_input_p"])).sum())}


def add_predictions(sums, plus, minus, predictions, weight):
    """Use precisely the same selected weights as the measured decomposition."""
    for ki, entry in enumerate(sums):
        for direction, leg in (("plus", plus), ("minus", minus)):
            keep = output_cuts(leg)[:, ki]
            for name in predictions:
                entry[direction][name] = weight * predictions[name][keep].sum(0)


def calculate(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("scene scan must run under Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    for key in ("emulator_model", "emulator_metadata"):
        path = manifest[key]
        if file_sha256(path) != manifest["static_sha256"][path]:
            raise ValueError(f"emulator hash mismatch: {path}")
    predictor = load_predictor(manifest, "cpu")
    candidates = {} if args.candidate_root is None else {
        arm: OutputConditionedResponse(args.candidate_root / arm, nthread=4)
        for arm in ("truth_only", "output_conditioned")
    }
    if args.disk_candidate_root is not None:
        candidates["disk_moment"] = DiskResponseMoment(args.disk_candidate_root, nthread=4)
    recipes = (*RECIPES, *(f"candidate_{arm}" for arm in candidates),
               *( ["candidate_disk_moment_linear"] if "disk_moment" in candidates else []))
    reports = []
    for case in range(args.case_start, args.case_stop):
        combined, counts = None, []
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
            scored = rendered[common]
            pairs = predictor.predict_response(truth[+H].loc[truth[+H]["index"].isin(scored)], truth[+H])
            predictions, report = aggregate_predictions(pairs, truth[+H], scored)
            if candidates:
                # No g0 scene exists in this instrument. This symmetric
                # photometry proxy is explicitly an O(h^2) approximation;
                # a fresh g0 leg is required before accepting deployment.
                midpoint = (plus[common] + minus[common]) / 2
                locations = pd.Index(scored).get_indexer(pairs["index_input_p"])
                if (locations < 0).any():
                    raise ValueError("candidate pair primary not in aligned outputs")
                secondary = truth[+H].set_index("index").reindex(pairs["index_input_s"])
                active = np.isclose(secondary["gamma1"], H, atol=1e-12, rtol=0)
                for arm, candidate in candidates.items():
                    prediction = candidate.predict_pairs(pairs, midpoint[locations, 2], midpoint[locations, 3])
                    summed = pd.Series(prediction * active, index=pairs["index_input_p"]).groupby(level=0).sum()
                    values = summed.reindex(scored, fill_value=0).to_numpy()
                    if candidate.response_coordinate == "disk_velocity":
                        shapes = torch.as_tensor(midpoint[:, :2], dtype=torch.float64)
                        velocity = torch.zeros_like(shapes)
                        velocity[:, 0] = torch.as_tensor(values)*H
                        upper, _ = disk_response_transport(shapes, velocity)
                        lower, _ = disk_response_transport(shapes, -velocity)
                        predictions[f"candidate_{arm}"] = ((upper-lower)/(2*H)).numpy()
                        predictions["candidate_disk_moment_linear"] = values[:, None]*infinitesimal_response_matrix(shapes)[:, :, 0].numpy()
                    else:
                        predictions[f"candidate_{arm}"] = np.column_stack((values, np.zeros(len(values))))
            weight = stratum_weight(case, stratum)
            sums = decomposition_sums(plus[common], minus[common], weight)
            add_predictions(sums, plus[common], minus[common], predictions, weight)
            report.update(stratum=stratum, anchors=len(ids), rendered=len(rendered),
                          unrendered_dropped=len(ids)-len(rendered), common_usable=int(common.sum()),
                          plus_only_dropped=int((up & ~um).sum()), minus_only_dropped=int((um & ~up).sum()),
                          neither_usable_dropped=int((~up & ~um).sum()))
            counts.append(report)
            if combined is None:
                combined = sums
            else:
                for entry, addition in zip(combined, sums):
                    for direction in ("plus", "minus"):
                        for field in entry[direction]:
                            entry[direction][field] += addition[direction][field]
                    entry["enters_plus"] += addition["enters_plus"]
                    entry["enters_minus"] += addition["enters_minus"]
        responses = {}
        for entry in combined:
            terms = response_terms(entry)
            for name in recipes:
                terms[name] = sum(entry[d][name] / entry[d]["weight"] for d in ("plus", "minus")) / 2
            responses[entry["cut"]] = terms
        reports.append({"case": case, "counts": counts, "sufficient": combined, "response": responses})
        print(f"EMULATOR_AMPLITUDE case={case} uncut={responses[CUTS[0]]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"h": H, "cuts": CUTS, "cases": reports,
                            "recipes": recipes,
                            "candidate_models": {arm: candidate.metadata for arm, candidate in candidates.items()},
                            "candidate_output_proxy": "mean of physical +/- outputs, including shape for disk transport; no actual g0 leg" if candidates else None,
                            "script_sha256": file_sha256(__file__), "manifest_sha256": file_sha256(args.manifest),
                            "emulator_sha256": manifest["static_sha256"][manifest["emulator_model"]]})


def summarize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    products = [json.loads(p.read_text()) for p in sorted(args.parts.glob("block*.json"))]
    recipes = list(products[0].get("recipes", RECIPES))
    candidate_models = products[0].get("candidate_models", {})
    if any(list(p.get("recipes", RECIPES)) != recipes or p.get("candidate_models", {}) != candidate_models
           for p in products):
        raise ValueError("inconsistent candidate models or recipes")
    for field in ("script_sha256", "manifest_sha256", "emulator_sha256"):
        if len({p[field] for p in products}) != 1:
            raise ValueError(f"inconsistent {field}")
    cases = sorted((c for p in products for c in p["cases"]), key=lambda c: c["case"])
    if [c["case"] for c in cases] != list(range(400, 900)):
        raise ValueError("summary requires exact cases400--899")
    picks = np.random.default_rng(20260917).integers(len(cases), size=(10000, len(cases)))
    summary = {}
    for cut in CUTS:
        values = {name: np.array([c["response"][cut][name] for c in cases])
                  for name in ("total", "within", "boundary", *recipes)}
        entries = {}
        for name, array in values.items():
            boot = array[picks].mean(1)
            entries[name] = {"mean": array.mean(0), "case_bootstrap_se": boot.std(0, ddof=1)}
        for name in recipes:
            for target in ("total", "within"):
                delta = values[target] - values[name]
                boot = delta[picks].mean(1)
                entries[f"{target}_minus_{name}"] = {
                    "mean": delta.mean(0), "case_bootstrap_se": boot.std(0, ddof=1),
                    "ci95": np.quantile(boot, [.025, .975], axis=0)}
        summary[cut] = entries
        print(f"EMULATOR_AMPLITUDE_SUMMARY {cut} {entries}", flush=True)
    write_json(args.output, {"case_count": len(cases), "summary": summary, "cases": cases,
                            "candidate_models": candidate_models,
                            "candidate_output_proxy": products[0].get("candidate_output_proxy"),
                            "emulator_sha256": products[0]["emulator_sha256"],
                            "limitations": ["common usable identities and V2 anchor truth parent",
                                            "native scalar emulator response is compared with measured R11",
                                            "primary-shape proxy is diagnostic, not a fitted correction",
                                            "no ConstGold numerical transfer or calibration claim"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("compute")
    compute.add_argument("--case-start", type=int, required=True)
    compute.add_argument("--case-stop", type=int, required=True)
    compute.add_argument("--manifest", type=Path, required=True)
    compute.add_argument("--candidate-root", type=Path)
    compute.add_argument("--disk-candidate-root", type=Path)
    compute.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--parts", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (calculate if args.command == "compute" else summarize)(args)


if __name__ == "__main__":
    main()
