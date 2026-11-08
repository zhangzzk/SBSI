#!/usr/bin/env python3
"""Audit joint response on a truth parent and observed-g0 subcohorts.

Selections act on the joint generated outputs independently in each shear
leg. Only the truth_parent rows have no observed-g0 selection. The other two
cohorts diagnose conditioning effects and are not an unconditional density
validation. The fixed-g0-trained emulator is deliberately held fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from plots.constgold_property_bias import (
    BRANCHES, load_lookup, make_case_weights, response_from_arrays, summarize_branch,
)
from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.disk_response_transport import disk_response_transport
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment, OutputConditionedResponse, TRUTH_FEATURES, predict_scene_draw_responses
from sbsi.selection_model import load_selection_model_ensemble
from scripts.evaluate_constgold_fixed_g0_response import raw_root, shear_frames
from scripts.train_fixed_g0_flow import sha256, write_json
from scripts import truth_parent_cohorts
from scripts import joint_predictive_diagnostics as predictive


COHORTS = truth_parent_cohorts.COHORTS
CUTS = ("uncut", "mag25.8_radius_gt0.60", "mag25.8_radius_ge0.75")
FLUX_MINIMUM = 10 ** (-0.4 * (25.8 - 30))


def output_cuts(values):
    """The same joint-output predicates for observed and generated legs."""
    radius, flux = values[..., 2], values[..., 3]
    if isinstance(values, torch.Tensor):
        return torch.stack((
            torch.ones_like(radius, dtype=torch.bool),
            (radius > 3.0) & (flux > FLUX_MINIMUM),
            (radius >= 3.75) & (flux > FLUX_MINIMUM),
        ), dim=-1)
    return np.stack((np.ones_like(radius, dtype=bool),
                     (radius > 3.0) & (flux > FLUX_MINIMUM),
                     (radius >= 3.75) & (flux > FLUX_MINIMUM)), axis=-1)


def measured_leg(root, case, shear, ids):
    catalogue = raw_root(root, case, shear)
    cross = pd.read_feather(catalogue / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather",
                            columns=["id_detec", "id_input"])
    shape = pd.read_feather(
        catalogue / "Shapes" / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
        columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"],
    )
    if (cross["id_input"].duplicated().any() or cross["id_detec"].duplicated().any()
            or shape["NUMBER"].duplicated().any()):
        raise ValueError("nonunique measured identities")
    joined = cross.merge(shape, left_on="id_detec", right_on="NUMBER",
                         how="inner", validate="one_to_one").set_index("id_input")
    aligned = joined.reindex(ids)
    raw = aligned[["NGMIX_G1", "NGMIX_G2", "FLUX_RADIUS", "MAG_AUTO"]].to_numpy(float)
    values = raw.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        values[:, 3] = 10 ** (-0.4 * (raw[:, 3] - 30))
    usable = np.isfinite(values).all(1)
    usable &= np.square(values[:, :2]).sum(1) < 1
    usable &= (values[:, 2] > 0) & (values[:, 3] > 0)
    report = {"cross_rows": len(cross), "shape_rows": len(shape),
              "cross_without_shape": len(cross) - len(joined),
              "shape_without_cross": len(shape) - len(joined),
              "parent_rows": len(ids), "parent_usable": int(usable.sum()),
              "parent_absent_crossmatch": int((~np.isin(ids, cross["id_input"])).sum()),
              "usable_zero_shape_rows": int((usable & (values[:, :2] == 0).all(1)).sum())}
    return values, usable, report


def selected_draw_moments(values, blend_shift):
    """Integrate selection, flow shape, and external blend shift jointly."""
    keep = output_cuts(values).to(values.dtype)
    mass = keep.mean(1)
    flow = (keep[..., None] * values[..., None, :2]).mean(1)
    if blend_shift.shape == (len(values), 2):
        blend = mass[..., None] * blend_shift[:, None, :]
    elif blend_shift.shape == values.shape[:2] + (2,):
        blend = (keep[..., None] * blend_shift[:, :, None, :]).mean(1)
    else:
        raise ValueError("blend shift must align with primaries or joint draws")
    return mass, flow, blend


def load_candidate_pairs(root, case, parent_path, ids, reference_response):
    path = root / f"case{case:03d}.feather"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (metadata["case"] != case or metadata["anchor_sha256"] != sha256(parent_path)
            or metadata["sha256"] != sha256(path) or metadata["features"] != list(TRUTH_FEATURES)):
        raise ValueError("candidate response-pair provenance mismatch")
    pairs = pd.read_feather(path)
    if (pairs.duplicated(["index_input_p", "index_input_s"]).any()
            or not np.isin(pairs["index_input_p"], ids).all()
            or not np.isfinite(pairs[[*TRUTH_FEATURES, "response"]].to_numpy()).all()):
        raise ValueError("invalid candidate response pairs")
    recovered = pairs.groupby("index_input_p")["response"].sum().reindex(ids, fill_value=0).to_numpy()
    np.testing.assert_allclose(recovered, reference_response, rtol=1e-12, atol=1e-12,
                               err_msg="candidate pairs do not reproduce the frozen baseline emulator")
    return pairs.sort_values("index_input_p", kind="stable").reset_index(drop=True), metadata


def statistics(ncases, cohorts=COHORTS):
    return {f"{cohort}/{cut}/{branch}": {
        source: {direction: {
            "denominator": np.zeros((ncases, 1)),
            "numerator": np.zeros((ncases, 1, 2)),
            **({"flow_numerator": np.zeros((ncases, 1, 2)),
                "blend_numerator": np.zeros((ncases, 1, 2))} if source == "model" else {}),
        } for direction in ("plus", "minus")} for source in ("measured", "model")
    } for cohort in cohorts for cut in CUTS for branch in BRANCHES}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, action="append", required=True)
    parser.add_argument("--rblend", type=Path, action="append", required=True)
    parser.add_argument("--response-candidate", type=Path,
                        help="Explicit experimental response model; leaves configured likelihood unchanged")
    parser.add_argument("--pair-root", type=Path)
    parser.add_argument("--response-threads", type=int, default=4)
    parser.add_argument("--response-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--response-composition", choices=("additive", "mobius"), default="additive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS,
                        help="Explicitly select truth_parent for endpoints without a g0 catalogue")
    parser.add_argument("--predictive-diagnostics", action="store_true",
                        help="Record fixed joint-output histograms and moments on actual-U legs")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold parent audit must run under Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.response_candidate is None) != (args.pair_root is None):
        raise ValueError("response candidate and frozen pair root must be supplied together")
    if args.response_composition == "mobius" and args.response_candidate is None:
        raise ValueError("Mobius composition requires a separately fitted disk-response artifact")
    if (len(set(args.cases)) != len(args.cases) or len(args.cases) < 2
            or args.h <= 0 or args.draws < 2 or args.draws % 2 or args.batch_size < 1):
        raise ValueError("distinct cases, positive step/batch, and even draws required")
    cases = tuple(sorted(args.cases))
    manifest_path = args.parent_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requested_cohorts = truth_parent_cohorts.resolve_cohorts(manifest, args.cohorts)
    if not set(cases) <= set(manifest["cases"]):
        raise ValueError("parent manifest does not cover cases")
    lookup = load_lookup(tuple(args.rblend), cases)
    response_class = DiskResponseMoment if args.response_composition == "mobius" else OutputConditionedResponse
    candidate = (None if args.response_candidate is None else
                 response_class(args.response_candidate, nthread=args.response_threads, device=args.response_device))
    candidate_reports = {}
    bundle = load_measurement_model(args.flow, device=torch.device("cuda"))
    bundle.model.eval()
    classifier = load_selection_model_ensemble(args.classifier, device=torch.device("cuda"))
    if (bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES)
            or classifier.preprocessor.feature_names != list(FLOW_FEATURES)
            or bundle.target_transform.target_names != list(FLOW_TARGETS)):
        raise ValueError("incompatible model contracts")
    stats = statistics(len(cases), requested_cohorts)
    predictive_stats = ({source: {direction: predictive.empty_statistics(len(cases), len(CUTS))
                                 for direction in ("plus", "minus")}
                         for source in ("observed", "generated")} if args.predictive_diagnostics else None)
    reports = {}
    for ci, case in enumerate(cases):
        case_started = time.monotonic()
        path = args.parent_root / f"case{case:03d}.npz"
        if sha256(path) != manifest["case_reports"][str(case)]["sha256"]:
            raise ValueError("parent artifact changed after preparation")
        with np.load(path, allow_pickle=False) as stored:
            ids, context = stored["input_index"], stored["context"].astype(np.float64)
            cohorts = truth_parent_cohorts.parent_cohort_masks(stored, requested_cohorts)
        if not np.array_equal(ids, lookup[case]["input_index"]):
            raise ValueError("blending lookup does not exactly cover the truth parent")
        if candidate is not None:
            if not np.all(ids[1:] > ids[:-1]):
                raise ValueError("candidate pair slicing requires sorted unique parent identities")
            pairs, pair_metadata = load_candidate_pairs(args.pair_root, case, path, ids, lookup[case]["R_blend"])
            pair_ids = pairs["index_input_p"].to_numpy()
            candidate_reports[str(case)] = {"pairs": pair_metadata, "draw_prediction_counts": {}}
            if not candidate.conditioned:
                predicted = candidate.predict_pairs(pairs, None, None)
                static_response = pd.Series(predicted, index=pair_ids).groupby(level=0).sum().reindex(ids, fill_value=0).to_numpy()
        frames = dict(zip(("plus", "minus"), shear_frames(context, args.h)))
        legs = {direction: measured_leg(args.constgold_root, case, sign * args.h, ids)
                for direction, sign in (("plus", 1), ("minus", -1))}
        common = legs["plus"][1] & legs["minus"][1]
        probabilities = {direction: np.asarray(classifier.predict_proba(frame), dtype=float)
                         for direction, frame in frames.items()}
        for probability in probabilities.values():
            if (probability.shape != (len(ids),) or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise ValueError("invalid modeled usability")
        for direction in ("plus", "minus"):
            values, usable, _ = legs[direction]
            cuts = output_cuts(values)
            if predictive_stats is not None:
                contribution = predictive.output_statistics(values, usable.astype(float), cuts)
                for field, value in contribution.items():
                    predictive_stats["observed"][direction][field][ci] += value
            for cohort, mask in cohorts.items():
                for ki, cut in enumerate(CUTS):
                    for branch in BRANCHES:
                        use = mask & cuts[:, ki] & (common if branch == "matched_usable" else usable)
                        entry = stats[f"{cohort}/{cut}/{branch}"]["measured"][direction]
                        entry["denominator"][ci, 0] = use.sum()
                        entry["numerator"][ci, 0] = values[use, :2].sum(0)
        contexts = [bundle.context_tensor(frames[d]) for d in ("plus", "minus")]
        shifts = {d: lookup[case]["R_blend"][:, None] * (frames[d].iloc[:, :2].to_numpy() - context[:, :2])
                  for d in ("plus", "minus")}
        for block, start in enumerate(range(0, len(ids), args.batch_size)):
            stop = min(len(ids), start + args.batch_size)
            generated = physical_context_draws(
                bundle.model, [x[start:stop] for x in contexts], args.draws,
                args.seed + 10000019 * case + 1000003 * block,
            )
            for direction, values in zip(("plus", "minus"), generated):
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("nonfinite generated measurements")
                shift = shifts[direction][start:stop]
                if candidate is not None:
                    if candidate.conditioned:
                        lo = np.searchsorted(pair_ids, ids[start], side="left")
                        hi = np.searchsorted(pair_ids, ids[stop-1], side="right")
                        response, counts = predict_scene_draw_responses(
                            candidate, pairs.iloc[lo:hi], ids[start:stop], values.cpu().numpy())
                        total_counts = candidate_reports[str(case)]["draw_prediction_counts"]
                        for key, count in counts.items():
                            total_counts[key] = total_counts.get(key, 0) + count
                    else:
                        response = static_response[start:stop, None]
                    if candidate.response_coordinate == "disk_velocity":
                        shear = torch.tensor([args.h if direction == "plus" else -args.h, 0.],
                                             dtype=torch.float64, device=values.device)
                        velocity = torch.as_tensor(response, dtype=torch.float64, device=values.device)[..., None]*shear
                        transformed, logdet = disk_response_transport(values[..., :2], velocity)
                        if not bool(torch.isfinite(transformed).all() and torch.isfinite(logdet).all()):
                            raise ValueError("nonfinite disk-response transport or Jacobian")
                        shift = transformed-values[..., :2]
                        cuts = output_cuts(values)
                        before = values[..., :2].square().sum(-1)
                        after = transformed.square().sum(-1)
                        support = candidate_reports[str(case)].setdefault("support", {})
                        for ki, cut in enumerate(CUTS):
                            row = support.setdefault(f"{direction}/{cut}", {"selected_draws": 0,
                                        "baseline_strict_invalid": 0, "transport_strict_invalid": 0,
                                        "outside_by_more_than_1e-12": 0, "maximum_squared_norm_excess": 0.})
                            use = cuts[..., ki]
                            row["selected_draws"] += int(use.sum())
                            row["baseline_strict_invalid"] += int(((before >= 1) & use).sum())
                            row["transport_strict_invalid"] += int(((after >= 1) & use).sum())
                            row["outside_by_more_than_1e-12"] += int(((after > 1+1e-12) & use).sum())
                            if bool(use.any()):
                                row["maximum_squared_norm_excess"] = max(row["maximum_squared_norm_excess"], float(after[use].max()-1))
                    else:
                        delta = frames[direction].iloc[start:stop, :2].to_numpy() - context[start:stop, :2]
                        shift = response[..., None]*delta[:, None, :]
                        if not candidate.conditioned:
                            shift = shift[:, 0]
                if predictive_stats is not None:
                    joint_values = values.clone()
                    diagnostic_shift = torch.as_tensor(shift, dtype=values.dtype, device=values.device)
                    if diagnostic_shift.ndim == 2:
                        diagnostic_shift = diagnostic_shift[:, None, :]
                    joint_values[..., :2] += diagnostic_shift
                    draw_weights = torch.as_tensor(legs[direction][1][start:stop],
                                                   dtype=torch.float64, device=values.device)
                    draw_weights = (draw_weights[:, None] / args.draws).expand(values.shape[:2])
                    contribution = predictive.output_statistics(
                        joint_values, draw_weights, output_cuts(joint_values))
                    for field, value in contribution.items():
                        predictive_stats["generated"][direction][field][ci] += value
                mass, flow, blend = selected_draw_moments(values, torch.as_tensor(shift, device="cuda"))
                mass, flow, blend = [x.cpu().numpy() for x in (mass, flow, blend)]
                for cohort, mask in cohorts.items():
                    for branch in BRANCHES:
                        if branch == "matched_usable":
                            weight = common[start:stop].astype(float)
                        elif branch == "actual_usable_flags":
                            weight = legs[direction][1][start:stop].astype(float)
                        else:
                            weight = probabilities[direction][start:stop]
                        weight = weight * mask[start:stop]
                        for ki, cut in enumerate(CUTS):
                            entry = stats[f"{cohort}/{cut}/{branch}"]["model"][direction]
                            entry["denominator"][ci, 0] += np.dot(weight, mass[:, ki])
                            flow_sum = (weight[:, None] * flow[:, ki]).sum(0)
                            blend_sum = (weight[:, None] * blend[:, ki]).sum(0)
                            entry["flow_numerator"][ci, 0] += flow_sum
                            entry["blend_numerator"][ci, 0] += blend_sum
                            entry["numerator"][ci, 0] += flow_sum + blend_sum
            if candidate is not None and (block == 0 or (block+1) % 25 == 0):
                print(f"TRUTH_PARENT_CANDIDATE case={case} rows={stop}/{len(ids)} "
                      f"elapsed={time.monotonic()-case_started:.1f}s", flush=True)
        reports[str(case)] = {"cohort_counts": {name: int(mask.sum()) for name, mask in cohorts.items()},
                              "legs": {d: legs[d][2] for d in legs},
                              "matched_usable": int(common.sum())}
        print(f"TRUTH_PARENT_RESPONSE case={case} counts={reports[str(case)]['cohort_counts']}", flush=True)
    weights = make_case_weights(10000, len(cases), 20260914)
    summaries = {}
    for key, entry in stats.items():
        result = summarize_branch(entry, weights, args.h)
        summary = {name: value[0] for name, value in result.items()}
        for component in ("flow", "blend"):
            terms = {d: {"denominator": entry["model"][d]["denominator"],
                         "numerator": entry["model"][d][f"{component}_numerator"]}
                     for d in ("plus", "minus")}
            summary[f"model_{component}_response"] = response_from_arrays(
                terms["plus"], terms["minus"], args.h,
            )[0]
        np.testing.assert_allclose(summary["model_flow_response"] + summary["model_blend_response"],
                                   summary["model_response"], atol=1e-11)
        summaries[key] = summary
        if key.endswith("actual_usable_flags"):
            print(f"TRUTH_PARENT_RESULT {key} m={summary['m_percent']:+.6f}% "
                  f"se={summary['m_standard_error_percentage_points']:.6f}%", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "cases": cases, "h": args.h, "draws": args.draws, "seed": args.seed,
        "population": manifest["population"], "cohorts": requested_cohorts, "cuts": CUTS,
        "usability": "unique raw crossmatch; finite ngmix, radius and flux; |e|<1; radius>0; flux>0",
        "case_reports": reports, "summary": summaries, "per_case_sufficient": stats,
        "predictive_diagnostics": None if predictive_stats is None else {
            "metadata": predictive.metadata(), "cuts": CUTS,
            "summary": {direction: predictive.summarize_statistics(
                predictive_stats["observed"][direction], predictive_stats["generated"][direction], weights)
                for direction in ("plus", "minus")},
            "per_case": predictive_stats,
        },
        "experimental_response": None if candidate is None else {
            "metadata": candidate.metadata, "case_reports": candidate_reports,
            "prediction_device": args.response_device,
            "output_conditions": "same generated joint draw in each +/- leg; no observed photometry",
            "composition": ("disk Mobius transport with radial_tanh(pair_sum_b*external_shear)" if args.response_composition == "mobius"
                            else "scalar response times unchanged primary intrinsic-shape shear increment"),
        },
        "inputs": {"parent_manifest_sha256": sha256(manifest_path),
                   "flow": str(args.flow), "flow_sha256": sha256(args.flow),
                   "classifiers": [{"path": str(p), "sha256": sha256(p)} for p in args.classifier],
                   "rblend": [{"path": str(p), "sha256": sha256(p)} for p in args.rblend],
                   "script_sha256": sha256(Path(__file__)),
                   "cohort_helper_sha256": sha256(Path(truth_parent_cohorts.__file__)),
                   "predictive_helper_sha256": sha256(Path(predictive.__file__))},
        "limitations": ["existing simulation endpoint; not fresh acceptance",
                        *manifest.get("limitations", []),
                        *( ["fixed-g0-trained emulator extrapolated outside its fit cohort"] if candidate is None else
                           ["experimental forward0.2 pair-response labels; both-leg usability and valid-g0 conditioning",
                            "output-conditioned scalar composition changes shape only; not a finite-shear density validation"] ),
                        "only truth_parent has no observed-g0 conditioning",
                        "matched_usable additionally conditions on both measured shear legs",
                        "one training seed and one latent stream; case bootstrap excludes these uncertainties",
                        "classifier excludes zero-shape sentinels; their counts are explicitly reported here"],
    })
    print(f"TRUTH_PARENT_RESPONSE_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
