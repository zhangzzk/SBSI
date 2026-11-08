"""Circularized joint NLL plus paired shape, optionally radius, mean-change loss.

The parent, populations and forward shear legs are frozen. Pair-uniform response
supervision and validation use no bins or fitted covariance. The old grid is
used only for the final independent response audit and cross-run comparison.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np
import torch

from sbsi.flow_paired_shape import PairedResponsePopulation, paired_shape_backward, evaluate_paired_shape
from sbsi.flow_physical_response import physical_outputs, marginal_physical_response_statistics
from sbsi.flow_physical_case_validation import paired_physical_case_statistics, fixed_cell_average_statistics
from sbsi.flow_grid_accumulation import summarize_validation_means
from sbsi.measurement_model import TargetStandardizer
from sbsi.selection_model import TabularPreprocessor

_continuation = runpy.run_path(str(Path(__file__).with_name("continue_circularized_flow.py")))
load, digest, verify, write_json, continue_fit, nll_pass, audit_response = (
    _continuation[k] for k in ("load", "digest", "verify", "write_json", "continue_fit", "nll_pass", "audit_response"))


def paired_shape_epoch(model, values, context, population, optimizer, recipe, seed):
    model.train()
    order = torch.randperm(len(values), device=values.device,
                           generator=torch.Generator(device=values.device).manual_seed(seed))
    total, response_total, pairs_seen, norms = 0., 0., 0, []
    component_total = np.zeros(len(population.output_scale))
    started = time.monotonic()
    for step, start in enumerate(range(0, len(values), recipe["batch_size"])):
        rows = order[start:start+recipe["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        nll = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(nll)):
            raise ValueError("nonfinite NLL")
        nll.backward()
        response = paired_shape_backward(model, population, recipe["pairs_per_step"],
            recipe["training_draws"], chunk_size=recipe["response_training_chunk_size"],
            seed=recipe["training_response_seed"]+seed*1000003+step*100003,
            weight=recipe.get("response_weight", recipe.get("shape_weight", 0.)))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        norms.append(float(norm))
        total += float(nll.detach())*len(rows)
        response_total += response["loss"]*len(rows)
        component_total += response["component_losses"]*len(rows)
        pairs_seen += response["pairs_per_group"]
        if step == 0:
            print("FIRST_GRID_FREE_UPDATE", seed, float(nll.detach()), response["loss"], float(norm),
                  "COMPONENT_LOSSES", response["component_losses"].tolist(), flush=True)
    return dict(nll=total/len(values), response_loss=response_total/len(values), rows=len(values),
                response_component_losses=(component_total/len(values)).tolist(),
                optimizer_steps=len(norms), gradient_norms=norms,
                clipped_step_fraction=float(np.mean(np.asarray(norms)>5)),
                response_pairs_per_group=pairs_seen, seconds=time.monotonic()-started)


def train(root, manifest, resume):
    recipe = manifest["recipe"]
    radius_scale = recipe.get("radius_scale_pixels")
    response_groups = ["shape"] + ([] if radius_scale is None else ["radius"])
    base = Path(manifest["base_root"])
    paths = json.loads((base/"paths.json").read_text())
    parent, initial = load(manifest["parent_checkpoint"]), load(manifest["parent_optimizer"])
    if initial["epoch"] != parent["metadata"]["epoch"]:
        raise ValueError("optimizer and checkpoint epochs disagree")
    for key in initial["state_dict"]:
        torch.testing.assert_close(initial["state_dict"][key], parent["state_dict"][key], rtol=0, atol=0)
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    if "circularized_Re_input_p" not in condition.feature_names:
        raise ValueError("circularized condition required")
    target = TargetStandardizer.from_state(parent["target_transform"])
    data, populations, counts = {}, {}, {}
    validation_payload = None
    for split in ("train", "validation"):
        context = torch.as_tensor(np.load(base/f"prepared/{split}_context.npy"), device="cuda")
        raw = np.load(Path(paths["raw"])/f"{split}_raw.npy", mmap_mode="r")
        values = torch.as_tensor(target.transform_array(physical_outputs(raw)), device="cuda")
        payload = load(Path(paths["prepared"])/f"{split}.pt")
        pairs = np.asarray(payload["pairs"])
        delta = np.asarray(raw[pairs[:, 1], :2]-raw[pairs[:, 0], :2], dtype=np.float64)
        if radius_scale is not None:
            delta = np.column_stack((delta, np.exp(raw[pairs[:, 1], 3])-np.exp(raw[pairs[:, 0], 3])))
        populations[split] = PairedResponsePopulation(context, pairs, delta, payload["gamma"],
                                                     radius_scale=radius_scale)
        data[split] = values, context
        counts[split] = dict(nll_rows=len(values), response_pairs=len(pairs),
            shear_amplitude_range=[float(populations[split].amplitude.min()),
                                   float(populations[split].amplitude.max())],
            pair_weighting="uniform; original matched population retained",
            response_groups=response_groups, radius_scale_pixels=radius_scale)
        if split == "validation":
            validation_payload = payload
        del raw, delta, payload
    print("GRID_FREE_DATA_READY", counts, flush=True)
    indices = np.load(base/"training/validation_nll_indices.npy")
    np.testing.assert_array_equal(indices, np.sort(np.random.default_rng(900000504).choice(
        len(data["validation"][0]), 500000, replace=False)))
    vi = torch.as_tensor(indices, device="cuda")
    subset = np.sort(np.random.default_rng(recipe["subset_seed"]).choice(
        len(populations["validation"].pairs), recipe["validation_response_pairs"], replace=False))
    diagnostics = root/"paired_diagnostics"
    diagnostics.mkdir(exist_ok=resume)
    np.save(diagnostics/"validation_pair_indices.npy", subset)
    write_json(root/"data_report.json", counts)

    def diagnostic(model, epoch):
        result = evaluate_paired_shape(model, populations["validation"], subset,
            draws=recipe["validation_draws"], groups=recipe["subset_groups"],
            batch_size=recipe["response_batch_size"], seed=recipe["subset_response_seed"])
        np.savez(diagnostics/("epoch%03d.npz" % epoch), **result)
        return {k:result[k] for k in ("response_loss", "shape_loss", "radius_loss", "component_losses",
                                     "mc_jackknife_se", "shape_mc_jackknife_se", "radius_mc_jackknife_se",
                                     "mean_quadratic", "integration_quadratic", "pairs", "groups", "draws", "seed")
                if k in result}

    metadata = dict(training_phase="circularized_grid_free_paired_"+"_".join(response_groups), branch="paired",
        grid_response_weight=0., paired_response_weight=recipe.get("response_weight", recipe.get("shape_weight", 0.)), nll_weight=1.,
        response_groups=response_groups, response_pair_weighting="uniform", radius_scale_pixels=radius_scale,
        response_objective="E_pair[sum((Delta_mu-Delta_measurement)^2 / output_scale^2) / (4*abs(gamma)^2)]",
        parent=manifest["parent_checkpoint"], parent_optimizer=manifest["parent_optimizer"],
        manifest_sha256=digest(root/"manifest.json"), recipe=recipe,
        grid_used_for_selection=False, radius_and_flux_in_nll=True,
        radius_and_flux_response_diagnostic_only=radius_scale is None,
        flux_response_diagnostic_only=True, experimental=True, constgold_used=False)
    output = root/"paired"
    model, selection = continue_fit(output, "paired", parent, initial, None, data["train"],
        (data["validation"][0][vi], data["validation"][1][vi]), recipe, metadata, diagnostic,
        populations["train"], resume, train_epoch=paired_shape_epoch)
    full_nll = nll_pass(model, *data["validation"], recipe["batch_size"])
    # Bins and fitted target covariances first enter here, after checkpoint selection.
    case_design = dict(np.load(base/"prepared/validation_case_design.npz"))
    statistics = dict(np.load(Path(paths["targets"])/"validation_targets.npz"))
    response, detailed, crosses = audit_response(model, data["validation"][1], validation_payload,
                                                case_design, statistics, recipe)
    shape = marginal_physical_response_statistics(statistics, ("shape",))
    score = summarize_validation_means(detailed["group_model_responses"][..., :4],
                                      shape["target"], shape["precision"])
    np.savez(output/"response.npz", **detailed, model_cross_groups=crosses, normal=case_design["normal"])
    verify(root, manifest)
    write_json(output/"results.json", dict(selection=selection, full_nll=full_nll, response=response,
        shape_response_score={k:score[k] for k in ("loss", "mc_jackknife_se", "mean_quadratic", "integration_quadratic")}))
    print("GRID_FREE_TRAINING_COMPLETE", flush=True)


def compare(root, manifest):
    old = Path(manifest["comparison_root"])
    current_name = "paired_shape_radius" if manifest["recipe"].get("radius_scale_pixels") is not None else "paired_shape"
    paths = {"nll":old/"nll", "grid_shape":old/"shape", current_name:root/"paired"}
    if manifest.get("shape_only_comparison_root"):
        paths["paired_shape_only"] = Path(manifest["shape_only_comparison_root"])/"paired"
    results = {name:json.loads((path/"results.json").read_text()) for name,path in paths.items()}
    arrays = {name:dict(np.load(path/"response.npz")) for name,path in paths.items()}
    for name in (key for key in paths if key != current_name):
        a, b = arrays[current_name], arrays[name]
        np.testing.assert_array_equal(a["normal"], b["normal"])
        np.testing.assert_array_equal(a["measured_response"], b["measured_response"])
        for key in ("seed", "groups", "draws", "draw_count"):
            if results[current_name]["response"][key] != results[name]["response"][key]:
                raise ValueError("response audit streams/populations are not aligned")
        cross = a["model_cross_groups"]-b["model_cross_groups"]
        stats = paired_physical_case_statistics(a["normal"], np.zeros_like(cross[0]), cross)
        name = ("paired_shape_radius_minus_" if current_name == "paired_shape_radius" else "paired_minus_")+name
        np.savez(root/(name+".npz"), **stats)
        results[name] = fixed_cell_average_statistics(stats, np.full(len(stats["model_response"]), 1/len(stats["model_response"])))
    results["limitations"] = ["one training seed; finite capped optimization",
        "grid-free and grid losses differ in target, weighting, and coefficient units",
        "matched usable population; no selection or full SBSI inference calibration"]
    write_json(root/"results.json", results)
    print("GRID_FREE_COMPARISON_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "compare"), default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.root/"manifest.json").read_text())
    verify(args.root, manifest, data=not args.preflight)
    if args.preflight:
        print("GRID_FREE_PREFLIGHT_PASSED", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("training and full audits require Slurm")
    started = time.monotonic()
    if args.stage == "train":
        train(args.root, manifest, args.resume)
    else:
        compare(args.root, manifest)
    print("ELAPSED_SECONDS", time.monotonic()-started, flush=True)


if __name__ == "__main__":
    main()
