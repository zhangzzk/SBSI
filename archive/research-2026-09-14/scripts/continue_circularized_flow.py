"""Resume circularized NLL training or add a four-component shape response loss.

Both branches restore the same final NLL weights and AdamW state. NLL preserves
its scheduler and epoch seeds; shape resets only the objective-dependent
scheduler. All heavy work runs under Slurm against a frozen manifest.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import time

import numpy as np
import torch

from sbsi.flow_grid_accumulation import sampled_response_backward, summarize_validation_means
from sbsi.flow_physical_case_validation import paired_physical_case_statistics, fixed_cell_average_statistics
from sbsi.flow_physical_population import PhysicalResponsePopulation, evaluate_physical_groups
from sbsi.flow_physical_response import marginal_physical_response_statistics, physical_outputs, fit_physical_response
from sbsi.measurement_model import build_flow, TargetStandardizer, save_measurement_model
from sbsi.selection_model import TabularPreprocessor

_trial = runpy.run_path(str(Path(__file__).with_name("trial_circularized_flow.py")))
digest, load, write_json, verify = (_trial[name] for name in ("digest", "load", "write_json", "verify"))
audit_response = _trial["audit_response"]
nll_pass = runpy.run_path(str(Path(__file__).with_name("train_flow_disk_nll.py")))["nll_pass"]


def shape_epoch(model, values, context, population, optimizer, recipe, seed):
    """Same NLL order/batches as nll_pass, adding an independent response gradient."""
    model.train()
    order = torch.randperm(len(values), device=values.device,
                           generator=torch.Generator(device=values.device).manual_seed(seed))
    total, response_total, pairs_seen, norms = 0., 0., 0, []
    started = time.monotonic()
    for step, start in enumerate(range(0, len(values), recipe["batch_size"])):
        rows = order[start:start+recipe["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        nll = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(nll)):
            raise ValueError("nonfinite NLL")
        nll.backward()
        result = sampled_response_backward(
            model, population, recipe["cells_per_step"], recipe["pairs_per_cell"],
            recipe["training_draws"], chunk_size=recipe["response_batch_size"],
            seed=recipe["training_response_seed"]+seed*1000003+step*100003,
            weight=recipe["shape_weight"])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        norms.append(float(norm))
        total += float(nll.detach())*len(rows)
        response_total += result["loss"]*len(rows)
        pairs_seen += result["pairs_per_group"]
        if step == 0:
            print("FIRST_SHAPE_UPDATE", seed, float(nll.detach()), result["loss"], float(norm), flush=True)
    return dict(nll=total/len(values), response_loss=response_total/len(values), rows=len(values),
                optimizer_steps=len(norms), gradient_norms=norms,
                clipped_step_fraction=float(np.mean(np.asarray(norms)>5)),
                response_pairs_per_group=pairs_seen, seconds=time.monotonic()-started)


def continue_fit(output, branch, parent, initial, incumbent, train, validation,
                 recipe, metadata, diagnostic, population=None, resume=False,
                 train_epoch=None):
    """Persist exact optimizer continuation and keep checkpoint selection explicit."""
    if branch not in ("nll", "shape", "paired") or (branch != "nll" and population is None):
        raise ValueError("valid branch and shape population required")
    if branch == "paired" and not callable(train_epoch):
        raise ValueError("paired branch requires an explicit grid-free epoch function")
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(exist_ok=True)
    model = build_flow(parent["model_config"]).to(train[0].device)
    model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=4, threshold=1e-4,
        threshold_mode="abs", min_lr=recipe["min_lr"])
    if branch == "nll":
        scheduler.load_state_dict(copy.deepcopy(initial["scheduler"]))
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    target = TargetStandardizer.from_state(parent["target_transform"])
    start_epoch = int(initial["epoch"])
    history = []
    if branch == "nll":
        best_epoch = int(initial["best_epoch"])
        best_metric = initial["history"][best_epoch]["validation"]["nll"]
        best_path = output/("epoch%03d.pt" % best_epoch)
        if not best_path.exists():
            shutil.copy2(incumbent, best_path)
    else:
        best_epoch, best_metric = start_epoch, float("inf")
    if resume:
        state = load(output/"optimizer.pt")
        if state["metadata"] != metadata:
            raise ValueError("resume metadata/recipe changed")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start_epoch = state["epoch"]
        best_epoch, best_metric = state["best_epoch"], state["best_metric"]
    else:
        for key, value in initial["state_dict"].items():
            torch.testing.assert_close(model.state_dict()[key].cpu(), value.cpu(), rtol=0, atol=0)
    last_epoch = start_epoch
    plateau = False
    response_weight = recipe.get("response_weight", recipe.get("shape_weight", 0.))
    response_loss_key = recipe.get("response_loss_key", "shape_loss")
    for epoch in range(start_epoch + int(resume), recipe["max_epoch"]+1):
        rate = optimizer.param_groups[0]["lr"]
        trained = None
        if epoch > start_epoch:
            epoch_seed = recipe["seed"]+epoch*1000003
            indices = np.random.default_rng(epoch_seed).choice(
                len(train[0]), min(recipe["rows_per_epoch"], len(train[0])), replace=False)
            ids = torch.as_tensor(indices, device=train[0].device)
            if branch == "nll":
                trained = nll_pass(model, train[0][ids], train[1][ids], recipe["batch_size"],
                                   optimizer=optimizer, seed=epoch_seed, clip=5.)
            else:
                epoch_function = shape_epoch if train_epoch is None else train_epoch
                trained = epoch_function(model, train[0][ids], train[1][ids], population,
                                         optimizer, recipe, epoch_seed)
        validated = nll_pass(model, *validation, recipe["batch_size"])
        if epoch == start_epoch and not resume:
            np.testing.assert_allclose(validated["nll"], initial["history"][-1]["validation"]["nll"],
                                       rtol=1e-6, atol=1e-7,
                                       err_msg="restored parent NLL differs on the fixed validation rows")
        response = None
        if branch != "nll" or epoch == start_epoch or epoch % recipe["diagnostic_interval"] == 0:
            response = diagnostic(model, epoch)
        metric = validated["nll"] + (response_weight*response[response_loss_key] if branch != "nll" else 0.)
        if not np.isfinite(metric):
            raise ValueError("nonfinite validation objective")
        if metric < best_metric:
            best_epoch, best_metric = epoch, metric
        record = dict(epoch=epoch, lr=rate, train=trained, validation=validated,
                      response=response, selection_metric=metric)
        history.append(record)
        path = output/("epoch%03d.pt" % epoch)
        temporary = path.with_suffix(".tmp")
        save_measurement_model(temporary, model, condition, target, parent["model_config"],
                               metadata | dict(epoch=epoch, record=record))
        os.replace(temporary, path)
        # The parent scheduler already consumed its final NLL. Do not step it twice.
        if trained is not None or branch != "nll":
            scheduler.step(metric)
        state = dict(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), history=history, best_epoch=best_epoch,
                     best_metric=best_metric, metadata=metadata)
        torch.save(state, output/"optimizer.tmp")
        os.replace(output/"optimizer.tmp", output/"optimizer.pt")
        write_json(output/"history.json", dict(history=history, best_epoch=best_epoch))
        last_epoch = epoch
        print("CONTINUATION_EPOCH", branch, epoch, "NLL", validated["nll"], "objective", metric,
              "lr", rate, "best", best_epoch, flush=True)
        plateau = (epoch-best_epoch >= recipe["early_stopping_patience"]
                   and optimizer.param_groups[0]["lr"] <= recipe["min_lr"])
        if plateau:
            break
    if history[-1]["response"] is None:
        diagnostic(model, last_epoch)
    selected = output/("epoch%03d.pt" % best_epoch)
    model.load_state_dict(load(selected)["state_dict"])
    selected_diagnostic = diagnostic(model, best_epoch)
    result = dict(best_epoch=best_epoch, final_epoch=last_epoch, best_checkpoint=str(selected),
                  best_sha256=digest(selected), best_metric=best_metric,
                  selection=("validation NLL" if branch == "nll" else
                             "validation NLL + lambda * paired shape+radius response loss" if branch == "paired" and response_loss_key == "response_loss" else
                             "validation NLL + lambda * paired shape response loss" if branch == "paired" else
                             "validation NLL + lambda * shape response loss"),
                  plateau_criterion_reached=plateau, convergence_established=False,
                  stop_reason="no improvement at minimum LR" if plateau else "epoch cap",
                  selected_subset_response=selected_diagnostic,
                  final_learning_rate=optimizer.param_groups[0]["lr"])
    write_json(output/"selection.json", result)
    return model, result


def validation_subset(payload, raw, case_design, full_statistics, recipe):
    """Fixed stratified pairs, with their own measured target and full-cell precision."""
    bins = np.asarray(payload["bin_ids"])
    rng = np.random.default_rng(recipe["subset_seed"])
    count = recipe["validation_response_pairs"]
    nc = len(payload["counts"])
    selected = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(bins == cell), count//nc + int(cell<count%nc), replace=False)
        for cell in range(nc)]))
    pairs = np.asarray(payload["pairs"])[selected]
    delta = physical_outputs(raw[pairs[:, 1]])-physical_outputs(raw[pairs[:, 0]])
    statistics = fit_physical_response(
        delta, np.asarray(payload["gamma"])[selected], bins[selected],
        np.asarray(payload["pair_keys"])[selected, 0], nc,
        scalar_gamma=case_design["scalar_gamma"][selected])
    return selected, statistics


def train(root, manifest, branch, resume=False):
    base = Path(manifest["base_root"])
    base_manifest = json.loads((base/"manifest.json").read_text())
    paths = base_manifest["paths"]
    recipe = manifest["branches"][branch]
    parent = load(manifest["parent_checkpoint"])
    initial = load(manifest["parent_optimizer"])
    if initial["epoch"] != parent["metadata"]["epoch"]:
        raise ValueError("optimizer and final checkpoint epochs disagree")
    for key in initial["state_dict"]:
        torch.testing.assert_close(initial["state_dict"][key], parent["state_dict"][key], rtol=0, atol=0)
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    if "circularized_Re_input_p" not in condition.feature_names:
        raise ValueError("circularized condition required")
    target = TargetStandardizer.from_state(parent["target_transform"])
    train_context = torch.as_tensor(np.load(base/"prepared/train_context.npy"), device="cuda")
    raw = np.load(Path(paths["raw"])/"train_raw.npy", mmap_mode="r")
    train_values = torch.as_tensor(target.transform_array(physical_outputs(raw)), device="cuda")
    del raw
    val_context = torch.as_tensor(np.load(base/"prepared/validation_context.npy"), device="cuda")
    raw_val = np.load(Path(paths["raw"])/"validation_raw.npy", mmap_mode="r")
    val_values = torch.as_tensor(target.transform_array(physical_outputs(raw_val)), device="cuda")
    val_payload = load(Path(paths["prepared"])/"validation.pt")
    case_design = dict(np.load(base/"prepared/validation_case_design.npz"))
    full_statistics = dict(np.load(Path(paths["targets"])/"validation_targets.npz"))
    ids = np.load(base/"training/validation_nll_indices.npy")
    np.testing.assert_array_equal(ids, np.sort(np.random.default_rng(900000504).choice(len(val_values), 500000, replace=False)))
    vi = torch.as_tensor(ids, device="cuda")
    subset, subset_statistics = validation_subset(val_payload, raw_val, case_design, full_statistics, recipe)
    val_population = PhysicalResponsePopulation(
        val_context, np.asarray(val_payload["pairs"])[subset], np.asarray(val_payload["bin_ids"])[subset],
        marginal_physical_response_statistics(subset_statistics, ("shape",)),
        marginal_physical_response_statistics(full_statistics, ("shape",))["precision"])
    population = None
    if branch == "shape":
        payload = load(Path(paths["prepared"])/"train.pt")
        stats = marginal_physical_response_statistics(dict(np.load(Path(paths["targets"])/"train_targets.npz")), ("shape",))
        population = PhysicalResponsePopulation(train_context, payload["pairs"], payload["bin_ids"], stats)
        del payload, stats
    output = root/branch
    diagnostics = root/(branch+"_diagnostics")
    diagnostics.mkdir(exist_ok=resume)
    np.savez(diagnostics/"subset.npz", selected_pair_indices=subset, **subset_statistics)

    def diagnostic(model, epoch):
        model.eval()
        result = evaluate_physical_groups(model, val_population, draws=recipe["validation_draws"],
            groups=recipe["subset_groups"], batch_size=recipe["response_batch_size"], seed=recipe["subset_response_seed"])
        np.savez(diagnostics/("epoch%03d.npz" % epoch), **result)
        return dict(shape_loss=result["loss"], mc_jackknife_se=result["mc_jackknife_se"],
                    shape_mean_quadratic=result["mean_quadratic"],
                    model_response=result["all_physical_group_means"].mean((0, 1)).tolist(),
                    target_response=subset_statistics["target"].mean(0).tolist(),
                    nonpositive_scalar_draws=result["nonpositive_scalar_draws"].tolist(),
                    material_disk_exterior=int(result["material_disk_exterior"]))

    metadata = dict(training_phase="circularized_"+branch+"_continuation", branch=branch,
                    grid_response_weight=recipe["shape_weight"], nll_weight=1.,
                    response_groups=["shape"] if branch=="shape" else [],
                    parent=manifest["parent_checkpoint"], parent_optimizer=manifest["parent_optimizer"],
                    manifest_sha256=digest(root/"manifest.json"), recipe=recipe,
                    radius_and_flux_in_nll=True, radius_and_flux_response_diagnostic_only=True,
                    experimental=True, constgold_used=False)
    print("CONTINUATION_DATA_READY", branch, len(train_values), len(val_values), flush=True)
    model, selection = continue_fit(output, branch, parent, initial, manifest["incumbent_checkpoint"],
        (train_values, train_context), (val_values[vi], val_context[vi]), recipe, metadata,
        diagnostic, population, resume)
    full_nll = nll_pass(model, val_values, val_context, recipe["batch_size"])
    response, detailed, crosses = audit_response(model, val_context, val_payload, case_design,
                                                 full_statistics, recipe)
    shape_statistics = marginal_physical_response_statistics(full_statistics, ("shape",))
    shape_score = summarize_validation_means(detailed["group_model_responses"][..., :4],
                                              shape_statistics["target"], shape_statistics["precision"])
    np.savez(output/"response.npz", **detailed, model_cross_groups=crosses, normal=case_design["normal"])
    verify(root, manifest)
    write_json(output/"results.json", dict(selection=selection, full_nll=full_nll, response=response,
               shape_response_score={key:shape_score[key] for key in ("loss", "mc_jackknife_se", "mean_quadratic", "integration_quadratic")}))
    print("CIRCULARIZED_CONTINUATION_COMPLETE", branch, flush=True)


def compare(root, manifest):
    results = {branch:json.loads((root/branch/"results.json").read_text()) for branch in ("nll", "shape")}
    arrays = {branch:dict(np.load(root/branch/"response.npz")) for branch in results}
    np.testing.assert_array_equal(arrays["shape"]["normal"], arrays["nll"]["normal"])
    np.testing.assert_array_equal(arrays["shape"]["measured_response"], arrays["nll"]["measured_response"])
    for key in ("response_seed", "validation_groups", "validation_draws", "response_batch_size"):
        if manifest["branches"]["nll"][key] != manifest["branches"]["shape"][key]:
            raise ValueError("CRN comparison requires identical audit recipe")
    cross = arrays["shape"]["model_cross_groups"]-arrays["nll"]["model_cross_groups"]
    statistics = paired_physical_case_statistics(arrays["nll"]["normal"], np.zeros_like(cross[0]), cross)
    nc = len(statistics["model_response"])
    results["shape_minus_nll"] = dict(full_nll=results["shape"]["full_nll"]["nll"]-results["nll"]["full_nll"]["nll"],
        equal_cell_response=fixed_cell_average_statistics(statistics, np.full(nc,1/nc)))
    np.savez(root/"paired_branch_difference.npz", **statistics)
    write_json(root/"results.json", results)
    print("CIRCULARIZED_FOLLOWUP_COMPARISON_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--branch", choices=("nll", "shape", "compare"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.root/"manifest.json").read_text())
    verify(args.root, manifest, data=not args.preflight)
    if args.preflight:
        print("CIRCULARIZED_CONTINUATION_PREFLIGHT_PASSED", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("training and full audits require Slurm")
    started = time.monotonic()
    if args.branch == "compare":
        compare(args.root, manifest)
    else:
        train(args.root, manifest, args.branch, args.resume)
    print("ELAPSED_SECONDS", time.monotonic()-started, flush=True)


if __name__ == "__main__":
    main()
