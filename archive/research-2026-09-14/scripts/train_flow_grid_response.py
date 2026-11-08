"""Prepare matched response covariances, then fine-tune a plain flow on NLL+lambda*Lgrid.

Run preparation and training under the scheduler. No ConstGold data are read.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.flow_coupling_target import coupling_bin_ids
from sbsi.flow_grid_response import fit_grid_response, GridResponsePopulation
from sbsi.flow_ratio import verify_ratio_base_properties
from sbsi.flow_training import (
    load_measurement_data, parse_args as base_args, transforms_from_reference,
)
from sbsi.flow_transport import matched_transport_indices
from sbsi.measurement_model import build_flow, save_measurement_model
from sbsi.selection_model import TabularPreprocessor
from sbsi.measurement_model import TargetStandardizer
from sbsi.training import GPUBatches, split_data, epoch_nll


DEFINITION = {
    "loss": "NLL + lambda/2 * sum_B (mu_A-target)^T precision_B (mu_C-target)",
    "components": ["R11", "R12", "R21", "R22"],
    "response": "paired forward regression delta_e = R gamma; physical ngmix shapes",
    "covariance": "delete-one-case jackknife covariance of each fitted cell matrix",
    "precision": "fixed inverse of simulation covariance; no flooring or clipping",
    "stochastic_estimator": "independent with-replacement galaxies and latent groups A,C; antithetic CRN within each group",
    "cell_weighting": "sum over cells, no occupancy weighting or cell-count normalization",
    "nll_population": "unit weight per finite usable leg row, unmatched rows retained",
    "response_population": "one-to-one matched galaxies usable in both legs; zero destination shear excluded",
    "limitations": "finite forward secant, matched selection, fixed grid; cross-cell covariance omitted",
    "validation": "separate held-out case targets/covariances; full matched population, fixed independent MC groups",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if any((root/name).exists() for name in ("train.pt", "validation.pt", "manifest.json")):
        raise FileExistsError("refusing to overwrite prepared data")
    reference = load_torch(args.parent)
    if reference["model_config"]["flow_type"] != "affine":
        raise ValueError("grid response requires the plain affine parent")
    condition = TabularPreprocessor.from_state(reference["condition_preprocessor"])
    target = TargetStandardizer.from_state(reference["target_transform"])
    if target.target_names != ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto", "measured_log_flux_radius"]:
        raise ValueError("unexpected parent target order")
    # Reuse the production population loader with an explicit new-data mode.
    recipe = base_args([
        "--catalogue", args.g0, "--additional-catalogue", args.g1,
        "--output", str(root/"unused.pt"), "--fold-catalogue-shear",
        "--max-cases", "200", "--expected-case-count", "200", "--max-rows", "0",
        "--validation-group-column", "case", "--validation-size", "0.2",
        "--seed", "501", "--primary-mag-max", "25.8", "--primary-re-min", "0.37",
    ])
    recipe.grid_response = True
    frame = load_measurement_data(recipe, condition.feature_names, target.target_names, data_seed=501)
    train, validation = split_data(frame, 501, .2, group_column="case")
    del frame
    condition, target = transforms_from_reference(
        reference, condition.feature_names, target.target_names, condition.log_features)
    with np.load(args.grid) as grid:
        edges = [grid[k].copy() for k in ("edges_flux", "edges_size", "edges_crowd")]
        expected_cases = {"train": grid["train_cases"].copy(), "validation": grid["validation_cases"].copy()}
    n_cells = int(np.prod([len(e)-1 for e in edges]))
    manifest = dict(version=1, definition=DEFINITION, parent=str(Path(args.parent).resolve()),
                    parent_sha256=digest(args.parent), grid=str(Path(args.grid).resolve()),
                    grid_sha256=digest(args.grid), n_cells=n_cells, splits={},
                    recipe=vars(recipe), files={},
                    catalogue_stats={p: dict(size=os.stat(p).st_size, mtime_ns=os.stat(p).st_mtime_ns)
                                     for p in (args.g0, args.g1)})
    for name, subset in (("train", train), ("validation", validation)):
        actual_cases = np.sort(subset["case"].unique())
        if not np.array_equal(actual_cases, np.sort(expected_cases[name])):
            raise ValueError("case split differs from frozen grid")
        pairs, report = matched_transport_indices(subset)
        shear = subset["_paired_shear"].to_numpy(float)
        if not np.allclose(shear[pairs[:, 0]], 0, atol=1e-10):
            raise ValueError("nonzero baseline shear")
        keep = np.isclose(shear[pairs[:, 1]], .05, atol=1e-7)
        report["dropped_zero_destination_shear"] = int((~keep).sum())
        pairs = pairs[keep]
        grid_values = subset[["r_input_p", "Re_input_p", "r_blend"]].to_numpy(float)
        finite_grid = np.isfinite(grid_values[pairs[:, 0]]).all(axis=1)
        report["dropped_nonfinite_grid_pairs"] = int((~finite_grid).sum())
        pairs = pairs[finite_grid]
        if not len(pairs):
            raise ValueError("no usable matched grid pairs")
        report["fixed_conditions"] = verify_ratio_base_properties(subset, pairs, condition.feature_names)
        baseline = subset.iloc[pairs[:, 0]]
        bin_ids = coupling_bin_ids(baseline, *edges)
        order = np.argsort(bin_ids, kind="stable")
        pairs, bin_ids = pairs[order], bin_ids[order]
        measured = subset[target.target_names[:2]].to_numpy(float)
        gamma = subset[["_grid_gamma1", "_grid_gamma2"]].to_numpy(float)[pairs[:, 1]]
        cases = subset["case"].to_numpy(int)[pairs[:, 0]]
        stats = fit_grid_response(measured[pairs[:, 1]]-measured[pairs[:, 0]], gamma, bin_ids, cases, n_cells)
        report.update(used_pairs=len(pairs), nll_rows=len(subset),
                      min_cell_count=int(stats["counts"].min()), min_case_coverage=int(stats["case_coverage"].min()),
                      covariance_min_eigenvalue=float(stats["eigenvalues"].min()),
                      covariance_max_condition=float((stats["eigenvalues"][:, -1]/stats["eigenvalues"][:, 0]).max()),
                      cell_response_se_quantiles=np.quantile(np.sqrt(np.diagonal(stats["covariance"], axis1=1, axis2=2)), [.0,.5,.95,1.], axis=0).tolist())
        np.savez(root/f"{name}_targets.npz", **stats, edges_flux=edges[0], edges_size=edges[1], edges_crowd=edges[2])
        payload = dict(values=torch.from_numpy(target.transform_frame(subset)),
                       context=torch.from_numpy(condition.transform_frame(subset)),
                       pairs=pairs, bin_ids=bin_ids, design=stats["design"],
                       target=stats["target"], precision=stats["precision"], counts=stats["counts"],
                       pair_keys=subset.iloc[pairs[:, 0]][["case", "input_index"]].to_numpy(np.int64),
                       gamma=gamma)
        torch.save(payload, root/f"{name}.pt")
        del payload
        manifest["splits"][name] = report
        for filename in (f"{name}.pt", f"{name}_targets.npz"):
            manifest["files"][filename] = digest(root/filename)
        print(f"PREPARED {name}: {json.dumps(report)}", flush=True)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("GRID_RESPONSE_PREPARATION_COMPLETE", flush=True)


def train(args):
    if (not np.isfinite(args.weight) or args.weight <= 0 or args.draws < 2 or args.draws % 2
            or args.validation_draws < 2 or args.validation_draws % 2
            or min(args.epochs, args.patience, args.batch_size, args.cells_per_step, args.pairs_per_cell) < 1):
        raise ValueError("invalid positive training options/even antithetic draw counts")
    root, output = Path(args.root), Path(args.output)
    if output.exists() or output.with_suffix(".history.json").exists():
        raise FileExistsError("refusing to overwrite a trial")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["version"] != 1 or manifest["definition"] != DEFINITION:
        raise ValueError("unsupported prepared-data definition")
    loss_divisor = 4 * manifest["n_cells"] if args.grid_loss_reduction == "mean" else 1
    training_definition = dict(DEFINITION)
    if args.grid_loss_reduction == "mean":
        training_definition.update(
            loss="NLL + lambda/(2*4*n_cells) * sum_B (mu_A-target)^T precision_B (mu_C-target)",
            cell_weighting="mean over cells and four response components; no occupancy weighting")
    if digest(manifest["parent"]) != manifest["parent_sha256"]:
        raise ValueError("parent changed")
    for name, expected in manifest["files"].items():
        if digest(root/name) != expected:
            raise ValueError(f"prepared artifact changed: {name}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("scheduled training requires a GPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    reference = load_torch(manifest["parent"])
    config = dict(reference["model_config"])
    if config["flow_type"] != "affine":
        raise ValueError("expected plain affine parent")
    config.update(response_difference="forward", response_components="matrix", response_error="covariance")
    model = build_flow(config).to(device)
    model.load_state_dict(reference["state_dict"], strict=True)
    condition = TabularPreprocessor.from_state(reference["condition_preprocessor"])
    target = TargetStandardizer.from_state(reference["target_transform"])
    populations, loaders = {}, {}
    for name in ("train", "validation"):
        payload = load_torch(root/f"{name}.pt")
        loaders[name] = GPUBatches([payload["values"], payload["context"]], args.batch_size, name == "train", device)
        populations[name] = GridResponsePopulation(loaders[name].tensors[1], payload, target.scales[:2])
        del payload
    del reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    history = []
    best, best_epoch, wait = float("inf"), 0, 0
    started = time.time()
    metadata = dict(grid_response_definition=training_definition, grid_response_weight=args.weight,
                    grid_loss_divisor=loss_divisor, grid_loss_reduction=args.grid_loss_reduction,
                    parent=manifest["parent"], parent_sha256=manifest["parent_sha256"],
                    prepared_manifest_sha256=digest(root/"manifest.json"), arguments=vars(args),
                    response_estimator="complete_flow", response_weight=0., coupling_weight=0.,
                    response_difference="forward", response_delta=.05,
                    response_draws=args.draws, response_components="matrix",
                    ratio_response_weight=0., paired_transport_weight=0.,
                    distribution_response_weight=0., response_dispersion_weight=0.,
                    feature_set="g0_meas_crowd_conc_szfl_noz", target_features=target.target_names,
                    condition_features=condition.feature_names, seed=args.seed,
                    selection_name="sextractor_detected", fold_catalogue_shear=True,
                    population=manifest["splits"], primary_re_min=.37, primary_mag_max=25.8)

    def save(path, extra):
        temporary = path.with_name(path.name+".tmp")
        save_measurement_model(str(temporary), model, condition, target, config,
                               metadata={**metadata, **extra})
        os.replace(temporary, path)

    for epoch in range(args.epochs+1):
        train_nll, train_score, n_rows = 0., 0., 0
        if epoch:
            model.train()
            for batch in loaders["train"]:
                values, context = batch[:2]
                optimizer.zero_grad(set_to_none=True)
                nll = -model.log_prob(values, context).mean()
                score = populations["train"].sampled_score(
                    model, args.cells_per_step, args.pairs_per_cell, args.draws) / loss_divisor
                loss = nll + args.weight*score
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite NLL+grid objective")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
                train_nll += float(nll.detach())*len(values)
                train_score += float(score.detach())*len(values)
                n_rows += len(values)
            train_nll, train_score = train_nll/n_rows, train_score/n_rows
        model.eval()
        val_nll = epoch_nll(model, loaders["validation"], device)
        result = populations["validation"].evaluate(model, args.validation_draws,
                                                    args.validation_batch_size, args.validation_seed)
        result.update(loss_sum=result["loss"], loss_divisor=loss_divisor,
                      loss=result["loss"] / loss_divisor)
        objective = val_nll+args.weight*result["loss"]
        if not np.isfinite(objective):
            raise ValueError("nonfinite validation selector")
        record = dict(epoch=epoch, train_nll=train_nll if epoch else None,
                      train_grid=train_score if epoch else None, val_nll=val_nll,
                      val_grid=result["loss"], val_total=objective, elapsed_seconds=time.time()-started)
        history.append(record)
        print(json.dumps(record), flush=True)
        np.savez(output.with_name(f"{output.stem}_epoch{epoch:03d}_validation.npz"), **result)
        # Include epoch zero: fine-tuning cannot silently replace a better parent.
        if objective < best:
            best, best_epoch, wait = objective, epoch, 0
            save(output, dict(epoch=epoch, best_epoch=best_epoch, best_val_objective=best,
                              checkpoint="best_validation", history=history))
        else:
            wait += 1
        state = dict(history=history, best_epoch=best_epoch, best_val_objective=best,
                     no_improvement_epochs=wait, stopped_on_patience=wait >= args.patience,
                     capped=epoch == args.epochs and wait < args.patience)
        output.with_suffix(".history.json").write_text(json.dumps(state, indent=2)+"\n")
        if epoch:
            # Retain a recoverable optimizer/current-model snapshot if walltime expires.
            temporary = output.with_suffix(".resume.tmp")
            torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch, history=history, best=best, best_epoch=best_epoch,
                            wait=wait, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
            os.replace(temporary, output.with_suffix(".resume.pt"))
        if wait >= args.patience:
            break
    print(f"GRID_RESPONSE_TRAINING_COMPLETE best_epoch={best_epoch} patience_reached={wait >= args.patience}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for flag in ("root", "parent", "grid", "g0", "g1"):
        p.add_argument("--"+flag, required=True)
    p = sub.add_parser("train")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--grid-loss-reduction", choices=["sum", "mean"], default="sum",
                   help="mean divides the raw half-quadratic sum by 4*n_cells in training and validation")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--cells-per-step", type=int, default=4)
    p.add_argument("--pairs-per-cell", type=int, default=128)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--validation-draws", type=int, default=16)
    p.add_argument("--validation-batch-size", type=int, default=1024)
    p.add_argument("--validation-seed", type=int, default=7301)
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=5.)
    args = parser.parse_args(argv)
    prepare(args) if args.command == "prepare" else train(args)


if __name__ == "__main__":
    main()
