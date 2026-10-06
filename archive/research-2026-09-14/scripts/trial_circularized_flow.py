"""Controlled NLL-only semi-major/circularized conditioning experiment.

Run preparation and training under Slurm against a source-frozen manifest.
Both arms start from identical fresh weights. The old physical checkpoint
supplies only architecture and train-fitted output coordinates, never weights.
Responses are evaluated only after selection by held-out NLL.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np
import torch

import sbsi
from sbsi.flow_physical_case_validation import (
    fixed_cell_average_statistics, paired_physical_case_statistics,
    physical_case_sufficient_statistics,
)
from sbsi.flow_physical_population import physical_context_means
from sbsi.flow_physical_response import (
    baseline_frame_gamma, marginal_physical_response_statistics, physical_outputs,
)
from sbsi.flow_grid_accumulation import summarize_validation_means
from sbsi.flow_size_condition import (
    CIRCULARIZED_FEATURE, circularized_context, circularized_radius,
    fit_circularized_preprocessor,
)
from sbsi.flow_training import load_measurement_data
from sbsi.measurement_model import TargetStandardizer, build_flow, save_measurement_model
from sbsi.selection_model import TabularPreprocessor
from sbsi.training import split_data


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def simple(value):
    if isinstance(value, dict):
        return {key: simple(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [simple(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(simple(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify(root, manifest, *, data=False):
    for path, expected in {**manifest["source_hashes"], **manifest["metadata_hashes"]}.items():
        if digest(path) != expected:
            raise ValueError("frozen source/metadata changed: " + path)
    for path, expected in manifest["input_stats"].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError("input changed: " + path)
    if data:
        for path, expected in manifest["input_hashes"].items():
            if digest(path) != expected:
                raise ValueError("input checksum changed: " + path)
    if Path(sbsi.__file__).resolve().parent != (root / "source/sbsi").resolve():
        raise RuntimeError("installed frozen SBSI snapshot required")


def frozen_scalar_frame_gamma(context, pairs, condition, gamma):
    """Reproduce the existing target's baseline frame from cached float32 inputs.

    The frozen physical targets decoded the original standardized context,
    rather than using the catalogue's higher-precision baseline shape. Keep
    that exact frame for comparison; invariant size still uses raw pre-fold truth.
    """
    names = condition.feature_names
    columns = [names.index("e1_input_p"), names.index("e2_input_p")]
    baseline = np.asarray(context)[np.asarray(pairs)[:, 0]][:, columns].astype(float)
    baseline = baseline*condition.scales[columns] + condition.means[columns]
    return baseline_frame_gamma(gamma, baseline)


def verify_reconstructed_raw_targets(actual, cached, names):
    """Check raw identity, allowing at most two float64 ULPs for recomputed log.

    NumPy's log can round differently across CPU implementations. The original
    cached array remains the sole NLL/response data source; this only verifies
    reconstruction. Direct catalogue columns must still match exactly.
    """
    actual, cached = np.asarray(actual), np.asarray(cached)
    if actual.dtype != np.float64 or cached.dtype != np.float64:
        raise ValueError("float64 raw targets required")
    if actual.shape != cached.shape or actual.ndim != 2 or actual.shape[1] != len(names):
        raise ValueError("raw target shape changed")
    if not np.isfinite(actual).all() or not np.isfinite(cached).all():
        raise ValueError("nonfinite raw targets")
    report = {}
    for column, name in enumerate(names):
        a, b = actual[:, column], cached[:, column]
        maxulp = 2 if name == "measured_log_flux_radius" else 0
        if maxulp:
            np.testing.assert_array_max_ulp(a, b, maxulp=maxulp)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
        difference = np.abs(a-b)
        report[name] = dict(nonidentical=int(np.count_nonzero(difference)),
                            max_abs_error=float(difference.max(initial=0)),
                            max_ulp=float((difference/np.abs(np.spacing(b))).max(initial=0)),
                            allowed_ulp=maxulp)
    return report


def prepare(root, manifest):
    output = root / "prepared"
    output.mkdir(exist_ok=False)
    paths = manifest["paths"]
    previous = Path(paths["prepared"])
    preparation = json.loads((previous / "manifest.json").read_text())
    original = load(preparation["parent"])
    condition = TabularPreprocessor.from_state(original["condition_preprocessor"])
    target = TargetStandardizer.from_state(original["target_transform"])
    args = argparse.Namespace(**preparation["recipe"])
    # The destination catalogue omits axis_ratio_input_p. The grid loader copies
    # the original unsheared ellipticity into _ratio_base_e1/e2 BEFORE folding.
    # Recover q0 from this copy, never from the resulting folded shape features.
    frame = load_measurement_data(args, condition.feature_names, target.target_names, data_seed=501)
    train_frame, validation_frame = split_data(frame, 501, .2, group_column="case")
    del frame
    report = {"splits": {}, "files": {}, "definition": manifest["definition"]}
    circular_pre = None
    for split, subset in (("train", train_frame), ("validation", validation_frame)):
        payload = load(previous / (split + ".pt"))
        raw = np.load(Path(paths["raw"]) / (split + "_raw.npy"), mmap_mode="r")
        row_keys = np.load(Path(paths["raw"]) / (split + "_row_keys.npy"), mmap_mode="r")
        if len(subset) != len(payload["context"]) or len(subset) != len(raw):
            raise ValueError("NLL row population changed")
        keys = subset[["case", "input_index", "__catalogue_index"]].to_numpy(np.int64)
        np.testing.assert_array_equal(keys, row_keys)
        raw_verification = {}
        for start in range(0, len(subset), 100000):
            stop = min(start + 100000, len(subset))
            part = subset.iloc[start:stop]
            np.testing.assert_array_equal(condition.transform_frame(part), payload["context"][start:stop])
            np.testing.assert_array_equal(target.transform_frame(part), payload["values"][start:stop])
            checked = verify_reconstructed_raw_targets(
                part[target.target_names].to_numpy(np.float64), raw[start:stop], target.target_names)
            for name, entry in checked.items():
                if name not in raw_verification:
                    raw_verification[name] = entry
                else:
                    saved = raw_verification[name]
                    saved["nonidentical"] += entry["nonidentical"]
                    for key in ("max_abs_error", "max_ulp"):
                        saved[key] = max(saved[key], entry[key])
        print("RAW_TARGET_VERIFICATION", split, json.dumps(raw_verification), flush=True)
        base_e = subset[["_ratio_base_e1", "_ratio_base_e2"]].to_numpy(np.float64)
        e0 = np.linalg.norm(base_e, axis=1)
        if not np.isfinite(e0).all() or np.any(e0 >= 1):
            raise ValueError("invalid baseline ellipticity")
        q0 = (1-e0)/(1+e0)
        radius = circularized_radius(subset.Re_input_p.to_numpy(np.float64), q0)
        pairs = np.asarray(payload["pairs"])
        for leg in (0, 1):
            np.testing.assert_array_equal(keys[pairs[:, leg], :2], payload["pair_keys"])
            np.testing.assert_array_equal(keys[pairs[:, leg], 2], leg)
        np.testing.assert_array_equal(base_e[pairs[:, 0]], base_e[pairs[:, 1]])
        np.testing.assert_array_equal(q0[pairs[:, 0]], q0[pairs[:, 1]])
        np.testing.assert_array_equal(radius[pairs[:, 0]], radius[pairs[:, 1]])
        gamma = subset[["_grid_gamma1", "_grid_gamma2"]].to_numpy(float)[pairs[:, 1]]
        np.testing.assert_array_equal(gamma, payload["gamma"])
        if split == "train":
            circular_pre = fit_circularized_preprocessor(condition, radius)
            torch.save(circular_pre.to_state(), output / "circularized_preprocessor.pt")
            report["files"]["circularized_preprocessor.pt"] = digest(output / "circularized_preprocessor.pt")
        context = circularized_context(payload["context"].numpy(), radius, circular_pre)
        size_index = condition.feature_names.index("Re_input_p")
        unchanged = [i for i in range(context.shape[1]) if i not in
                     (size_index, size_index + len(condition.feature_names))]
        for start in range(0, len(context), 100000):
            np.testing.assert_array_equal(context[start:start+100000, unchanged],
                                          payload["context"][start:start+100000].numpy()[:, unchanged])
        np.testing.assert_array_equal(context[pairs[:, 0], size_index], context[pairs[:, 1], size_index])
        for name, array in ((split + "_context.npy", context), (split + "_circularized_radius.npy", radius)):
            np.save(output / name, array)
            report["files"][name] = digest(output / name)
        if split == "validation":
            scalar_gamma = frozen_scalar_frame_gamma(payload["context"], pairs, condition, gamma)
            case_labels = np.unique(keys[pairs[:, 0], 0])
            normal = measured = counts = None
            for start in range(0, len(pairs), 100000):
                rows = pairs[start:start+100000]
                delta = physical_outputs(raw[rows[:, 1]]) - physical_outputs(raw[rows[:, 0]])
                stats = physical_case_sufficient_statistics(
                    gamma[start:start+len(rows)], scalar_gamma[start:start+len(rows)],
                    payload["bin_ids"][start:start+len(rows)], keys[rows[:, 0], 0], delta,
                    n_cells=len(payload["counts"]), case_labels=case_labels)
                if normal is None:
                    normal, measured, counts = stats["normal"], stats["cross"], stats["counts"]
                else:
                    normal += stats["normal"]
                    measured += stats["cross"]
                    counts += stats["counts"]
            actual = paired_physical_case_statistics(normal, measured, np.zeros((4, *measured.shape)))
            frozen = dict(np.load(Path(paths["targets"]) / "validation_targets.npz"))
            np.testing.assert_allclose(actual["measured_response"], frozen["target"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(actual["measured_jackknife"], frozen["jackknife"], rtol=1e-9, atol=1e-9)
            name = "validation_case_design.npz"
            np.savez(output / name, normal=normal, measured_cross=measured, counts=counts,
                     scalar_gamma=scalar_gamma, case_labels=case_labels)
            report["files"][name] = digest(output / name)
        report["splits"][split] = dict(
            nll_rows=len(subset), response_pairs=len(pairs),
            outside_response_pair_rows_retained=len(subset)-2*len(pairs),
            original_matching_report=preparation["splits"][split],
            original_row_keys_targets_contexts_exact=True,
            reconstructed_raw_target_verification=raw_verification,
            cached_raw_targets_used_without_modification=True,
            unsheared_axis_ratio_source="(1-abs(_ratio_base_e))/(1+abs(_ratio_base_e)); saved before folding",
            scalar_response_frame_source="original cached float32 context decoded in float64, matching frozen targets",
            original_pair_keys_shear_grid_preserved=True, paired_invariant_radius_exact=True,
            circularized_radius_range=[radius.min(), radius.max()],
            original_semimajor_cuts_and_grid_unchanged=True)
        if split == "validation":
            report["splits"][split].update(
                frozen_response_max_abs_error=float(np.max(np.abs(actual["measured_response"]-frozen["target"]))),
                frozen_jackknife_max_abs_error=float(np.max(np.abs(actual["measured_jackknife"]-frozen["jackknife"]))))
        print("PREPARED", split, len(subset), len(pairs), flush=True)
        del payload, raw, row_keys, keys, base_e, q0, e0, radius, context
    verify(root, manifest)
    write_json(output / "results.json", report)
    print("CIRCULARIZED_PREPARATION_COMPLETE", flush=True)


def fit_nll_arm(output, arm, config, condition, target, train, validation, recipe, metadata):
    """Fresh, identically seeded fits; held-out NLL alone selects checkpoints."""
    output.mkdir(exist_ok=False)
    nll_pass = runpy.run_path(str(Path(__file__).with_name("train_flow_disk_nll.py")))["nll_pass"]
    seed = recipe["seed"]
    torch.manual_seed(seed)
    model = build_flow(config).to(train[0].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=4, threshold=1e-4,
        threshold_mode="abs", min_lr=1e-6)
    history = []
    best, best_nll = 0, float("inf")
    for epoch in range(recipe["epochs"] + 1):
        trained = None
        rate = optimizer.param_groups[0]["lr"]
        if epoch:
            epoch_seed = seed + epoch*1000003
            indices = np.random.default_rng(epoch_seed).choice(
                len(train[0]), min(recipe["rows_per_epoch"], len(train[0])), replace=False)
            ix = torch.as_tensor(indices, device=train[0].device)
            trained = nll_pass(model, train[0][ix], train[1][ix], recipe["batch_size"],
                               optimizer=optimizer, seed=epoch_seed, clip=5.)
        validated = nll_pass(model, *validation, recipe["batch_size"])
        record = dict(epoch=epoch, lr=rate, train=trained, validation=validated)
        if validated["nll"] < best_nll:
            best, best_nll = epoch, validated["nll"]
        history.append(record)
        save_measurement_model(output / ("epoch%03d.pt" % epoch), model, condition, target,
                               config, {**metadata, "arm": arm, "epoch": epoch, "record": record})
        if epoch:
            scheduler.step(validated["nll"])
        temporary = output / "optimizer.tmp"
        torch.save(dict(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), best_epoch=best, history=history), temporary)
        os.replace(temporary, output / "optimizer.pt")
        write_json(output / "history.json", dict(history=history, best_epoch=best))
        print("NLL", arm, epoch, validated["nll"], rate, flush=True)
        if epoch-best >= recipe["early_stopping_patience"] and optimizer.param_groups[0]["lr"] <= 1e-6:
            break
    selected_path = output / ("epoch%03d.pt" % best)
    model.load_state_dict(load(selected_path)["state_dict"])
    result = dict(best_epoch=best, epochs=epoch, best_checkpoint=str(selected_path),
                  best_sha256=digest(selected_path), best_validation_nll=best_nll,
                  initialization="fresh identical seed; no pretrained weights", response_weight=0.,
                  selection="minimum fixed-subset validation NLL", convergence_established=False)
    write_json(output / "nll_result.json", result)
    return model, result


def audit_response(model, context, payload, case_design, target_statistics, recipe):
    """Full matched response, paired case uncertainty, and independent MC groups."""
    pairs = np.asarray(payload["pairs"])
    gamma, bins = np.asarray(payload["gamma"]), np.asarray(payload["bin_ids"])
    cases = np.asarray(payload["pair_keys"])[:, 0]
    group_cross = []
    minima, maxima = np.full(4, np.inf), np.full(4, -np.inf)
    nonpositive = np.zeros(2, dtype=np.int64)
    boundary = exterior = draw_count = 0
    model.eval()
    for group in range(recipe["validation_groups"]):
        cross = np.zeros_like(case_design["measured_cross"])
        with torch.no_grad():
            for start in range(0, len(pairs), recipe["response_batch_size"]):
                end = min(start + recipe["response_batch_size"], len(pairs))
                rows = torch.as_tensor(pairs[start:end], device=context.device)
                means, samples = physical_context_means(
                    model, [context[rows[:, 0]], context[rows[:, 1]]], recipe["validation_draws"],
                    recipe["response_seed"] + group*1000000000 + start, return_samples=True)
                for sample in samples:
                    if not bool(torch.isfinite(sample).all()):
                        raise ValueError("nonfinite physical samples; no draws discarded")
                    minima = np.minimum(minima, sample.amin((0, 1)).cpu().numpy())
                    maxima = np.maximum(maxima, sample.amax((0, 1)).cpu().numpy())
                    nonpositive += (sample[..., 2:] <= 0).sum((0, 1)).cpu().numpy()
                    r2 = sample[..., :2].square().sum(-1)
                    boundary += int((r2 >= 1).sum())
                    exterior += int((r2 > 1 + 16*np.finfo(float).eps).sum())
                    draw_count += sample.shape[0]*sample.shape[1]
                stats = physical_case_sufficient_statistics(
                    gamma[start:end], case_design["scalar_gamma"][start:end], bins[start:end],
                    cases[start:end], (means[1]-means[0]).cpu().numpy(),
                    n_cells=len(payload["counts"]), case_labels=case_design["case_labels"])
                cross += stats["cross"]
        group_cross.append(cross)
        print("RESPONSE_GROUP_COMPLETE", group, flush=True)
    statistics = paired_physical_case_statistics(
        case_design["normal"], case_design["measured_cross"], np.stack(group_cross))
    np.testing.assert_allclose(statistics["measured_response"], target_statistics["target"],
                               rtol=1e-9, atol=1e-9)
    marginal = marginal_physical_response_statistics(target_statistics)
    score = summarize_validation_means(statistics["group_model_responses"][..., :6],
                                       marginal["target"], marginal["precision"])
    average = fixed_cell_average_statistics(statistics, np.full(len(payload["counts"]), 1/len(payload["counts"])))
    prediction, measured = statistics["model_response"], statistics["measured_response"]
    correlation = (float(np.corrcoef(prediction[:, 4], measured[:, 4])[0, 1])
                   if np.std(prediction[:, 4]) > 0 and np.std(measured[:, 4]) > 0 else None)
    summary = dict(equal_cell=average,
                   component_rms=np.sqrt(np.mean((prediction-measured)**2, axis=0)),
                   positive_radius_parallel_cells=int((prediction[:, 4] > 0).sum()),
                   positive_target_radius_parallel_cells=int((measured[:, 4] > 0).sum()),
                   radius_parallel_cell_correlation=correlation,
                   shape_radius_response_score={k: score[k] for k in
                       ("loss", "mean_quadratic", "integration_quadratic", "mc_jackknife_se")},
                   minima=minima, maxima=maxima, nonpositive_scalar_draws=nonpositive,
                   literal_disk_boundary=boundary, material_disk_exterior=exterior,
                   draw_count=draw_count, seed=recipe["response_seed"],
                   draws=recipe["validation_draws"], groups=recipe["validation_groups"])
    return summary, statistics, np.stack(group_cross)


def train(root, manifest):
    paths, recipe = manifest["paths"], manifest["recipe"]
    preparation = json.loads((root / "prepared/results.json").read_text())
    for name, expected in preparation["files"].items():
        if digest(root / "prepared" / name) != expected:
            raise ValueError("trial prepared data changed: " + name)
    output = root / "training"
    output.mkdir(exist_ok=False)
    parent = load(paths["physical_reference"])
    config = parent["model_config"]
    if config["flow_type"] != "physical_disk_affine":
        raise ValueError("physical disk architecture required")
    target = TargetStandardizer.from_state(parent["target_transform"])
    expected_targets = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto"]
    if target.target_names != expected_targets:
        raise ValueError("wrong physical output contract")
    preprocessors = dict(semimajor=TabularPreprocessor.from_state(parent["condition_preprocessor"]),
                         circularized=TabularPreprocessor.from_state(load(root / "prepared/circularized_preprocessor.pt")))
    raw_train = np.load(Path(paths["raw"]) / "train_raw.npy", mmap_mode="r")
    raw_val = np.load(Path(paths["raw"]) / "validation_raw.npy", mmap_mode="r")
    train_values = torch.as_tensor(target.transform_array(physical_outputs(raw_train)), device="cuda")
    val_values = torch.as_tensor(target.transform_array(physical_outputs(raw_val)), device="cuda")
    train_payload = load(Path(paths["prepared"]) / "train.pt")
    val_payload = load(Path(paths["prepared"]) / "validation.pt")
    case_design = dict(np.load(root / "prepared/validation_case_design.npz"))
    statistics = dict(np.load(Path(paths["targets"]) / "validation_targets.npz"))
    ids = np.sort(np.random.default_rng(recipe["validation_seed"]).choice(
        len(val_values), min(recipe["validation_nll_rows"], len(val_values)), replace=False))
    np.save(output / "validation_nll_indices.npy", ids)
    vi = torch.as_tensor(ids, device="cuda")
    metadata = dict(training_phase="NLL_only_size_condition_trial", grid_response_weight=0.,
                    nll_weight=1., physical_outputs=True, seed=recipe["seed"],
                    condition_definition=manifest["definition"], experimental=True,
                    density_reference_measure="same standardized e1,e2,physical radius pixels,instrumental flux",
                    physical_coordinate_reference=paths["physical_reference"],
                    pretrained_weights_used=False, constgold_used=False,
                    manifest_sha256=digest(root / "manifest.json"), recipe=recipe)
    nll_pass = runpy.run_path(str(Path(__file__).with_name("train_flow_disk_nll.py")))["nll_pass"]
    results, crosses = {}, {}
    for arm in ("semimajor", "circularized"):
        if arm == "semimajor":
            train_context = train_payload["context"].cuda()
            val_context = val_payload["context"].cuda()
        else:
            train_context = torch.as_tensor(np.load(root / "prepared/train_context.npy"), device="cuda")
            val_context = torch.as_tensor(np.load(root / "prepared/validation_context.npy"), device="cuda")
        model, selected = fit_nll_arm(output / arm, arm, config, preprocessors[arm], target,
                                      (train_values, train_context), (val_values[vi], val_context[vi]),
                                      recipe, metadata)
        # Same physical outputs and standardization make this NLL directly comparable.
        full_nll = nll_pass(model, val_values, val_context, recipe["batch_size"])
        response, detailed, crosses[arm] = audit_response(
            model, val_context, val_payload, case_design, statistics, recipe)
        np.savez(output / arm / "response.npz", **detailed, model_cross_groups=crosses[arm],
                 normal=case_design["normal"], measured_cross=case_design["measured_cross"])
        results[arm] = dict(selection=selected, full_nll=full_nll, response=response)
        write_json(output / arm / "results.json", results[arm])
        del model, train_context, val_context, detailed
        torch.cuda.empty_cache()
    # CRN between arms: retain the covariance of their differences, not a sum
    # of separate Monte Carlo variances.
    comparison = paired_physical_case_statistics(
        case_design["normal"], np.zeros_like(case_design["measured_cross"]),
        crosses["circularized"]-crosses["semimajor"])
    np.savez(output / "paired_arm_difference.npz", **comparison)
    results["circularized_minus_semimajor"] = dict(
        full_nll=results["circularized"]["full_nll"]["nll"]-results["semimajor"]["full_nll"]["nll"],
        equal_cell_response=fixed_cell_average_statistics(comparison, np.full(len(statistics["target"]), 1/len(statistics["target"]))))
    results["limitations"] = ["one initialization seed", "bounded training, no convergence claim",
                              "matched detected population", "no ConstGold or inference promotion"]
    verify(root, manifest)
    write_json(output / "results.json", results)
    print("CIRCULARIZED_FLOW_TRIAL_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "train"], required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    verify(root, manifest, data=not args.preflight)
    if args.preflight:
        print("CIRCULARIZED_PREFLIGHT_PASSED", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("catalogue preparation and training require Slurm")
    started = time.monotonic()
    (prepare if args.phase == "prepare" else train)(root, manifest)
    print("ELAPSED_SECONDS", time.monotonic()-started, flush=True)


if __name__ == "__main__":
    main()
