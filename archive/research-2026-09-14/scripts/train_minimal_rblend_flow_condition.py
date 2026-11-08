#!/usr/bin/env python3
"""Minimal NLL-only test of sharing R_blend with the measurement flow.

The control removes the parent flow's eight identically-zero missing flags.
The trial adds one standardized, externally predicted R_blend scalar.  Both
models are exact function-preserving maps of the same frozen checkpoint at
epoch zero, use the same rows and minibatch order, and select checkpoints only
by held-out NLL.  This worker never reads a response statistic or m.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch

from sbsi.flow_physical_response import physical_outputs
from sbsi.measurement_model import (
    TargetStandardizer,
    build_flow,
    save_measurement_model,
)
from sbsi.selection_model import TabularPreprocessor


FLOW_FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
)
TRIAL9_RAW_FEATURES = (
    "index_input", "RA_input", "DEC_input", "r_input", "Re_input", "sersic_n_input"
)
ARMS = ("flow8", "flow8_rblend")
TRAIN_CASES = tuple(range(40, 100))
VALIDATION_CASES = tuple(range(100, 120))


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _input_stats(paths) -> dict:
    return {
        str(Path(path).resolve()): [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        for path in paths
    }


def _lookup_paths() -> list[Path]:
    old = Path(
        "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
        "plain_complete_flow_response_re037_v1/"
        "constgold50_trial9_base_lambda05_epoch060_v1"
    )
    replicated = Path(
        "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
        "detection_circularized_repair_v1/selection_replication_c90_139_v1/prepared"
    )
    return [old / f"lookup_{i}.feather" for i in range(5)] + [
        replicated / f"case{case:03d}/rblend.feather" for case in range(90, 120)
    ]


def init_protocol(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"preserving existing protocol: {protocol_path}")

    parent = Path(args.parent).resolve()
    data_root = Path(args.data_root).resolve()
    context_root = Path(args.context_root).resolve()
    source_root = Path(args.source_root).resolve()
    script = Path(__file__).resolve()
    job_script = Path(args.job_script).resolve()
    lookup_paths = _lookup_paths()
    small_inputs = [
        parent,
        script,
        job_script,
        source_root / "sbsi/measurement_model.py",
        source_root / "sbsi/selection_model.py",
        source_root / "sbsi/flow_disk.py",
        source_root / "sbsi/flow_physical_disk.py",
        source_root / "sbsi/flow_physical_response.py",
        *lookup_paths,
    ]
    large_inputs = [
        data_root / f"{split}_{name}.npy"
        for split in ("train", "validation")
        for name in ("raw", "row_keys")
    ] + [context_root / f"{split}_context.npy" for split in ("train", "validation")]
    for path in [*small_inputs, *large_inputs]:
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = {
        "version": "minimal-rblend-flow-condition-v1",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "NLL-only development gate for conditioning the frozen measurement flow "
            "on the same external R_blend scalar already used by the detector."
        ),
        "arms": {
            "flow8": list(FLOW_FEATURES),
            "flow8_rblend": [*FLOW_FEATURES, "R_blend"],
        },
        "splits": {
            "train": list(TRAIN_CASES),
            "validation": list(VALIDATION_CASES),
            "response": list(range(120, 140)),
        },
        "recipe": {
            "seed": 501,
            "learning_rate": 1.0e-5,
            "epochs": 20,
            "rows_per_epoch": 2_000_000,
            "batch_size": 8192,
            "gradient_clip": 5.0,
            "validation_rows": 500_000,
            "validation_seed": 900_000_504,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260913,
        },
        "gate": {
            "contrast": "flow8_rblend minus flow8 full held-out case-mean NLL",
            "continue_rule": (
                "the pooled contrast is negative and the paired case-bootstrap "
                "95% interval has an upper endpoint below zero"
            ),
            "unit": "simulation case",
        },
        "initialization": {
            "parent": str(parent),
            "mapping": (
                "copy every parameter; in each coupling first layer copy target+eight "
                "real-context columns, discard eight dead missing-flag columns, and "
                "zero-initialize the optional final R_blend column"
            ),
            "optimizer": "fresh AdamW for both arms; no parent optimizer at selected epoch154 exists",
            "historical_limitation": (
                "the common epoch154 initialization was selected with a paired shape+radius "
                "response objective; this new continuation and selection are NLL-only"
            ),
        },
        "scope": {
            "new_image_simulation": False,
            "catalogues": "existing g=0 and g=0.05 measured catalogues only",
            "rblend": (
                "reuse exact trial9 tables from existing coherent catalogues after verifying "
                "all seven trial9 source features are unchanged from the existing g=0 truth"
            ),
            "forbidden": [
                "response statistics or m in training, checkpointing, or the proper-score gate",
                "cases 120-139 before a checkpoint is frozen",
                "new input features beyond R_blend",
                "response offsets, rescaling, temperatures, or response-selected stopping",
            ],
        },
        "paths": {
            "root": str(root),
            "parent": str(parent),
            "data_root": str(data_root),
            "context_root": str(context_root),
            "source_root": str(source_root),
            "script": str(script),
            "job_script": str(job_script),
            "g0_input_pattern": args.g0_input_pattern,
            "coherent_input_pattern": args.coherent_input_pattern,
            "lookups": [str(path.resolve()) for path in lookup_paths],
        },
        "hashes": {str(path.resolve()): sha256(path) for path in small_inputs},
        "large_input_stats": _input_stats(large_inputs),
        "interpretation": (
            "Development proper-score evidence only. A pass permits one cached response "
            "screen; it is neither a 0.3% claim nor model promotion."
        ),
    }
    write_json(protocol_path, protocol)
    print(f"MINIMAL_RBLEND_FLOW_PROTOCOL_READY root={root}", flush=True)


def load_protocol(root: str | Path, *, verify_prepared: bool = False) -> dict:
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["version"] != "minimal-rblend-flow-condition-v1":
        raise RuntimeError("wrong protocol version")
    if Path(protocol["paths"]["root"]) != root:
        raise RuntimeError("protocol/root mismatch")
    if protocol["splits"]["response"] != list(range(120, 140)):
        raise RuntimeError("response firewall changed")
    for path, expected in protocol["hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    for path, expected in protocol["large_input_stats"].items():
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise RuntimeError(f"large input changed: {path}")
    if verify_prepared:
        receipt = json.loads((root / "prepared/receipt.json").read_text())
        for name, expected in receipt["hashes"].items():
            if sha256(root / "prepared" / name) != expected:
                raise RuntimeError(f"prepared input changed: {name}")
    return protocol


def load_rblend(protocol: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for text in protocol["paths"]["lookups"]:
        frame = pd.read_feather(text, columns=["case", "input_index", "R_blend"])
        for case, group in frame.groupby("case", sort=False):
            case = int(case)
            if case not in (*TRAIN_CASES, *VALIDATION_CASES):
                continue
            if case in result:
                raise RuntimeError(f"duplicate R_blend case {case}")
            group = group.sort_values("input_index")
            ids = group["input_index"].to_numpy(np.int64)
            values = group["R_blend"].to_numpy(np.float64)
            if len(np.unique(ids)) != len(ids) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid R_blend lookup for case {case}")
            result[case] = ids, values
    expected = set((*TRAIN_CASES, *VALIDATION_CASES))
    if set(result) != expected:
        raise RuntimeError(f"R_blend case coverage differs: {sorted(expected-set(result))}")
    return result


def _assert_trial9_source_invariance(protocol: dict, lookups: dict) -> dict:
    report = {}
    for case in (*TRAIN_CASES, *VALIDATION_CASES):
        paths = [
            Path(protocol["paths"][name].format(case=case))
            for name in ("g0_input_pattern", "coherent_input_pattern")
        ]
        tables = [pf.read_table(path, columns=list(TRIAL9_RAW_FEATURES)) for path in paths]
        if tables[0].num_rows != tables[1].num_rows:
            raise RuntimeError(f"truth row count changed for case {case}")
        for name in TRIAL9_RAW_FEATURES:
            a = tables[0][name].combine_chunks().to_numpy(zero_copy_only=False)
            b = tables[1][name].combine_chunks().to_numpy(zero_copy_only=False)
            np.testing.assert_array_equal(a, b, err_msg=f"case{case} {name}")
        raw = tables[0].to_pandas().rename(
            columns={name: name.removesuffix("_input") for name in TRIAL9_RAW_FEATURES}
        )
        eligible = raw.loc[
            np.isfinite(raw[["r", "Re"]]).all(axis=1)
            & (raw.r > 18.0)
            & (raw.r < 25.8)
            & (raw.Re > 0.37)
            & (raw.Re < 1.5),
            "index",
        ].to_numpy(np.int64)
        np.testing.assert_array_equal(np.sort(eligible), lookups[case][0])
        report[str(case)] = {
            "rows": int(len(raw)),
            "eligible_lookup_rows": int(len(eligible)),
            "raw_trial9_features_exact": True,
        }
    return report


def _join_rblend(keys: np.ndarray, lookups: dict) -> np.ndarray:
    output = np.empty(len(keys), dtype=np.float32)
    for case in np.unique(keys[:, 0]):
        rows = np.flatnonzero(keys[:, 0] == case)
        ids, values = lookups[int(case)]
        position = np.searchsorted(ids, keys[rows, 1])
        valid = position < len(ids)
        if valid.any():
            valid[valid] &= ids[position[valid]] == keys[rows[valid], 1]
        if not valid.all():
            raise RuntimeError(f"{int((~valid).sum())} unmatched R_blend rows in case {case}")
        output[rows] = values[position].astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite joined R_blend")
    return output


def prepare_data(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    protocol = load_protocol(root)
    output = root / "prepared"
    output.mkdir(exist_ok=False)
    lookups = load_rblend(protocol)
    source_report = _assert_trial9_source_invariance(protocol, lookups)
    parent = torch.load(protocol["paths"]["parent"], map_location="cpu", weights_only=False)
    target = TargetStandardizer.from_state(parent["target_transform"])
    data_root = Path(protocol["paths"]["data_root"])
    context_root = Path(protocol["paths"]["context_root"])
    pieces = {name: {"values": [], "context8": [], "rblend": [], "cases": []}
              for name in ("train", "validation")}
    dead_nonzero = 0
    source_counts = {}
    for source_split in ("train", "validation"):
        contexts = np.load(context_root / f"{source_split}_context.npy", mmap_mode="r")
        raw = np.load(data_root / f"{source_split}_raw.npy", mmap_mode="r")
        keys = np.load(data_root / f"{source_split}_row_keys.npy", mmap_mode="r")
        if len(contexts) != len(raw) or len(raw) != len(keys) or contexts.shape[1] != 16:
            raise RuntimeError(f"source alignment changed for {source_split}")
        dead_nonzero += int(np.count_nonzero(contexts[:, 8:]))
        for destination, cases in (("train", TRAIN_CASES), ("validation", VALIDATION_CASES)):
            selected = np.isin(keys[:, 0], cases)
            selected_keys = np.asarray(keys[selected])
            if len(selected_keys) == 0:
                continue
            if not np.isin(selected_keys[:, 2], (0, 1)).all():
                raise RuntimeError("unexpected catalogue leg")
            compact = np.asarray(contexts[selected, :8], dtype=np.float32)
            if not np.isfinite(compact).all():
                raise RuntimeError("nonfinite compact context")
            values = target.transform_array(physical_outputs(np.asarray(raw[selected])))
            pieces[destination]["values"].append(values)
            pieces[destination]["context8"].append(compact)
            pieces[destination]["rblend"].append(_join_rblend(selected_keys, lookups))
            pieces[destination]["cases"].append(selected_keys[:, 0].astype(np.int16))
            source_counts[f"{source_split}_to_{destination}"] = int(len(selected_keys))
    if dead_nonzero:
        raise RuntimeError(f"missing flags are not dead: {dead_nonzero} nonzero entries")

    arrays = {}
    split_report = {}
    for split in ("train", "validation"):
        expected_cases = TRAIN_CASES if split == "train" else VALIDATION_CASES
        for name, parts in pieces[split].items():
            value = np.concatenate(parts)
            arrays[f"{split}_{name}.npy"] = value
            np.save(output / f"{split}_{name}.npy", value)
        cases = arrays[f"{split}_cases.npy"]
        if set(np.unique(cases)) != set(expected_cases):
            raise RuntimeError(f"wrong {split} cases")
        n = len(cases)
        if any(len(arrays[f"{split}_{name}.npy"]) != n for name in ("values", "context8", "rblend")):
            raise RuntimeError(f"unaligned {split} arrays")
        split_report[split] = {
            "rows": int(n),
            "cases": [int(v) for v in np.unique(cases)],
            "case_counts": {
                str(case): int(np.count_nonzero(cases == case)) for case in np.unique(cases)
            },
            "rblend_mean": float(arrays[f"{split}_rblend.npy"].mean(dtype=np.float64)),
            "rblend_std": float(arrays[f"{split}_rblend.npy"].std(dtype=np.float64, ddof=1)),
        }
    train_rb = arrays["train_rblend.npy"].astype(np.float64)
    rb_mean, rb_scale = float(train_rb.mean()), float(train_rb.std(ddof=1))
    if not np.isfinite(rb_scale) or rb_scale < 1e-6:
        raise RuntimeError("degenerate training R_blend")
    normalization = np.array([rb_mean, rb_scale], dtype=np.float64)
    np.save(output / "rblend_normalization.npy", normalization)
    rng = np.random.default_rng(protocol["recipe"]["validation_seed"])
    count = min(protocol["recipe"]["validation_rows"], len(arrays["validation_values.npy"]))
    validation_indices = np.sort(rng.choice(len(arrays["validation_values.npy"]), count, replace=False))
    np.save(output / "validation_indices.npy", validation_indices)
    files = [*arrays, "rblend_normalization.npy", "validation_indices.npy"]
    receipt = {
        "source_trial9_invariance": source_report,
        "source_counts": source_counts,
        "dead_missing_indicator_nonzero": dead_nonzero,
        "splits": split_report,
        "rblend_normalization": {"mean": rb_mean, "scale": rb_scale, "fit_on": "training NLL rows"},
        "validation_subset_rows": int(count),
        "hashes": {name: sha256(output / name) for name in files},
    }
    write_json(output / "receipt.json", receipt)
    load_protocol(root, verify_prepared=True)
    print("MINIMAL_RBLEND_FLOW_DATA_READY", json.dumps(split_report), flush=True)


def compact_preprocessor(parent: dict, arm: str, normalization: np.ndarray) -> TabularPreprocessor:
    state = parent["condition_preprocessor"]
    if tuple(state["feature_names"]) != FLOW_FEATURES or tuple(state.get("log_features", ())) != ():
        raise RuntimeError("parent feature contract changed")
    if not state.get("add_missing_indicators", False):
        raise RuntimeError("parent is not the expected 16-context checkpoint")
    names = list(FLOW_FEATURES)
    fill = np.asarray(state["fill_values"], dtype=np.float32)
    means = np.asarray(state["means"], dtype=np.float32)
    scales = np.asarray(state["scales"], dtype=np.float32)
    if arm == "flow8_rblend":
        mean, scale = normalization.astype(np.float32)
        names.append("R_blend")
        fill = np.append(fill, mean).astype(np.float32)
        means = np.append(means, mean).astype(np.float32)
        scales = np.append(scales, scale).astype(np.float32)
    elif arm != "flow8":
        raise ValueError(arm)
    return TabularPreprocessor(names, fill, means, scales,
                               add_missing_indicators=False, log_features=())


def map_parent_model(parent: dict, new_context_dim: int):
    old_config = dict(parent["model_config"])
    if old_config.get("target_dim") != 4 or old_config.get("context_dim") != 16:
        raise RuntimeError("wrong parent architecture")
    config = dict(old_config, context_dim=new_context_dim)
    model = build_flow(config)
    source = parent["state_dict"]
    mapped = model.state_dict()
    first_layers = 0
    for name, target_value in mapped.items():
        source_value = source[name]
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
        elif name.endswith(".net.0.weight"):
            if source_value.shape[1] != 20 or target_value.shape[1] != 4 + new_context_dim:
                raise RuntimeError(f"unexpected first-layer shape: {name}")
            target_value.zero_()
            target_value[:, :12].copy_(source_value[:, :12])
            first_layers += 1
        else:
            raise RuntimeError(f"cannot map {name}: {source_value.shape} -> {target_value.shape}")
    if first_layers != old_config["n_flows"]:
        raise RuntimeError("not every coupling first layer was mapped")
    model.load_state_dict(mapped)
    return model, config, first_layers


def _nll(model, values, context, batch_size: int, *, per_row: bool = False):
    model.eval()
    chunks = []
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            loss = -model.log_prob(values[start:start + batch_size], context[start:start + batch_size])
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("nonfinite NLL")
            if per_row:
                chunks.append(loss.detach().cpu().numpy())
            else:
                total += float(loss.sum())
    return np.concatenate(chunks) if per_row else total / len(values)


def train_arm(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA")
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify_prepared=True)
    arm = ARMS[args.index]
    output = root / "training" / arm
    output.mkdir(parents=True, exist_ok=False)
    prepared = root / "prepared"
    parent = torch.load(protocol["paths"]["parent"], map_location="cpu", weights_only=False)
    normalization = np.load(prepared / "rblend_normalization.npy")
    preprocessor = compact_preprocessor(parent, arm, normalization)
    model, config, mapped_layers = map_parent_model(parent, preprocessor.output_dim)
    model = model.cuda()
    target = TargetStandardizer.from_state(parent["target_transform"])
    train_values = torch.as_tensor(np.load(prepared / "train_values.npy"), device="cuda")
    validation_values = torch.as_tensor(np.load(prepared / "validation_values.npy"), device="cuda")
    train_context = torch.as_tensor(np.load(prepared / "train_context8.npy"), device="cuda")
    validation_context = torch.as_tensor(np.load(prepared / "validation_context8.npy"), device="cuda")
    if arm == "flow8_rblend":
        mean, scale = normalization
        train_rb = np.asarray(
            (np.load(prepared / "train_rblend.npy").astype(np.float32) - mean) / scale,
            dtype=np.float32,
        )
        validation_rb = np.asarray(
            (np.load(prepared / "validation_rblend.npy").astype(np.float32) - mean) / scale,
            dtype=np.float32,
        )
        train_context = torch.cat((train_context, torch.as_tensor(train_rb[:, None], device="cuda")), 1)
        validation_context = torch.cat((validation_context, torch.as_tensor(validation_rb[:, None], device="cuda")), 1)
    vi = torch.as_tensor(np.load(prepared / "validation_indices.npy"), device="cuda")
    recipe = protocol["recipe"]

    parent_model = build_flow(parent["model_config"]).cuda()
    parent_model.load_state_dict(parent["state_dict"])
    sample = min(16_384, len(validation_values))
    parent_context = torch.cat((validation_context[:sample, :8],
                                torch.zeros((sample, 8), device="cuda")), 1)
    old_nll = _nll(parent_model, validation_values[:sample], parent_context,
                   recipe["batch_size"], per_row=True)
    new_nll = _nll(model, validation_values[:sample], validation_context[:sample],
                   recipe["batch_size"], per_row=True)
    initial_max_abs = float(np.max(np.abs(old_nll - new_nll)))
    np.testing.assert_allclose(new_nll, old_nll, rtol=2e-5, atol=2e-6)
    del parent_model, parent_context

    torch.manual_seed(recipe["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=0.0)
    history = []
    best_epoch, best_nll = 0, float("inf")
    for epoch in range(recipe["epochs"] + 1):
        trained = None
        if epoch:
            model.train()
            epoch_seed = recipe["seed"] + epoch * 1_000_003
            indices = np.random.default_rng(epoch_seed).choice(
                len(train_values), min(recipe["rows_per_epoch"], len(train_values)), replace=False
            )
            total = 0.0
            gradient_max = 0.0
            for start in range(0, len(indices), recipe["batch_size"]):
                rows = torch.as_tensor(indices[start:start + recipe["batch_size"]], device="cuda")
                optimizer.zero_grad(set_to_none=True)
                loss = -model.log_prob(train_values[rows], train_context[rows]).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(rows)
                gradient_max = max(gradient_max, float(norm))
            trained = {"nll": total / len(indices), "rows": int(len(indices)),
                       "maximum_preclip_gradient_norm": gradient_max}
        validation_nll = _nll(model, validation_values[vi], validation_context[vi],
                              recipe["batch_size"])
        record = {"epoch": epoch, "train": trained, "validation_subset_nll": validation_nll}
        history.append(record)
        path = output / f"epoch{epoch:03d}.pt"
        metadata = {
            "training_phase": "minimal_rblend_flow_condition_nll_only",
            "arm": arm,
            "epoch": epoch,
            "response_used": False,
            "parent": protocol["paths"]["parent"],
            "manifest_sha256": sha256(root / "protocol.json"),
            "recipe": recipe,
            "initial_mapping": protocol["initialization"]["mapping"],
            "record": record,
        }
        save_measurement_model(path, model, preprocessor, target, config, metadata)
        if validation_nll < best_nll:
            best_epoch, best_nll = epoch, validation_nll
        write_json(output / "history.json", {"history": history, "best_epoch": best_epoch})
        print("NLL", arm, epoch, validation_nll, flush=True)

    selected_path = output / f"epoch{best_epoch:03d}.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["state_dict"])
    row_nll = _nll(model, validation_values, validation_context,
                   recipe["batch_size"], per_row=True)
    validation_cases = np.load(prepared / "validation_cases.npy")
    cases = np.unique(validation_cases)
    sums = np.array([row_nll[validation_cases == case].sum() for case in cases])
    counts = np.array([np.count_nonzero(validation_cases == case) for case in cases])
    np.savez(output / "selected_validation.npz", cases=cases, sums=sums, counts=counts,
             means=sums / counts)
    result = {
        "arm": arm,
        "best_epoch": best_epoch,
        "best_validation_subset_nll": best_nll,
        "selected_checkpoint": str(selected_path),
        "selected_sha256": sha256(selected_path),
        "full_validation_nll": float(sums.sum() / counts.sum()),
        "initial_parent_nll_max_abs_difference": initial_max_abs,
        "mapped_first_layers": mapped_layers,
        "response_used": False,
        "selection": "minimum fixed-subset validation NLL",
        "limitations": [
            "one optimization seed and a bounded fixed schedule",
            "common epoch154 initialization was historically response-informed",
            "development cases reused from the flow lineage; no calibration claim",
        ],
    }
    write_json(output / "result.json", result)
    load_protocol(root, verify_prepared=True)
    print("MINIMAL_RBLEND_FLOW_ARM_COMPLETE", json.dumps(result), flush=True)


def review(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify_prepared=True)
    results = {arm: json.loads((root / f"training/{arm}/result.json").read_text())
               for arm in ARMS}
    arrays = {arm: dict(np.load(root / f"training/{arm}/selected_validation.npz"))
              for arm in ARMS}
    np.testing.assert_array_equal(arrays["flow8"]["cases"], arrays["flow8_rblend"]["cases"])
    np.testing.assert_array_equal(arrays["flow8"]["counts"], arrays["flow8_rblend"]["counts"])
    difference = arrays["flow8_rblend"]["means"] - arrays["flow8"]["means"]
    pooled = (
        arrays["flow8_rblend"]["sums"].sum() - arrays["flow8"]["sums"].sum()
    ) / arrays["flow8"]["counts"].sum()
    recipe = protocol["recipe"]
    rng = np.random.default_rng(recipe["bootstrap_seed"])
    draws = difference[rng.integers(0, len(difference),
                                    size=(recipe["bootstrap_replicates"], len(difference)))].mean(1)
    interval = np.quantile(draws, (0.025, 0.975))
    passed = bool(pooled < 0 and interval[1] < 0)
    review_result = {
        "passed": passed,
        "gate": protocol["gate"],
        "pooled_validation_nll_difference": float(pooled),
        "mean_case_nll_difference": float(difference.mean()),
        "case_standard_error": float(difference.std(ddof=1) / np.sqrt(len(difference))),
        "paired_case_bootstrap_95": interval.tolist(),
        "cases_improved": int(np.count_nonzero(difference < 0)),
        "n_cases": int(len(difference)),
        "arms": results,
        "response_used": False,
        "next_step": (
            "freeze the nine-input checkpoint and run the cached cases120-139 response screen"
            if passed else
            "prune the R_blend-conditioned flow branch; do not run response or new simulations"
        ),
    }
    write_json(root / "development_result.json", review_result)
    print("MINIMAL_RBLEND_FLOW_REVIEW", json.dumps(review_result), flush=True)


def selftest() -> None:
    config = {
        "flow_type": "physical_disk_affine", "target_dim": 4, "context_dim": 16,
        "hidden_dim": 8, "n_layers": 1, "n_flows": 2, "scale_limit": 1.0,
        "activation": "silu", "mean_hidden": 0, "disk_map": "radial_tanh",
        "disk_shape_means": [0.0, 0.0], "disk_shape_scales": [0.3, 0.3],
        "disk_coordinate_means": [0.0, 0.0], "disk_coordinate_scales": [0.5, 0.5],
        "physical_map": "softplus", "physical_means": [3.0, 100.0],
        "physical_scales": [2.0, 80.0], "physical_units": [2.0, 80.0],
        "physical_coordinate_means": [1.0, 0.0], "physical_coordinate_scales": [1.0, 2.0],
    }
    torch.manual_seed(7)
    old = build_flow(config)
    with torch.no_grad():
        for value in old.parameters():
            value.normal_(0, 0.1)
    parent = {"model_config": config, "state_dict": old.state_dict()}
    values = torch.tensor([[0.1, -0.2, 0.5, 1.0], [-0.2, 0.1, -0.3, 0.4]], dtype=torch.float64)
    context8 = torch.randn(2, 8)
    context16 = torch.cat((context8, torch.zeros(2, 8)), 1)
    expected = old.log_prob(values, context16)
    for dim in (8, 9):
        model, _, layers = map_parent_model(parent, dim)
        context = context8 if dim == 8 else torch.cat((context8, torch.randn(2, 1)), 1)
        torch.testing.assert_close(model.log_prob(values, context), expected, rtol=0, atol=0)
        assert layers == 2
    print("MINIMAL_RBLEND_FLOW_SELFTEST_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    initial = sub.add_parser("init")
    initial.add_argument("--root", required=True)
    initial.add_argument("--parent", required=True)
    initial.add_argument("--data-root", required=True)
    initial.add_argument("--context-root", required=True)
    initial.add_argument("--source-root", required=True)
    initial.add_argument("--job-script", required=True)
    initial.add_argument("--g0-input-pattern", required=True)
    initial.add_argument("--coherent-input-pattern", required=True)
    for phase in ("prepare", "review"):
        command = sub.add_parser(phase)
        command.add_argument("--root", required=True)
    train = sub.add_parser("train")
    train.add_argument("--root", required=True)
    train.add_argument("--index", type=int, choices=range(len(ARMS)), required=True)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.phase != "init" and args.phase != "selftest" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("preparation, training, and review require Slurm")
    {"init": init_protocol, "prepare": prepare_data, "train": train_arm,
     "review": review, "selftest": lambda _: selftest()}[args.phase](args)


if __name__ == "__main__":
    main()
