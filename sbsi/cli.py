"""Command-line interface for SBSI training and model-path inspection."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from .flow import FlowTrainingConfig, train_flow, tune_flow
from .models import MODEL_PRESETS, get_model


_FLOW_PATH_FIELDS = {"catalogue", "output", "response_target", "coupling_target"}


def _model_payload(model):
    return {
        "name": model.name,
        "flow_checkpoints": [str(path) for path in model.flow_checkpoints],
        "emulator_model": None if model.emulator_model is None else str(model.emulator_model),
        "emulator_metadata": (
            None if model.emulator_metadata is None else str(model.emulator_metadata)
        ),
        "emulator_sha256": model.emulator_sha256,
    }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is part of the supported sims1 env
        raise RuntimeError("the SBSI CLI requires PyYAML to read --config") from exc

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"CLI config must contain a YAML mapping: {path}")
    return payload


def _flow_config(payload: Mapping[str, Any]) -> FlowTrainingConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("the 'training' section must be a mapping")
    allowed = {field.name for field in fields(FlowTrainingConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown flow training options: {unknown}")

    values = dict(payload)
    for name in _FLOW_PATH_FIELDS & values.keys():
        values[name] = Path(values[name]).expanduser()
    if "extra_arguments" in values:
        extra = values["extra_arguments"]
        if not isinstance(extra, (list, tuple)) or not all(isinstance(item, str) for item in extra):
            raise TypeError("training.extra_arguments must be a sequence of CLI strings")
        values["extra_arguments"] = tuple(extra)
    try:
        return FlowTrainingConfig(**values)
    except TypeError as exc:
        raise TypeError(f"invalid 'training' section: {exc}") from exc


def _candidate_overrides(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise TypeError("tuning.candidates must be a non-empty list of mappings")
    candidates = []
    allowed = {field.name for field in fields(FlowTrainingConfig)}
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise TypeError(f"tuning candidate {index} must be a mapping")
        if "output" not in item:
            raise KeyError(f"tuning candidate {index} must set its own output path")
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ValueError(f"unknown options in tuning candidate {index}: {unknown}")
        candidate = dict(item)
        for name in _FLOW_PATH_FIELDS & candidate.keys():
            candidate[name] = Path(candidate[name]).expanduser()
        if "extra_arguments" in candidate:
            extra = candidate["extra_arguments"]
            if not isinstance(extra, (list, tuple)) or not all(
                isinstance(argument, str) for argument in extra
            ):
                raise TypeError(
                    f"tuning candidate {index} extra_arguments must be a sequence of CLI strings"
                )
            candidate["extra_arguments"] = tuple(extra)
        candidates.append(candidate)
    return candidates


def _load_callable(reference: str) -> Callable:
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("tuning.scorer must have the form 'package.module:function'")
    module_name, attribute = reference.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        scorer = getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"tuning scorer does not exist: {reference}") from exc
    if not callable(scorer):
        raise TypeError(f"tuning scorer is not callable: {reference}")
    return scorer


def _json_ready_config(config: FlowTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for name in _FLOW_PATH_FIELDS:
        payload[name] = str(payload[name])
    payload["extra_arguments"] = list(payload["extra_arguments"])
    return payload


def _run_flow(args) -> int:
    payload = _load_yaml(args.config)
    if "training" not in payload:
        raise KeyError("CLI config is missing the required 'training' section")
    base = _flow_config(payload["training"])

    if args.mode == "train":
        checkpoint = train_flow(base)
        print(checkpoint)
        return 0

    tuning = payload.get("tuning")
    if not isinstance(tuning, Mapping):
        raise TypeError("--mode tune requires a 'tuning' mapping in the CLI config")
    required = {"validation_catalogue", "scorer", "results", "candidates"}
    missing = sorted(required - set(tuning))
    if missing:
        raise KeyError(f"tuning section is missing required options: {missing}")
    unknown = sorted(set(tuning) - required)
    if unknown:
        raise ValueError(f"unknown tuning options: {unknown}")

    validation_catalogue = Path(tuning["validation_catalogue"]).expanduser()
    if not validation_catalogue.is_file():
        raise FileNotFoundError(validation_catalogue)
    results_path = Path(tuning["results"]).expanduser()
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite tuning results: {results_path}")

    trials = tune_flow(
        base,
        _candidate_overrides(tuning["candidates"]),
        validation_catalogue=validation_catalogue,
        scorer=_load_callable(tuning["scorer"]),
    )
    result = {
        "validation_catalogue": str(validation_catalogue),
        "scorer": tuning["scorer"],
        "best_checkpoint": str(trials[0].config.swa_output),
        "trials": [
            {
                "rank": rank,
                "score": trial.score,
                "checkpoint": str(trial.config.swa_output),
                "training": _json_ready_config(trial.config),
            }
            for rank, trial in enumerate(trials, start=1)
        ],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(results_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sbsi")
    sub = parser.add_subparsers(dest="command", required=True)

    flow = sub.add_parser("flow", help="train or tune the SBSI measurement flow")
    flow.add_argument("--config", type=Path, required=True, help="user-owned YAML configuration")
    flow.add_argument("--mode", choices=("train", "tune"), required=True)
    flow.set_defaults(handler=_run_flow)

    show = sub.add_parser("show-model", help="show one optional named model-path preset")
    show.add_argument("name")
    show.set_defaults(handler=None)

    validate = sub.add_parser("validate-model", help="validate named model paths")
    validate.add_argument("names", nargs="*", default=list(MODEL_PRESETS))
    validate.set_defaults(handler=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show-model":
        print(json.dumps(_model_payload(get_model(args.name)), indent=2))
        return 0
    if args.command == "validate-model":
        for name in args.names:
            model = get_model(name)
            model.validate()
            print(f"{model.name}: model paths and emulator hash OK")
        return 0
    return args.handler(args)


__all__ = ["build_parser", "main"]
