"""Small command-line surface for inspecting SBSI model identities."""

from __future__ import annotations

import argparse
import json

from .models import MODEL_PRESETS, get_model


def _model_payload(model):
    return {
        "name": model.name,
        "flow_checkpoints": [str(path) for path in model.flow_checkpoints],
        "flow_sha256s": list(model.flow_sha256s),
        "emulator_model": (
            None if model.emulator_model is None else str(model.emulator_model)
        ),
        "emulator_metadata": (
            None if model.emulator_metadata is None else str(model.emulator_metadata)
        ),
        "emulator_sha256": model.emulator_sha256,
        "emulator_metadata_sha256": model.emulator_metadata_sha256,
        "detection_classifiers": [
            str(path) for path in model.detection_classifiers
        ],
        "detection_classifier_sha256s": list(
            model.detection_classifier_sha256s
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sbsi")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-model", help="show a named model-path preset")
    show.add_argument("name")

    validate = sub.add_parser("validate-model", help="validate model paths and hashes")
    validate.add_argument("names", nargs="*", default=list(MODEL_PRESETS))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show-model":
        print(json.dumps(_model_payload(get_model(args.name)), indent=2))
        return 0
    for name in args.names:
        model = get_model(name)
        model.validate()
        print(f"{model.name}: model artifact paths and pinned hashes OK")
    return 0


__all__ = ["build_parser", "main"]
