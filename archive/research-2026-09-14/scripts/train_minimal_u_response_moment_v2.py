#!/usr/bin/env python3
"""Stable lambda=1 repair of the frozen response-moment experiment.

Version 1 is retained unchanged as the rejected lambda=100 run.  This wrapper
reuses its gradient-checked implementation and changes one training constant,
fixed from the approximately 100-fold first-epoch response overshoot before any
response-screen case was read.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys


VERSION = "minimal-coherent-u-response-moment-lambda1-v2"
RESPONSE_WEIGHT = 1.0
IMPLEMENTATION = Path(__file__).with_name("train_minimal_u_response_moment.py")
IMPLEMENTATION_SHA256 = (
    "fd1205310ffd60433d4f3cac57f70c13bdc3abcd8988283760be5e5f5fe1ce4e"
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def implementation():
    if digest(IMPLEMENTATION) != IMPLEMENTATION_SHA256:
        raise RuntimeError("frozen lambda=100 implementation changed")
    spec = importlib.util.spec_from_file_location("minimal_u_response_v2_base", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.RESPONSE_WEIGHT = RESPONSE_WEIGHT
    # The reused implementation freezes and validates its driver via __file__.
    module.__file__ = str(Path(__file__).resolve())
    return module


def prepare(module, args) -> None:
    module.prepare(args)
    protocol_path = Path(args.output_root).resolve() / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["purpose"] = (
        "Training-only stability repair of the single-change nine-input "
        "coherent-U response-moment classifier."
    )
    protocol["training"]["objective"] = (
        "coherent-U BCE + 1*(R_truth_probability-R_truth_U)^2"
    )
    protocol["training"]["response_weight_rationale"] = (
        "Fixed from the lambda=100 first-epoch training-response overshoot: "
        "reduce its gradient scale by 100 before reading any screen case."
    )
    protocol["repair"] = {
        "rejected_root": (
            "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
            "minimal_coherent_u_response_moment_lambda100_v1"
        ),
        "evidence": (
            "seed20260913 epoch1 moved train dR from +0.00410689 to tune "
            "dR -0.41178304; subsequent epochs oscillated and degraded BCE"
        ),
        "changed": "response_weight 100 -> 1 only",
        "screen_read_before_fix": False,
    }
    protocol["paths"]["implementation_base"] = str(IMPLEMENTATION.resolve())
    protocol["hashes"]["implementation_base"] = IMPLEMENTATION_SHA256
    module.write_json(protocol_path, protocol)
    print(f"MINIMAL_U_RESPONSE_V2_PREPARED root={protocol_path.parent}", flush=True)


def main() -> None:
    module = implementation()
    args = module.parser().parse_args()
    if args.command == "prepare":
        prepare(module, args)
    elif args.command == "train":
        module.train(args)
    elif args.command == "preflight":
        module.preflight(args)
    elif args.command == "review":
        module.review(args)
    elif args.command == "screen":
        module.screen(args)
    else:
        module.self_test()


if __name__ == "__main__":
    main()
