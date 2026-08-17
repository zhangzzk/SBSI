"""Refuse to score a response-weighted emulator if its fixed recipe drifted."""
from __future__ import annotations

import argparse
import json
import os


MODELS = "/home/z/Zekang.Zhang/blendemu/models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--trees", type=int, required=True)
    ap.add_argument("--minimum-training-case", type=int, required=True)
    args = ap.parse_args()

    path = os.path.join(MODELS, f"emulator_metadata_{args.tag}.json")
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    metrics = metadata["tasks"]["regression"]["metrics"]
    recipe = metrics["response_power_weight"]
    expected = {
        "source_model": args.source,
        "alpha": args.alpha,
        "cap": args.cap,
        "mode": args.mode,
        "fixed_tree_count": args.trees,
        "minimum_training_case": args.minimum_training_case,
        "label_used_in_weight": False,
    }
    found = {key: recipe.get(key) for key in expected}
    if found != expected:
        raise SystemExit(
            "REFUSING: response-weighted recipe drifted\n"
            f"  expected={expected}\n  found={found}"
        )
    model = os.path.join(MODELS, metadata["tasks"]["regression"]["model_file"])
    if not os.path.isfile(model) or os.path.getsize(model) == 0:
        raise SystemExit(f"REFUSING: missing/empty regression model {model}")
    print(f"response-weighted provenance OK: {args.tag}")
    print(f"  {found}")


if __name__ == "__main__":
    main()
