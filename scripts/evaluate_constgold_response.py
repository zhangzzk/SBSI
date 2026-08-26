#!/usr/bin/env python
"""Evaluate direct ConstGold response closure for one measurement flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sbsi.constgold_response import evaluate_constgold_response


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--blend-lookup", nargs="+", required=True)
    parser.add_argument("--crowd-lookup", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    result = evaluate_constgold_response(
        args.model,
        args.catalogue,
        args.blend_lookup,
        args.crowd_lookup,
        min_case=args.min_case,
        max_case=args.max_case,
        batch_size=args.batch_size,
        device=args.device,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "n_rows", "simulation_response", "flow_response", "blend_response",
        "model_response", "m", "bootstrap_standard_error_m", "simulation_matrix",
        "flow_matrix", "model_matrix", "selected_before_lookup",
        "dropped_unmatched_blend", "dropped_unmatched_crowd",
    )}, indent=2))


if __name__ == "__main__":
    main()
