"""Cache the flow's predicted self response for a list of cases.

The response-profile builder predicts self *and* blend response and so needs
the pair catalogue and the emulator.  Reading the flow on its own needs only
the flow and the two domain legs, so this entry point reuses that builder's
matching and projection verbatim and skips everything else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from scripts.build_fixed_g0_response_profile_predictions import (
    load_flow_case,
    predict_self_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = tuple(sorted(set(args.case)))
    device = torch.device(args.device)
    bundle = load_measurement_model(args.flow, device=device)
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise RuntimeError("the flow was not conditioned on the fixed-g0 feature set")

    frames, reports = [], {}
    for offset, case in enumerate(cases):
        zero = load_flow_case(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_flow_case(args.domain_root / "flow" / "g05" / f"case{case:03d}.npz")
        frame, counts = predict_self_case(
            bundle,
            zero,
            sheared,
            draws=args.draws,
            batch_size=args.batch_size,
            seed=args.sampling_seed + 10_007 * offset,
        )
        frames.append(frame)
        reports[str(case)] = counts
        print(f"case {case:03d}: {counts['response_rows']} rows", flush=True)

    predictions = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_feather(args.output)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "cases": list(cases),
                "draws": args.draws,
                "sampling_seed": args.sampling_seed,
                "batch_size": args.batch_size,
                "flow": str(args.flow),
                "domain_root": str(args.domain_root),
                "rows": int(len(predictions)),
                "case_reports": reports,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
