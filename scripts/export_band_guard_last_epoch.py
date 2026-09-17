#!/usr/bin/env python3
"""Export the last-epoch model of a band-guard run as a loadable checkpoint.

``selected.pt`` is chosen by the smallest validation objective over the run,
and that objective is evaluated on the same cases the per-leg script then
measures.  The guard term is a noisy Monte Carlo estimate, so taking its
minimum over twenty epochs biases the selected value low on exactly those
cases.  The last epoch involves no selection at all, so measuring it as well
separates a real improvement from a favourable fluctuation that the choice of
checkpoint picked out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    state_path = run_root / "latest_optimizer.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing optimizer state: {state_path}")
    if args.output.exists():
        raise FileExistsError(f"preserving existing checkpoint: {args.output}")

    history = json.loads((run_root / "history.json").read_text())
    protocol = json.loads((run_root / "protocol.json").read_text())
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    epoch = int(state["epoch"])
    if epoch != history["history"][-1]["epoch"]:
        raise ValueError("optimizer state and history disagree on the last epoch")

    bundle = load_measurement_model(args.initial_flow, device=torch.device("cpu"))
    initial = torch.load(args.initial_flow, map_location="cpu", weights_only=False)
    missing, unexpected = bundle.model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise ValueError("last-epoch state does not match the flow architecture")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.save_checkpoint(
        args.output,
        bundle.model,
        bundle.condition_preprocessor,
        bundle.target_transform,
        initial["model_config"],
        {
            "training_phase": "full_domain_band_guard_last_epoch",
            "epoch": epoch,
            "selection_free": True,
            "source_run": str(run_root),
            "guard_weight": protocol["recipe"]["guard_weight"],
            "measurement_conditions_used": False,
        },
    )
    print(f"EXPORTED_LAST_EPOCH epoch={epoch} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
