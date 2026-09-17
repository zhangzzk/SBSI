#!/usr/bin/env python3
"""Read measured and modeled flow self-response on fixed-g0 radius edges.

This is a component-level companion to the ConstGold small-radius diagnostic.
It uses cached flow predictions on the flow's own zero/half-shear catalogue,
joins the measured forward response for the same identities, and bins on the
zero-shear measured FLUX_RADIUS.  No blending emulator enters the arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_flow_self_response_bias import (
    case_sums,
    load_leg,
    measured_case,
    summarize,
)


def radius_labels(inner: tuple[float, ...]) -> list[str]:
    edges = (-np.inf, *inner, np.inf)
    return [
        (
            f"<{edges[index + 1]:.4f}"
            if index == 0
            else f">={edges[index]:.4f}"
            if index == len(edges) - 2
            else f"[{edges[index]:.4f},{edges[index + 1]:.4f})"
        )
        for index in range(len(edges) - 1)
    ]


def add_zero_radius(measured: pd.DataFrame, zero: dict[str, np.ndarray]) -> pd.DataFrame:
    radius = pd.DataFrame(
        {
            "case": np.asarray(zero["case"], dtype=np.int16),
            "input_index": np.asarray(zero["input_index"], dtype=np.int64),
            "g0_flux_radius_arcsec": 0.2
            * np.asarray(zero["target"], dtype=np.float64)[:, 2],
        }
    )
    if radius.duplicated(["case", "input_index"]).any():
        raise ValueError("zero-shear radius table contains duplicate keys")
    joined = measured.merge(
        radius,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
    )
    values = joined["g0_flux_radius_arcsec"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("measured response rows lack a physical zero-shear radius")
    return joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--radius-edge", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    inner = tuple(sorted(set(map(float, args.radius_edge))))
    if len(inner) != len(args.radius_edge) or not np.all(np.diff(inner) > 0):
        raise ValueError("radius edges must be unique and strictly increasing")

    predictions = pd.read_feather(args.predictions)
    if predictions.duplicated(["case", "input_index"]).any():
        raise ValueError("prediction table contains duplicate keys")
    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    train_cases = set(map(int, manifest["split"]["train_cases"]))

    frames = []
    for case in sorted(map(int, predictions["case"].unique())):
        zero = load_leg(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_leg(
            args.domain_root / "flow" / "g05" / f"case{case:03d}.npz"
        )
        measured, _ = measured_case(zero, sheared)
        frames.append(add_zero_radius(measured, zero))
    measured = pd.concat(frames, ignore_index=True)
    merged = measured.merge(
        predictions,
        on=["case", "input_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(measured) or len(merged) != len(predictions):
        raise RuntimeError(
            "measured/predicted identity mismatch: "
            f"measured={len(measured)} predicted={len(predictions)} joined={len(merged)}"
        )
    merged["split"] = np.where(
        merged["case"].isin(train_cases), "trained", "held-out"
    )
    labels = radius_labels(inner)
    merged["radius_bin"] = pd.cut(
        merged["g0_flux_radius_arcsec"],
        bins=[-np.inf, *inner, np.inf],
        labels=labels,
        right=False,
    )

    rows = []
    for split in ("all", "trained", "held-out"):
        base = merged if split == "all" else merged.loc[merged["split"] == split]
        for label in labels:
            selected = base.loc[base["radius_bin"] == label]
            if selected.empty:
                continue
            entry = summarize(
                case_sums(selected),
                replicates=args.replicates,
                seed=args.seed,
            )
            rows.append({"split": split, "radius_bin": label, **entry})

    result = {
        "format_version": 1,
        "purpose": (
            "measured versus modeled flow self-response on fixed-g0 measured-radius "
            "edges, with no blending term"
        ),
        "inputs": {
            "domain_root": str(args.domain_root.resolve()),
            "predictions": str(args.predictions.resolve()),
        },
        "radius_edges_arcsec": [None, *inner, None],
        "rows": rows,
        "row_accounting": {
            "measured": int(len(measured)),
            "predicted": int(len(predictions)),
            "joined": int(len(merged)),
        },
        "limitations": [
            "This is the half-shear forward response at |g|=0.05, whereas ConstGold "
            "uses antithetic +/-0.02 legs.",
            "The bins condition on the noisy g=0 radius used in S0 and are not an "
            "independent calibration partition.",
            "Only three held-out cases occur in the cached cases0-19 prediction set.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"FLOW_SELF_RADIUS_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
