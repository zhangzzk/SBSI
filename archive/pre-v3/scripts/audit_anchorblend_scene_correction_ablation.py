"""Ablate the exploratory coherent scene correction's fixed feature groups."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from fit_anchorblend_coherent_scene_correction import summarize  # noqa: E402
from fit_anchorblend_scene_correction import (  # noqa: E402
    COND, KEY, MODEL_FEATURES, aggregate_scene_features,
    fit_case_blocked_ensemble,
)


GROUPS = {
    "r_v22_only": ["R_v22"],
    "primary_plus_r_v22": ["r_primary", "log_Re_primary", "log_n_primary", "R_v22"],
    "scene_without_r_v22": [name for name in MODEL_FEATURES if name != "R_v22"],
    "full_frozen_scene": MODEL_FEATURES,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--response", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.base) != len(args.response):
        raise RuntimeError("each --response requires its corresponding --base")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    parts = [pd.read_feather(path) for path in args.response]
    observed = pd.concat(parts, ignore_index=True)
    if observed.duplicated(KEY).any() or sorted(observed.case.unique()) != list(range(300)):
        raise RuntimeError("expected unique cases 0--299")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = pd.concat([
        aggregate_scene_features(base, part, predictor, 0.05)
        for base, part in zip(args.base, parts)
    ], ignore_index=True)
    development = frame.case.to_numpy(int) <= 199
    validation = ~development
    target = (
        frame.R_blend_truth.to_numpy(float)
        - frame.R_blend_lsst_r_extnbr_v22.to_numpy(float)
    )
    constant = float(pd.DataFrame({
        "case": frame.loc[development, "case"].to_numpy(int),
        "target": target[development],
    }).groupby("case", sort=True).target.mean().mean())
    edges = np.quantile(
        frame.loc[development, "R_blend_lsst_r_extnbr_v22"], np.linspace(0, 1, 6),
    )
    edges[0], edges[-1] = -np.inf, np.inf
    arms = {}
    for name, features in GROUPS.items():
        correction, _, stability = fit_case_blocked_ensemble(
            frame, 199, 311, model_features=features,
        )
        arms[name] = {
            "features": features,
            "ensemble_validation_mean_stability_sd": stability,
            "development_oof": summarize(
                frame.loc[development], correction[development], constant, edges,
            ),
            "validation": summarize(
                frame.loc[validation], correction[validation], constant, edges,
            ),
        }
    payload = {
        "design": "fixed coherent c0--199 training and c200--299 descriptive feature ablation",
        "purpose": "interpretation only; validation cases were previously inspected",
        "arms": arms,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_SCENE_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
