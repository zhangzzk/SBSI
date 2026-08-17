"""Score the V2.2 flow self-response on matched decomposition anchors."""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from scripts.dump_halfshear_selfresp import flow_self_response  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    frame = pd.read_feather(args.input)
    need = [
        "case", "input_index", "e1_input_rot0_p", "e2_input_rot0_p",
        "r_input_p", "Re_input_p", "sersic_n_input_p",
        "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
    ]
    missing = set(need) - set(frame.columns)
    if missing:
        raise KeyError(f"input lacks flow columns {sorted(missing)}")
    base = frame[need].copy()
    # ``flow_self_response`` deliberately uses the shared pair-style ``rescale``
    # preprocessor.  That routine always constructs secondary/distance features
    # before the bundle selects its actual conditioner columns, even though the
    # V2.2 flow below is primary+crowding only.  Supply inert placeholders; the
    # exact feature-name assertion below guarantees none can enter this flow.
    base["Re_input_s"] = base["Re_input_p"]
    base["r_input_s"] = base["r_input_p"]
    base["distance"] = 0.0
    base["neighbored"] = 0
    gh1 = np.ones(len(base), dtype=float)
    gh2 = np.zeros(len(base), dtype=float)
    output = base[["case", "input_index"]].copy()
    seen = set()
    for ckpt in args.ckpt:
        match = re.search(r"_s(\d+)(?:_|\.pt$)", os.path.basename(ckpt))
        if not match:
            raise ValueError(f"cannot parse seed from {ckpt}")
        seed = match.group(1)
        if seed in seen:
            raise ValueError(f"duplicate seed {seed}")
        seen.add(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = load_measurement_model(ckpt, device=device)
        got = list(bundle.condition_preprocessor.feature_names)
        expected = [
            "e1_input_p", "e2_input_p", "sersic_n_input_p", "r_input_p",
            "Re_input_p", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
        ]
        if got != expected:
            raise RuntimeError(f"{ckpt}: unexpected V2.2 conditioning {got}")
        response = flow_self_response(
            bundle, base, gh1, gh2, args.g, args.n_samples, args.batch_size,
            args.flow_seed, extraction="central", ext_delta=args.g,
        )
        output[f"R_flow_s{seed}"] = response
        print(f"s{seed}: N={len(response):,} <R_flow>={np.mean(response):+.6f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output.to_feather(args.output)
    print(f"wrote {args.output}: {len(output):,} anchors x {len(seen)} seeds")
    print("V22_MATCHED_DECOMP_FLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
