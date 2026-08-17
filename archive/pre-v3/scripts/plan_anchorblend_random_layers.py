"""Count deterministic non-overlapping layers of the original anchor manifests."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from prepare_anchorblend_guarded_catalogues import sparse_anchor_layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--min-separation", type=float, default=30.01)
    ap.add_argument("--max-layers", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = []
    totals = np.zeros(args.max_layers, dtype=np.int64)
    max_used = 0
    for case in range(args.case_min, args.case_max + 1):
        manifest = pd.read_feather(
            os.path.join(args.source, f"anchors_case{case}.feather"), columns=["index"],
        )
        frame = pd.read_feather(
            os.path.join(args.source, f"gals{case}_0.05.feather"),
            columns=["index", "RA", "DEC"],
        )
        candidate = frame["index"].isin(manifest["index"]).to_numpy()
        assigned = np.zeros(len(frame), dtype=bool)
        counts = []
        for layer in range(args.max_layers):
            chosen = sparse_anchor_layer(frame, candidate, args.min_separation, layer)
            if np.any(chosen & assigned):
                raise RuntimeError(f"case {case}: layer overlap")
            count = int(chosen.sum())
            counts.append(count)
            totals[layer] += count
            assigned |= chosen
            if int((assigned & candidate).sum()) == len(manifest):
                max_used = max(max_used, layer + 1)
                break
        else:
            raise RuntimeError(f"case {case}: max layer limit did not cover candidates")
        cases.append({
            "case": case, "n_candidates": len(manifest), "layer_counts": counts,
            "n_assigned": int((assigned & candidate).sum()),
        })
        print(f"case {case}: candidates={len(manifest):,} layers={counts}", flush=True)
    payload = {
        "case_window": [args.case_min, args.case_max],
        "min_separation_arcsec": args.min_separation,
        "n_layers_required": max_used,
        "layer_totals": totals[:max_used].tolist(),
        "candidate_total": int(sum(item["n_candidates"] for item in cases)),
        "cases": cases,
    }
    if sum(payload["layer_totals"]) != payload["candidate_total"]:
        raise RuntimeError("layer totals do not cover all candidate anchors")
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"},
                     indent=2, sort_keys=True))
    print("ANCHORBLEND_RANDOM_LAYER_PLAN_DONE", flush=True)


if __name__ == "__main__":
    main()
