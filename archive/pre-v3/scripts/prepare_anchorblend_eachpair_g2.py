"""Prepare matched fixed-g2 pair-rank and coherent-neighbour shear arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_anchorblend_eachpair_g1 import arm_path as g1_arm_path
from scripts.prepare_v22_matched_decomposition import assert_same_latents, generated_path


def pair_arm_path(prefix: str, rank: int) -> str:
    return g1_arm_path(prefix, rank)


def assign_rank_g2(frame: pd.DataFrame, selected: pd.DataFrame,
                   sign: float, g: float) -> pd.DataFrame:
    """Shear only the frozen pair-rank sources along fixed g2."""
    if selected["secondary_index"].duplicated().any():
        raise RuntimeError("one source is selected by multiple anchors in one rank")
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    active = out["index"].isin(selected["secondary_index"]).to_numpy()
    if int(active.sum()) != len(selected):
        raise RuntimeError(f"selected coverage {active.sum()} != {len(selected)}")
    out.loc[active, "g2"] = float(sign) * float(g)
    return out


def assign_coherent_g2(frame: pd.DataFrame, anchor_ids: np.ndarray,
                       sign: float, g: float) -> pd.DataFrame:
    """Shear every non-anchor along g2, matching the old coherent g1 arm."""
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    anchor = out["index"].isin(np.asarray(anchor_ids, np.int64)).to_numpy()
    if int(anchor.sum()) != len(anchor_ids):
        raise RuntimeError(f"anchor coverage {anchor.sum()} != {len(anchor_ids)}")
    out.loc[~anchor, "g2"] = float(sign) * float(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--pair-prefix", required=True)
    ap.add_argument("--coherent-base", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--design-json", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--max-rank", type=int, default=18)
    args = ap.parse_args()
    if args.max_rank < 0:
        raise ValueError("max-rank must be non-negative")
    if os.path.exists(args.design_json):
        raise FileExistsError(f"refusing existing output {args.design_json}")

    manifest_dir = Path(args.manifest_dir)
    summary = {
        "design": (
            "same frozen pair ranks as g1, sheared one rank at a time along g2; "
            "coherent arm shears every non-anchor along g2"
        ),
        "reference_base": args.reference_base,
        "pair_prefix": args.pair_prefix,
        "coherent_base": args.coherent_base,
        "manifest_dir": args.manifest_dir,
        "cases": args.cases,
        "g": args.g,
        "max_rank": args.max_rank,
        "per_case": [],
    }
    for case in args.cases:
        anchors = pd.read_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs = pd.read_feather(manifest_dir / f"pairs_case{case}.feather")
        required = {"anchor_index", "secondary_index", "rank", "response"}
        if missing := required - set(pairs):
            raise KeyError(f"case {case}: pair manifest lacks {sorted(missing)}")
        if pairs.duplicated(["anchor_index", "rank"]).any():
            raise RuntimeError(f"case {case}: duplicate pair rank within anchor")
        if len(pairs) and int(pairs["rank"].max()) > args.max_rank:
            raise RuntimeError(f"case {case}: frozen rank exceeds {args.max_rank}")

        reference = pd.read_feather(generated_path(args.reference_base, case, +args.g))
        anchor_ids = anchors["index"].to_numpy(np.int64)
        if not np.isin(anchor_ids, reference["index"].to_numpy(np.int64)).all():
            raise RuntimeError(f"case {case}: frozen anchors missing from reference")

        for rank in range(args.max_rank + 1):
            base = pair_arm_path(args.pair_prefix, rank)
            selected = pairs.loc[pairs["rank"] == rank]
            for direction in (+1.0, -1.0):
                path = generated_path(base, case, direction * args.g)
                frame = pd.read_feather(path)
                assert_same_latents(reference, frame, f"case {case} rank{rank} g2 {direction:+g}")
                render_tree = os.path.join(base, f"case{case}_{str(direction * args.g)}")
                if os.path.exists(render_tree):
                    raise FileExistsError(f"already-rendered tree {render_tree}")
                assign_rank_g2(frame, selected, direction, args.g).to_feather(path)

        for direction in (+1.0, -1.0):
            path = generated_path(args.coherent_base, case, direction * args.g)
            frame = pd.read_feather(path)
            assert_same_latents(reference, frame, f"case {case} coherent g2 {direction:+g}")
            render_tree = os.path.join(
                args.coherent_base, f"case{case}_{str(direction * args.g)}",
            )
            if os.path.exists(render_tree):
                raise FileExistsError(f"already-rendered tree {render_tree}")
            assign_coherent_g2(frame, anchor_ids, direction, args.g).to_feather(path)

        summary["per_case"].append({
            "case": int(case),
            "n_anchors": int(len(anchors)),
            "n_pairs": int(len(pairs)),
            "max_rank": int(pairs["rank"].max()) if len(pairs) else -1,
        })
        print(
            f"case {case}: anchors={len(anchors):,} pairs={len(pairs):,} "
            f"fixed g2 pair ranks plus coherent arm prepared", flush=True,
        )

    Path(args.design_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.design_json, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote orthogonal design -> {args.design_json}")
    print("ANCHORBLEND_EACHPAIR_G2_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
