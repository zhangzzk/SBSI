"""Prepare exact one-neighbour-at-a-time g1 rank arms on existing anchors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_v22_matched_decomposition import assert_same_latents, generated_path


def arm_path(prefix: str, rank: int) -> str:
    return f"{prefix}_rank{rank:02d}_pilot_c400-409"


def assign_ranks(pairs: pd.DataFrame) -> pd.DataFrame:
    """Put the frozen one-active selection first, then stable hash order."""
    required = {
        "anchor_index", "secondary_index", "selection_hash", "selected",
        "response", "distance", "n_pairs",
    }
    if missing := required - set(pairs):
        raise KeyError(f"pair manifest lacks {sorted(missing)}")
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed pair")
    selected_count = pairs.groupby("anchor_index", sort=False)["selected"].sum()
    if not (selected_count == 1).all():
        raise RuntimeError("every non-empty anchor must have one frozen selected pair")
    out = pairs.copy()
    out["rank"] = np.int16(-1)
    out.loc[out["selected"], "rank"] = np.int16(0)
    rest = out.loc[~out["selected"]].sort_values(
        ["anchor_index", "selection_hash", "secondary_index"], kind="mergesort",
    )
    out.loc[rest.index, "rank"] = (
        rest.groupby("anchor_index", sort=False).cumcount().to_numpy(np.int16) + 1
    )
    if (out["rank"] < 0).any() or out.duplicated(["anchor_index", "rank"]).any():
        raise RuntimeError("invalid pair-rank assignment")
    observed = out.groupby("anchor_index")["rank"].max() + 1
    expected = out.groupby("anchor_index")["secondary_index"].size()
    if not np.array_equal(observed.to_numpy(int), expected.to_numpy(int)):
        raise RuntimeError("pair ranks are not contiguous within anchor")
    return out


def assign_rank_g1(frame: pd.DataFrame, selected: pd.DataFrame,
                   sign: float, g: float) -> pd.DataFrame:
    if selected["secondary_index"].duplicated().any():
        raise RuntimeError("one source is selected by multiple anchors in one rank")
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    active = out["index"].isin(selected["secondary_index"]).to_numpy()
    if int(active.sum()) != len(selected):
        raise RuntimeError(f"selected coverage {active.sum()} != {len(selected)}")
    out.loc[active, "g1"] = float(sign) * float(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--max-rank", type=int, default=18)
    args = ap.parse_args()
    if args.max_rank < 0:
        raise ValueError("max-rank must be non-negative")

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    source_manifest = Path(args.source_manifest)
    summary = {
        "design": "one deployed neighbour per rank and anchor; coherent g1 direction",
        "pair_sum_operation": "sum all rank responses, never average",
        "reference_base": args.reference_base,
        "output_prefix": args.output_prefix,
        "source_manifest": args.source_manifest,
        "cases": args.cases,
        "g": args.g,
        "max_rank": args.max_rank,
        "per_case": [],
    }
    for case in args.cases:
        anchors = pd.read_feather(source_manifest / f"anchors_case{case}.feather")
        pairs = assign_ranks(pd.read_feather(source_manifest / f"pairs_case{case}.feather"))
        if len(pairs) and int(pairs["rank"].max()) > args.max_rank:
            raise RuntimeError(
                f"case {case}: needs rank {int(pairs['rank'].max())}, "
                f"above configured {args.max_rank}"
            )
        reference = pd.read_feather(generated_path(args.reference_base, case, +args.g))
        for rank in range(args.max_rank + 1):
            base = arm_path(args.output_prefix, rank)
            selected = pairs.loc[pairs["rank"] == rank]
            for sign in (+1.0, -1.0):
                path = generated_path(base, case, sign * args.g)
                frame = pd.read_feather(path)
                assert_same_latents(reference, frame, f"case {case} rank{rank} {sign:+g}")
                render_tree = os.path.join(base, f"case{case}_{str(sign * args.g)}")
                if os.path.exists(render_tree):
                    raise FileExistsError(f"already-rendered tree {render_tree}")
                assign_rank_g1(frame, selected, sign, args.g).to_feather(path)
        anchors.to_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs.to_feather(manifest_dir / f"pairs_case{case}.feather")
        counts = pairs.groupby("anchor_index").size()
        summary["per_case"].append({
            "case": int(case), "n_anchors": int(len(anchors)),
            "n_pairs": int(len(pairs)), "mean_pairs_per_anchor": float(len(pairs) / len(anchors)),
            "max_pairs_per_anchor": int(counts.max()) if len(counts) else 0,
        })
        print(
            f"case {case}: anchors={len(anchors):,} pairs={len(pairs):,} "
            f"max-rank={int(pairs['rank'].max()) if len(pairs) else -1}", flush=True,
        )
    with open(manifest_dir / "design.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote exact g1 rank design -> {manifest_dir}")
    print("ANCHORBLEND_EACHPAIR_G1_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
