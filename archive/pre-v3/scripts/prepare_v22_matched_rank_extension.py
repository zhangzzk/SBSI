"""Prepare exact one-neighbour-per-rank extensions of a matched V2.2 pilot.

The broad ``total``, ``self`` and ``neighbour`` controls already exist.  This
script validates newly generated ``rankN`` catalogues against the same latent
scenes, partitions every deployed V2.2 pair by stable distance rank, and shears
only that rank's neighbour in each mode.  Every pair receives one deterministic
random spin-2 direction.  Summing the later per-rank projections therefore
estimates the individual-pair response sum without Hutchinson cross-neighbour
terms.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_v22_matched_decomposition import (
    assert_same_latents,
    generated_path,
    label,
    random_pair_directions,
)


def numbered_rank_bases(items: list[str]) -> dict[str, str]:
    bases: dict[str, str] = {}
    for item in items:
        mode, sep, path = item.partition("=")
        if not sep or not mode.startswith("rank") or not path:
            raise ValueError(f"--base must be rankN=PATH, got {item!r}")
        if mode in bases:
            raise ValueError(f"duplicate mode {mode!r}")
        bases[mode] = path
    modes = sorted(bases, key=lambda mode: int(mode[4:]))
    for rank, mode in enumerate(modes):
        if mode != f"rank{rank}":
            raise ValueError(f"rank modes must be contiguous rank0..rankN, got {modes}")
    return {mode: bases[mode] for mode in modes}


def assign_rank_shear(frame: pd.DataFrame, sign: float, g: float,
                      selected: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    if selected["secondary_index"].duplicated().any():
        raise RuntimeError("a secondary is assigned more than once in one rank")
    directions = selected.set_index("secondary_index", verify_integrity=True)[["u1", "u2"]]
    mapped = directions.reindex(out["index"].to_numpy(np.int64))
    present = mapped["u1"].notna().to_numpy()
    if int(present.sum()) != len(selected):
        raise RuntimeError(
            f"rank-direction coverage {present.sum()} != selected pairs {len(selected)}"
        )
    direction = 1.0 if sign > 0 else -1.0
    out.loc[present, "g1"] = direction * g * mapped.loc[present, "u1"].to_numpy(float)
    out.loc[present, "g2"] = direction * g * mapped.loc[present, "u2"].to_numpy(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True, help="rankN=PATH")
    ap.add_argument("--reference-base", required=True,
                    help="existing matched total-mode base supplying the latent reference")
    ap.add_argument("--source-manifest", required=True,
                    help="existing matched-pilot manifest containing the exact V2.2 pairs")
    ap.add_argument("--manifest-dir", required=True,
                    help="new immutable manifest for the rank extension")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--random-seed", type=int, default=73021)
    args = ap.parse_args()

    bases = numbered_rank_bases(args.base)
    source_manifest = Path(args.source_manifest)
    with open(source_manifest / "design.json", encoding="utf-8") as handle:
        source_design = json.load(handle)
    if [int(case) for case in source_design["cases"]] != args.cases:
        raise RuntimeError("rank-extension cases differ from source matched design")
    if not np.isclose(float(source_design["g"]), args.g, rtol=0.0, atol=1e-12):
        raise RuntimeError("rank-extension shear differs from source matched design")

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if any(manifest_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty manifest directory {manifest_dir}")

    summary = {
        "design": "one deployed neighbour per stable distance rank",
        "pair_projection": "deterministic random spin-2 direction per deployed pair",
        "pair_sum_operation": "sum across ranks (never average)",
        "source_manifest": str(source_manifest),
        "reference_base": args.reference_base,
        "g": args.g,
        "cases": args.cases,
        "random_seed": args.random_seed,
        "modes": list(bases),
        "per_case": [],
    }
    for case in args.cases:
        reference = pd.read_feather(generated_path(args.reference_base, case, +args.g))
        for mode, base in bases.items():
            for sign in (+args.g, -args.g):
                frame = pd.read_feather(generated_path(base, case, sign))
                assert_same_latents(reference, frame, f"case {case} {mode} {sign:+g}")
                render_tree = os.path.join(base, f"case{case}_{label(sign)}")
                if os.path.exists(render_tree):
                    raise FileExistsError(f"refusing already-rendered tree {render_tree}")

        anchors = pd.read_feather(source_manifest / f"anchors_case{case}.feather")
        pairs = pd.read_feather(source_manifest / f"pairs_case{case}.feather")
        required = {"anchor_index", "secondary_index", "distance", "response"}
        missing = required - set(pairs.columns)
        if missing:
            raise KeyError(f"case {case}: source pair manifest lacks {sorted(missing)}")
        if pairs.duplicated(["anchor_index", "secondary_index"]).any():
            raise RuntimeError(f"case {case}: duplicate source pair")
        if not np.isfinite(pairs["distance"].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite pair distance")

        pairs = pairs.sort_values(
            ["anchor_index", "distance", "secondary_index"], kind="mergesort",
        ).reset_index(drop=True)
        pairs["rank"] = pairs.groupby("anchor_index", sort=False).cumcount().astype(np.int16)
        if len(pairs) and int(pairs["rank"].max()) >= len(bases):
            raise RuntimeError(
                f"case {case}: needs rank{int(pairs['rank'].max())}, "
                f"but only {len(bases)} modes were supplied"
            )
        u1, u2 = random_pair_directions(
            case, pairs["secondary_index"].to_numpy(np.int64), 0, args.random_seed,
        )
        pairs["u1"] = u1
        pairs["u2"] = u2
        if not np.allclose(u1 * u1 + u2 * u2, 1.0, rtol=0.0, atol=2e-15):
            raise RuntimeError(f"case {case}: non-unit pair direction")

        anchors.to_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs.to_feather(manifest_dir / f"pairs_case{case}.feather")
        for rank, (mode, base) in enumerate(bases.items()):
            selected = pairs.loc[pairs["rank"] == rank]
            if selected.duplicated("anchor_index").any():
                raise RuntimeError(f"case {case} {mode}: multiple neighbours per anchor")
            for sign in (+args.g, -args.g):
                path = generated_path(base, case, sign)
                frame = pd.read_feather(path)
                assigned = assign_rank_shear(frame, sign, args.g, selected)
                assigned.to_feather(path)

        counts = pairs.groupby("anchor_index").size()
        summary["per_case"].append({
            "case": case,
            "n_anchors": len(anchors),
            "n_pairs": len(pairs),
            "n_anchors_with_pairs": int(counts.size),
            "max_pairs_per_anchor": int(counts.max()) if len(counts) else 0,
            "mean_pairs_per_anchor": float(len(pairs) / len(anchors)),
        })
        print(
            f"case {case}: anchors={len(anchors):,} pairs={len(pairs):,} "
            f"max-rank={int(pairs['rank'].max()) if len(pairs) else -1} "
            f"modes={len(bases)}",
            flush=True,
        )

    with open(manifest_dir / "design.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote exact rank design -> {manifest_dir}")
    print("V22_MATCHED_RANK_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
