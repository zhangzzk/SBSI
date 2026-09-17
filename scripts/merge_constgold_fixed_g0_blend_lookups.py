#!/usr/bin/env python3
"""Merge disjoint fixed-g0 ConstGold R_blend lookup blocks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    expected = set(args.case)
    if len(expected) != len(args.case):
        raise ValueError("duplicate expected cases")
    # ``R_blend`` is the signed sum and can sit near zero through cancellation,
    # so a diagnostic that bins on it cannot tell a genuinely quiet object from
    # a loud one whose contributions happened to cancel.  ``R_blend_abs`` is the
    # sum of the per-pair magnitudes and is the column
    # ``diagnose_constgold_response_split.py`` recommends binning on.  Dropping
    # it here silently removed that option from every merged lookup, so the
    # optional columns are now carried through when every block has them.
    #  Per-shell sums (``R_blend_r<lo>_<hi>`` and their ``_abs`` partners) are
    #  carried the same way.  They are what lets a downstream fit ask ConstGold
    #  how much of each neighbour-separation range is real, instead of forcing
    #  the whole 20-neighbour sum to be wrong by one common factor.  Which shell
    #  columns exist is a property of the blocks, so they are discovered rather
    #  than named here.
    optional = ("R_blend_abs", "blend_pairs")
    available = [set(pd.read_feather(path).columns) for path in args.input]
    shared = set.intersection(*available)
    shells = sorted(name for name in shared if name.startswith("R_blend_r"))
    #  The share columns are per-object ratios rather than additive sums, but a
    #  merge only concatenates rows, so they travel unchanged.  They are the
    #  axis the per-shell fit bins on, so losing them silently would quietly
    #  remove the only well-conditioned binning available.
    shares = sorted(name for name in shared if name.startswith("blend_abs_share_"))
    optional = (*optional, *shells, *shares)
    carried = [name for name in optional if all(name in columns for columns in available)]
    missing = [name for name in optional if name not in carried]
    wanted = ["case", "input_index", "R_blend", *carried]
    frames = []
    inputs = []
    for path, columns in zip(args.input, available):
        absent = {"case", "input_index", "R_blend"} - columns
        if absent:
            raise ValueError(f"{path} is missing required columns: {sorted(absent)}")
        frames.append(pd.read_feather(path, columns=wanted))
        inputs.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["case", "input_index"]).any():
        raise ValueError("lookup blocks overlap or contain duplicate keys")
    actual = set(merged["case"].unique())
    if actual != expected:
        raise ValueError(f"case coverage mismatch: expected={expected}, actual={actual}")
    for name in ["R_blend", *carried]:
        if not np.isfinite(merged[name].to_numpy(float)).all():
            raise ValueError(f"merged lookup contains non-finite {name}")
    if "R_blend_abs" in carried:
        signed = merged["R_blend"].to_numpy(float)
        absolute = merged["R_blend_abs"].to_numpy(float)
        if (absolute + 1.0e-12 < np.abs(signed)).any():
            raise ValueError("absolute blending sum below the signed sum")
    signed_shells = [name for name in shells if not name.endswith("_abs")]
    if signed_shells:
        #  The shells partition the pairs R_blend sums over, so they must add
        #  back to it exactly.  Checking after the merge catches blocks built
        #  with different shell edges being concatenated into one lookup.
        rebuilt = merged[signed_shells].to_numpy(float).sum(axis=1)
        whole = merged["R_blend"].to_numpy(float)
        #  The shells are summed in float64 while R_blend is a float32
        #  accumulation of up to `blend_pairs` per-pair values, so the two agree
        #  only to that precision.  The bound is the standard one for summing n
        #  float32 terms, n * eps * sum|x|, widened by four for summation order.
        eps32 = float(np.finfo(np.float32).eps)
        if "blend_pairs" in carried and "R_blend_abs" in carried:
            scale = (merged["blend_pairs"].to_numpy(float)
                     * merged["R_blend_abs"].to_numpy(float))
        else:
            scale = 20.0 * np.abs(whole)
        bound = 4.0 * eps32 * (scale + 1.0)
        if np.any(np.abs(rebuilt - whole) > bound):
            worst = int(np.argmax(np.abs(rebuilt - whole) - bound))
            raise ValueError(
                "merged per-shell sums do not reconstruct R_blend "
                f"(difference={abs(rebuilt[worst] - whole[worst])}, "
                f"bound={bound[worst]}); blocks may use different shell edges"
            )
    merged = merged.sort_values(["case", "input_index"], kind="stable").reset_index(
        drop=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_feather(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "rows": int(len(merged)),
                "cases": sorted(int(case) for case in actual),
                "columns": list(merged.columns),
                "optional_columns_dropped": missing,
                "inputs": inputs,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"CONSTGOLD_FIXED_G0_RBLEND_MERGED output={args.output} "
        f"rows={len(merged):,} cases={len(actual)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
