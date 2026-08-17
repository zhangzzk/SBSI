"""Strictly concatenate V2.7 scene-feature shards against a manifest."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf


def exact_key_match(left: pd.DataFrame, right: pd.DataFrame,
                    keys: list[str]) -> bool:
    """Compare integer catalogue keys by value, independent of Arrow width."""
    if len(left) != len(right):
        return False
    return np.array_equal(
        left[keys].to_numpy(dtype=np.int64),
        right[keys].to_numpy(dtype=np.int64),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="paths or glob patterns")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = []
    for pattern in args.inputs:
        match = sorted(glob.glob(pattern))
        paths.extend(match if match else [pattern])
    if not paths or any(not os.path.exists(path) for path in paths):
        raise FileNotFoundError(f"missing shard in {paths}")
    parts = [pd.read_feather(path) for path in paths]
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["case", "input_index"], kind="mergesort").reset_index(drop=True)
    keys = ["case", "input_index"]
    if out.duplicated(keys).any():
        raise RuntimeError("feature shards contain duplicate keys")
    manifest = pf.read_table(args.manifest, columns=keys).to_pandas().sort_values(
        keys, kind="mergesort").reset_index(drop=True)
    if not exact_key_match(out, manifest, keys):
        both = manifest.merge(out[keys], on=keys, how="outer", indicator=True)
        raise RuntimeError(
            f"manifest mismatch: features={len(out):,}, manifest={len(manifest):,}, "
            f"counts={both['_merge'].value_counts().to_dict()}")
    feature_columns = [column for column in out if column not in keys]
    numeric = out[[column for column in feature_columns
                   if np.issubdtype(out[column].dtype, np.number)]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite scene feature")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_feather(args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump({
            "output": os.path.abspath(args.output), "manifest": os.path.abspath(args.manifest),
            "inputs": [os.path.abspath(path) for path in paths], "n_rows": len(out),
            "n_cases": int(out["case"].nunique()), "columns": feature_columns,
            "exact_manifest_match": True,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(out):,} rows, exact manifest match", flush=True)


if __name__ == "__main__":
    main()
