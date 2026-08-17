"""Aggregate case-level forward-versus-central R_blend diagnostics."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def stat(values) -> dict:
    values = np.asarray(values, float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "case_sem": sem, "n_cases": int(len(values)),
            "normal_95_interval": [mean - 1.96 * sem, mean + 1.96 * sem]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = sorted(glob.glob(os.path.join(args.input_dir, "case*.json")))
    rows = [json.load(open(path, encoding="utf-8")) for path in paths]
    cases = [row["case"] for row in rows]
    if cases != list(range(args.n_cases)):
        raise RuntimeError(f"expected cases 0--{args.n_cases-1}, got {cases}")

    pair_keys = list(rows[0]["pair"])
    primary_keys = list(rows[0]["primary_sum"])
    payload = {
        "design": rows[0]["design"],
        "case_window": [0, args.n_cases - 1],
        "pair_weighted_within_case": {
            key: stat([row["pair"][key] for row in rows])
            for key in pair_keys if key != "n_pairs"
        },
        "primary_sum_weighted_within_case": {
            key: stat([row["primary_sum"][key] for row in rows])
            for key in primary_keys if key != "n_primaries"
        },
        "n_common_pairs": int(sum(row["pair"]["n_pairs"] for row in rows)),
        "n_common_primaries": int(sum(row["primary_sum"]["n_primaries"] for row in rows)),
        "mean_common_fraction_forward": float(np.mean([row["common_fraction_forward"] for row in rows])),
        "mean_common_fraction_central": float(np.mean([row["common_fraction_central"] for row in rows])),
        "distance_change_max_abs": float(max(row["distance_change_max_abs"] for row in rows)),
        "distance_change_case_mean": stat([row["distance_change_mean"] for row in rows]),
        "separation_bins": [],
        "constgold_opened": False,
    }
    for index in range(len(rows[0]["separation_bins"])):
        items = [row["separation_bins"][index] for row in rows]
        payload["separation_bins"].append({
            "index": index, "lo": items[0]["lo"], "hi": items[0]["hi"],
            **{key: stat([item[key] for item in items])
               for key in items[0] if key not in {"index", "lo", "hi", "n_pairs"}},
            "n_pairs": int(sum(item["n_pairs"] for item in items)),
        })
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANTITHETIC_RBLEND_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
