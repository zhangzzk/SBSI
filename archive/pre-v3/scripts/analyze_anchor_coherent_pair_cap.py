"""Case-balanced closure report for coherent V2.2 k=20 versus wide-k scoring."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    columns = ["gap_k20", "gap_wide", "wide_increment", "n_pairs_k20", "n_pairs_wide"]
    case = frame.groupby("case", sort=True)[columns].mean()
    return {
        "n_rows": int(len(frame)),
        **{column: stat(case[column]) for column in columns},
        "fraction_at_k20_ceiling": float((frame.n_pairs_k20 >= 19).mean()),
        "fraction_with_omitted_pairs": float((frame.n_pairs_wide > frame.n_pairs_k20).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-max", type=int, default=249)
    args = parser.parse_args()
    for path in (args.output_table, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    paths = sorted(glob.glob(os.path.join(args.input_dir, "case*.feather")))
    frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    cases = np.sort(frame.case.unique())
    if len(cases) != 100 or cases[0] != 200 or cases[-1] != 299:
        raise RuntimeError(f"expected cases 200--299, got {cases}")
    frame.to_feather(args.output_table)
    edges = [-0.5, 9.5, 14.5, 18.5, 24.5, np.inf]
    frame["wide_count_bin"] = pd.cut(frame.n_pairs_wide, edges, right=True)
    bins = {}
    for label, subset in frame.groupby("wide_count_bin", observed=True):
        bins[str(label)] = summarize(subset)
    development = frame.case <= args.development_max
    payload = {
        "design": "frozen V2.2 model and coherent anchor truth; only neighbour enumeration changes from persisted k=20 (19 non-self maximum) to k=64",
        "all": summarize(frame),
        "development": summarize(frame.loc[development]),
        "validation": summarize(frame.loc[~development]),
        "by_wide_neighbour_count": bins,
        "k20_prediction_replay_max_abs": float(np.max(np.abs(
            frame.prediction_k20 - frame.R_blend_lsst_r_extnbr_v22
        ))),
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_COHERENT_PAIR_CAP_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
