"""Summarize the centred-cutout remeasurement: per-galaxy (variant - prod48) shape differences by MAG_AUTO band.

A galaxy enters a comparison when both fits are finite with |g| < 1. Errors are the spread of per-case means over
cases (std/sqrt(n_cases)); the pooled galaxy-level standard error is reported alongside. prod48 - catalogue checks
that the production replay reproduces the stored catalogue up to optimizer noise.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MAG_EDGES = [0.0, 23.0, 24.0, 25.0, 25.8, 99.0]
BANDS = [(MAG_EDGES[i], MAG_EDGES[i + 1]) for i in range(len(MAG_EDGES) - 1)] + [(23.0, 25.8)]
PAIRS = [("cen49", "prod48"), ("cen65", "prod48"), ("cen65", "cen49"), ("prod48", "cat")]


def ok(df, v):
    g1, g2 = df[f"{v}_g1"].to_numpy(), df[f"{v}_g2"].to_numpy()
    good = np.isfinite(g1) & np.isfinite(g2) & (g1 ** 2 + g2 ** 2 < 1.0)
    if v == "cat":
        good &= ~((g1 == -1.0) & (g2 == -1.0))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(f"{a.dir}/case*_0.02.feather"))
    per_case = {}
    pooled = {}
    for f in files:
        df = pd.read_feather(f)
        name = os.path.basename(f).replace(".feather", "")
        for v, w in PAIRS:
            sel_vw = ok(df, v) & ok(df, w)
            for lo, hi in BANDS:
                s = sel_vw & (df["mag_auto"] >= lo) & (df["mag_auto"] < hi)
                key = f"{v}-{w}|{lo}-{hi}"
                d = np.stack([df[f"{v}_g1"][s] - df[f"{w}_g1"][s], df[f"{v}_g2"][s] - df[f"{w}_g2"][s]], 1)
                per_case.setdefault(key, []).append((name, len(d), d.mean(0) if len(d) else [np.nan, np.nan]))
                pooled.setdefault(key, []).append(d)
    out = dict(description=__doc__.strip().splitlines()[0], n_cases=len(files),
               cases=[os.path.basename(f) for f in files], mag_auto_bands=BANDS, results={})
    for key, rows in per_case.items():
        means = np.array([r[2] for r in rows], dtype=float)
        d = np.concatenate(pooled[key])
        n_c = np.sum(np.isfinite(means[:, 0]))
        res = dict(n_galaxies=int(len(d)))
        for k in range(2):
            res[f"dg{k+1}"] = float(d[:, k].mean())
            res[f"dg{k+1}_err_cases"] = float(np.nanstd(means[:, k], ddof=1) / np.sqrt(n_c))
            res[f"dg{k+1}_err_galaxies"] = float(d[:, k].std(ddof=1) / np.sqrt(len(d)))
            res[f"dg{k+1}_median"] = float(np.median(d[:, k]))
        out["results"][key] = res
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for key, r in out["results"].items():
        print(f"{key:28s} n={r['n_galaxies']:8d}  dg1 {r['dg1']:+.2e} ± {r['dg1_err_cases']:.2e}"
              f"  dg2 {r['dg2']:+.2e} ± {r['dg2_err_cases']:.2e} (gal ± {r['dg2_err_galaxies']:.2e})")


if __name__ == "__main__":
    main()
