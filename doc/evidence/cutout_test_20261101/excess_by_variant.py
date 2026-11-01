"""Mean measured-minus-intrinsic e1/e2 (as in doc/evidence/e2_excess_20261029) for the catalogue and each stamp variant,
on the same remeasured galaxies, by MAG_AUTO band. A galaxy enters when all four measurements are finite with |g| < 1.
Errors: spread of per-case means / sqrt(n_cases), and pooled galaxy standard error.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
BANDS = [(0.0, 23.0), (23.0, 24.0), (24.0, 25.0), (25.0, 25.8), (25.8, 99.0), (23.0, 25.8)]
VARIANTS = ("cat", "prod48", "cen49", "cen65")

src, out_path = sys.argv[1], sys.argv[2]
per_case, pooled = {}, {}
for f in sorted(glob.glob(f"{src}/case*_0.02.feather")):
    name = os.path.basename(f)[:-8]
    df = pd.read_feather(f)
    t = pd.read_feather(f"{ROOT}/{name}/real0/catalogues/input/gals_info_{TILE}.feather",
                        columns=["index_input", "e1_input_rot0", "e2_input_rot0"])
    df = df.merge(t, left_on="id_input", right_on="index_input", how="inner", validate="one_to_one")
    good = np.ones(len(df), bool)
    for v in VARIANTS:
        g1, g2 = df[f"{v}_g1"].to_numpy(), df[f"{v}_g2"].to_numpy()
        good &= np.isfinite(g1) & np.isfinite(g2) & (g1 ** 2 + g2 ** 2 < 1.0) & ~((g1 == -1) & (g2 == -1))
    for lo, hi in BANDS:
        s = good & (df["mag_auto"] >= lo) & (df["mag_auto"] < hi)
        for v in VARIANTS:
            d = np.stack([df[f"{v}_g1"][s] - df["e1_input_rot0"][s], df[f"{v}_g2"][s] - df["e2_input_rot0"][s]], 1)
            key = f"{v}|{lo}-{hi}"
            per_case.setdefault(key, []).append(d.mean(0))
            pooled.setdefault(key, []).append(d)
res = {}
for key, means in per_case.items():
    means = np.array(means)
    d = np.concatenate(pooled[key])
    res[key] = dict(n=int(len(d)), n_cases=len(means))
    for k in range(2):
        res[key][f"e{k+1}"] = float(d[:, k].mean())
        res[key][f"e{k+1}_err_cases"] = float(means[:, k].std(ddof=1) / np.sqrt(len(means)))
        res[key][f"e{k+1}_err_galaxies"] = float(d[:, k].std(ddof=1) / np.sqrt(len(d)))
json.dump(dict(description=__doc__.strip().splitlines()[0], results=res), open(out_path, "w"), indent=1)
print("measured - intrinsic, e-4: e1 (case err) | e2 (case err, galaxy err); plus leg g1=+0.02, so e1 carries the shear")
for key, r in res.items():
    print(f"{key:18s} n={r['n']:8d}  e1 {r['e1']*1e4:+7.2f} ({r['e1_err_cases']*1e4:.2f})"
          f"  e2 {r['e2']*1e4:+6.2f} ({r['e2_err_cases']*1e4:.2f}, {r['e2_err_galaxies']*1e4:.2f})")
