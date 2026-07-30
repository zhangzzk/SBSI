"""Where in TRUE-property cut space is |m| <= 0.3% actually met?

WHY
---
The exact additive decomposition (job 15348636) shows the smallest true-size bin [0.30,0.3552)
contributes **-0.731%** of the -0.508% global m, and the (A)/(B) split (job 15348680) shows why that
bin is special: it is the ONE place where the pin residual and the target defect have the SAME sign
(+0.0250 and +0.0136), so they add instead of cancelling. Everywhere else they oppose.

The deliverable population is defined by a TRUE-property cut (mag < 26, Re > 0.3), and the lower size
edge sits exactly where the flow's response turns over steeply (required response 0.413 in the
boundary bin against 0.697 in the next one). So the acceptance number is unusually sensitive to where
that edge is placed. This scans it.

The point is NOT to shop for a cut that passes -- that would be selecting the deliverable on the
metric. It is to report honestly WHERE the current pipeline meets spec and where it does not, so the
cut is chosen on physics with the calibration cost known. Any change to the deliverable definition is
an owner decision.

m is a ratio of means, so every cut is a re-average of dumps already on disk: no GPU, no reruns.
Errors: across-seed sem plus the per-case bootstrap (common-mode, does not shrink with seeds), in
quadrature -- the same convention as every other quoted m.

FIREWALL: reads constgold per-object dumps. Trains nothing, selects nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.feather as pf

CONSTCAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
            "constant_response_catalogue_train.feather")


def boot_case_err(case, rs, rf, rb, n_boot, seed=0):
    """Per-case bootstrap of m, re-averaging all three quantities with the draw."""
    uc, inv = np.unique(case, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=x, minlength=len(uc)) for x in (rs, rf, rb)])
    N = np.bincount(inv, minlength=len(uc)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uc), size=(n_boot, len(uc)))
    n = N[pick].sum(1)
    num, flow, blend = (S[k][pick].sum(1) / n for k in range(3))
    return float(np.std(num / (flow + blend) - 1.0)) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps")
    ap.add_argument("--tag", default="ablate_s2c_lt500_dom6x6")
    ap.add_argument("--catalogue", default=CONSTCAT)
    ap.add_argument("--re-mins", default="0.30,0.32,0.34,0.356,0.38,0.42,0.45,0.50")
    ap.add_argument("--mag-maxs", default="26.0,25.5,25.0,24.5")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--target", type=float, default=0.3, help="the |m| spec, in percent")
    args = ap.parse_args()

    truth = pf.read_table(args.catalogue,
                          columns=["case", "input_index", "r_input_p", "Re_input_p"],
                          memory_map=True).to_pandas().drop_duplicates(["case", "input_index"])
    truth = truth[["case", "input_index", "Re_input_p"]]

    paths = sorted(glob.glob(os.path.join(args.dump_dir, f"{args.tag}_perobj_s*.feather")))
    if not paths:
        raise SystemExit(f"no dumps for {args.tag}")
    seeds = [int(re.search(r"perobj_s(\d+)", os.path.basename(p)).group(1)) for p in paths]
    print(f"{len(paths)} dumps for {args.tag}, seeds {seeds}\n")

    frames = []
    for p in paths:
        d = pf.read_table(p, columns=["case", "input_index", "r_input_p", "r_sim",
                                      "R_flow", "R_blend"], memory_map=True).to_pandas()
        frames.append(d.merge(truth, on=["case", "input_index"], how="inner"))
    print(f"loaded, {len(frames[0]):,} rows per seed after the size join\n")

    re_mins = [float(x) for x in args.re_mins.split(",")]
    mag_maxs = [float(x) for x in args.mag_maxs.split(",")]

    print("in-domain m (%) by TRUE-property cut. +- is seed sem and the per-case term in quadrature.")
    print(f"a cell is marked PASS when |m| + err <= {args.target}% (i.e. spec is DEMONSTRATED,")
    print(" not merely centred inside it -- the distinction that matters here).\n")
    hdr = f"{'Re >':>7} {'kept':>7} " + " ".join(f"{'mag<' + format(mm, '.1f'):>16}" for mm in mag_maxs)
    print(hdr)
    print("-" * len(hdr))
    for rm in re_mins:
        cells, kept0 = [], None
        for mm in mag_maxs:
            ms, case0 = [], None
            acc = None
            for d in frames:
                sel = ((d["r_input_p"].to_numpy(float) < mm)
                       & (d["Re_input_p"].to_numpy(float) > rm))
                s = d[sel]
                rs, rf, rb = (s[c].to_numpy(float) for c in ("r_sim", "R_flow", "R_blend"))
                ms.append(100.0 * (rs.mean() / (rf.mean() + rb.mean()) - 1.0))
                if acc is None:
                    acc = (s["case"].to_numpy(), rs, rf, rb)
                    kept0 = len(s) if mm == mag_maxs[0] else kept0
            m = float(np.mean(ms))
            sem = float(np.std(ms, ddof=1)) / np.sqrt(len(ms)) if len(ms) > 1 else 0.0
            cerr = boot_case_err(*acc, n_boot=args.n_boot)
            err = float(np.hypot(sem, cerr))
            flag = "PASS" if abs(m) + err <= args.target else ("~" if abs(m) <= args.target else "")
            cells.append(f"{m:+7.3f}+-{err:5.3f}{flag:>4}")
        print(f"{rm:>7.3f} {kept0 / len(frames[0]) * 100:>6.1f}% " + " ".join(cells))

    print("\nlegend: PASS = |m|+err within spec (demonstrated);  ~ = central value inside spec but the")
    print("        error bar is not (the state the certified wide-convention result is also in).")
    print("\nNOTE: raising the lower size cut is a change to the DELIVERABLE DEFINITION and is an owner")
    print("decision, not a calibration fix. This table reports the cost of the current cut, it does not")
    print("recommend moving it.")


if __name__ == "__main__":
    main()
