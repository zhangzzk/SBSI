"""WHERE does the in-domain m live? An EXACT additive decomposition, from existing dumps.

WHY THIS EXISTS
---------------
The cut-population m for the best cut-trained flow is -0.508 +- 0.240% (dom6x6, 8 seeds) against a
|m| <= 0.3% target. WORKLOG 2026-07-28c narrowed the cause to R_flow and pre-registered the response
TARGET DEFINITION as the remaining suspect, after grid resolution and the finite-difference stencil
were both refuted. What was never done is ask WHERE in the population the residual sits, which is
what decides whether the fix is global (a normalisation / pin-slack problem) or local (a specific
region of true-property space the target mis-measures).

THE DECOMPOSITION IS EXACT, NOT A PER-BIN m AVERAGE
---------------------------------------------------
    m = <r_sim> / (<R_flow> + <R_blend>) - 1 = -E / (<R_flow> + <R_blend>),
    E := <R_flow> + <R_blend> - <r_sim>          (the "excess": how much the model over-responds)

E is a mean, so it splits over any partition of the population with population weights w_b:

    E = sum_b w_b * E_b,   E_b = <R_flow + R_blend - r_sim>_b,   w_b = n_b / n

    => dm_b = -w_b * E_b / (<R_flow> + <R_blend>)   and   sum_b dm_b = m   EXACTLY.

Per-bin m values do NOT add up to the global m; these dm_b do. That is the whole point -- it lets a
bin be locally awful yet globally irrelevant (small w_b), which a per-bin m table hides.

THE PIN-RESIDUAL VIEW
---------------------
The flow is trained to satisfy, per response-target cell, R_flow ~ r_sim - R_blend_emu. So

    implied target_b := <r_sim>_b - <R_blend>_b        and       pin error_b := <R_flow>_b / target_b - 1

is the pin residual re-measured on constgold in whatever bins we choose. E_b = <R_flow>_b - target_b,
i.e. the excess IS the pin residual in absolute units. If pin error is roughly CONSTANT across bins
the defect is a global normalisation (pin slack, or a target built with a different weighting); if it
is concentrated the target is wrong in a specific place.

NO GPU AND NO RERUN. r_sim and R_blend are emulator/flow-independent columns already in the dumps,
R_flow is the only per-seed quantity, and the true-property cut needs one extra column (Re_input_p)
from the constgold catalogue.

FIREWALL: reads constgold per-object dumps. Trains nothing and selects nothing -- this is a
diagnostic that explains an already-measured number, it is not a model-selection instrument.
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
DOMDUMPS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps"


def seed_of(path):
    mo = re.search(r"_s(\d+)(?:_|\.)", os.path.basename(path))
    return int(mo.group(1)) if mo else -1


def load_truth(cat_path):
    """(case, input_index) -> true mag and true Re, for the cut and for the binning axes."""
    t = pf.read_table(cat_path,
                      columns=["case", "input_index", "r_input_p", "Re_input_p"],
                      memory_map=True).to_pandas()
    dup = t.duplicated(["case", "input_index"]).sum()
    if dup:
        print(f"  *** {dup:,} duplicate (case,input_index) keys -- deduplicating ***")
        t = t.drop_duplicates(["case", "input_index"])
    return t


def edges_from_quantiles(x, n):
    """Equal-population edges, so every bin carries the same weight w_b = 1/n.

    Equal-population binning is deliberate: it makes dm_b directly comparable between bins (no bin
    can look important merely by being big), which is what we want when hunting for a LOCAL defect.
    """
    q = np.linspace(0.0, 1.0, n + 1)
    e = np.quantile(x, q)
    e[0], e[-1] = -np.inf, np.inf
    return e


def decompose(df, key, edges, labels=None):
    """Exact additive split of m over the bins of `key`. df carries r_sim/R_flow/R_blend/w columns."""
    idx = np.digitize(df[key].to_numpy(), edges[1:-1], right=False)
    g = df.groupby(idx)
    n = len(df)
    denom = df["R_flow"].mean() + df["R_blend"].mean()
    rows = []
    for b, sub in g:
        w = len(sub) / n
        rs, rf, rb = sub["r_sim"].mean(), sub["R_flow"].mean(), sub["R_blend"].mean()
        tgt = rs - rb
        e_b = rf + rb - rs
        rows.append(dict(
            bin=labels[b] if labels else f"[{edges[b]:.3g},{edges[b + 1]:.3g})",
            w=w, n=len(sub), r_sim=rs, R_flow=rf, R_blend=rb,
            target=tgt, pin_err_pct=100.0 * (rf / tgt - 1.0) if tgt != 0 else np.nan,
            m_local_pct=100.0 * (rs / (rf + rb) - 1.0),
            dm_pct=-100.0 * w * e_b / denom,
        ))
    return pd.DataFrame(rows)


def show(name, tab, total_m):
    print(f"\n--- by {name} " + "-" * (66 - len(name)))
    print(f"{'bin':>16} {'w':>7} {'r_sim':>8} {'R_flow':>8} {'target':>8} "
          f"{'pin err':>9} {'m local':>9} {'dm -> global':>12}")
    for _, r in tab.iterrows():
        print(f"{r['bin']:>16} {r['w']:>7.3f} {r['r_sim']:>8.4f} {r['R_flow']:>8.4f} "
              f"{r['target']:>8.4f} {r['pin_err_pct']:>8.2f}% {r['m_local_pct']:>8.2f}% "
              f"{r['dm_pct']:>11.3f}%")
    s = tab["dm_pct"].sum()
    print(f"{'SUM':>16} {tab['w'].sum():>7.3f} {'':>8} {'':>8} {'':>8} {'':>9} {'':>9} "
          f"{s:>11.3f}%   (global m = {total_m:+.3f}%, closure {abs(s - total_m):.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DOMDUMPS)
    ap.add_argument("--tag", default="ablate_s2c_lt500_dom6x6",
                    help="dump basename prefix, e.g. ablate_s2c_lt500_dom6x6")
    ap.add_argument("--catalogue", default=CONSTCAT)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--no-cut", action="store_true",
                    help="skip the true-property cut (WIDE convention) for comparison")
    ap.add_argument("--nbin", type=int, default=6)
    ap.add_argument("--alt-lookup", default=None,
                    help="optional emulator lookup to swap in for R_blend before decomposing")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dump_dir, f"{args.tag}_perobj_s*.feather")))
    if not paths:
        raise SystemExit(f"no dumps matching {args.tag}_perobj_s*.feather in {args.dump_dir}")
    print(f"{len(paths)} dumps for tag {args.tag}: seeds {[seed_of(p) for p in paths]}")

    truth = load_truth(args.catalogue)
    if args.no_cut:
        keep = truth[["case", "input_index", "r_input_p", "Re_input_p"]]
        print("WIDE convention: no true-property cut applied")
    else:
        sel = ((truth["r_input_p"].to_numpy(float) < args.true_mag_max)
               & (truth["Re_input_p"].to_numpy(float) > args.true_re_min))
        keep = truth[sel][["case", "input_index", "r_input_p", "Re_input_p"]]
        print(f"in-domain cut mag < {args.true_mag_max}, Re > {args.true_re_min}: "
              f"{len(truth):,} -> {len(keep):,} ({100 * len(keep) / len(truth):.1f}%)")

    alt = None
    if args.alt_lookup:
        alt = pf.read_table(args.alt_lookup, memory_map=True).to_pandas()
        cols = [c for c in alt.columns if c.lower() in ("r_blend", "rblend", "blend_response")]
        if not cols:
            raise SystemExit(f"no R_blend-like column in {args.alt_lookup}: {list(alt.columns)}")
        alt = alt[["case", "input_index", cols[0]]].rename(columns={cols[0]: "R_blend_alt"})
        print(f"alt lookup {os.path.basename(args.alt_lookup)}: {len(alt):,} rows")

    # Accumulate the decomposition per seed, then average. Averaging the TABLES (not the objects)
    # matches the convention every quoted ensemble m uses: mean over per-seed m, not m of the mean.
    tabs = {k: [] for k in ("mag", "size", "nbr", "sep", "magsize")}
    per_seed_m = []
    ref = None
    for p in paths:
        d = pf.read_table(p, memory_map=True).to_pandas()
        d = d.merge(keep, on=["case", "input_index"], how="inner", suffixes=("", "_cat"))
        if alt is not None:
            d = d.merge(alt, on=["case", "input_index"], how="left")
            miss = d["R_blend_alt"].isna().mean()
            d["R_blend"] = d["R_blend_alt"].fillna(d["R_blend"])
            if miss:
                print(f"  seed {seed_of(p)}: alt lookup missing for {100 * miss:.2f}% of rows")
        mm = 100.0 * (d["r_sim"].mean() / (d["R_flow"].mean() + d["R_blend"].mean()) - 1.0)
        per_seed_m.append(mm)

        if ref is None:                      # fix the bin edges once, from the first seed
            ref = dict(
                mag=edges_from_quantiles(d["r_input_p"].to_numpy(), args.nbin),
                size=edges_from_quantiles(d["Re_input_p"].to_numpy(), args.nbin),
                sep=np.array([-np.inf, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]),
            )
        tabs["mag"].append(decompose(d, "r_input_p", ref["mag"]))
        tabs["size"].append(decompose(d, "Re_input_p", ref["size"]))
        tabs["nbr"].append(decompose(d, "neighbored", np.array([-np.inf, 0.5, np.inf]),
                                     labels=["isolated", "neighboured"]))
        nb = d[d["neighbored"].to_numpy(bool)].copy()
        tabs["sep"].append(decompose(nb, "distance", ref["sep"]))
        # 2D: which (mag,size) corner carries it
        mi = np.digitize(d["r_input_p"].to_numpy(), ref["mag"][1:-1])
        si = np.digitize(d["Re_input_p"].to_numpy(), ref["size"][1:-1])
        d["_cell"] = mi * args.nbin + si
        # label each cell with its own index so the readout below needs no edge arithmetic
        tabs["magsize"].append(decompose(
            d, "_cell", np.arange(-0.5, args.nbin * args.nbin + 0.5, 1.0),
            labels=[str(i) for i in range(args.nbin * args.nbin)]))
        del d, nb

    total_m = float(np.mean(per_seed_m))
    sd = float(np.std(per_seed_m, ddof=1)) if len(per_seed_m) > 1 else 0.0
    print("\n" + "=" * 78)
    print(f"ENSEMBLE m = {total_m:+.3f}%   (sd {sd:.3f}, sem {sd / max(1, np.sqrt(len(paths))):.3f}, "
          f"n={len(paths)})")
    print("per-seed:", "  ".join(f"{v:+.3f}" for v in per_seed_m))
    print("=" * 78)
    print("\npin err = <R_flow>/( <r_sim> - <R_blend> ) - 1, i.e. the pin residual re-measured here.")
    print("dm      = that bin's EXACT additive contribution to the global m; the column SUMS to m.")

    def seed_average(frames):
        """Mean of the per-seed tables. Only numeric columns can be summed; `bin` is a label."""
        num = [f.drop(columns=["bin"]) for f in frames]
        avg = sum(num) / len(num)
        avg.insert(0, "bin", frames[0]["bin"].to_numpy())
        return avg

    for k, nm in (("mag", "TRUE MAG"), ("size", "TRUE SIZE"),
                  ("nbr", "NEIGHBOURED"), ("sep", "SEPARATION (neighboured only)")):
        avg = seed_average(tabs[k])
        show(nm, avg, total_m if k != "sep" else avg["dm_pct"].sum())

    ms = seed_average(tabs["magsize"])
    ms = ms[ms["n"] > 0].sort_values("dm_pct")
    print("\n--- worst 8 (mag,size) cells by contribution to m " + "-" * 26)
    print(f"{'cell(mag,size)':>16} {'w':>7} {'pin err':>9} {'m local':>9} {'dm -> global':>12}")
    for _, r in pd.concat([ms.head(5), ms.tail(3)]).iterrows():
        c = int(r["bin"])
        print(f"{f'({c // args.nbin},{c % args.nbin})':>16} {r['w']:>7.3f} "
              f"{r['pin_err_pct']:>8.2f}% {r['m_local_pct']:>8.2f}% {r['dm_pct']:>11.3f}%")
    print(f"\n(mag index 0 = brightest, size index 0 = smallest; {args.nbin} equal-population bins each)")


if __name__ == "__main__":
    main()
