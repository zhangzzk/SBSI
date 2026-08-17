"""Is the forward-vs-antithetic extraction gap real, and does it depend on resolution?

WHY. CONVENTIONS.md 6c records a 0.49-vs-0.60 self-response gap between the forward (`0 -> +g`) and
antithetic (`+-g`) extractions on image-identical sims, and calls it "pure extraction convention ...
not physics". WORKLOG 2026-08-05n then found that tonight's target-vs-constgold comparison crosses
exactly that boundary, which is why it could not be read as one SIM being wrong. This measures the
convention directly, with NO sim difference in play at all: one catalogue, one estimator, both
extractions on identical rows.

THE IDENTITY THAT MAKES THIS SHARP. With `p`, `z`, `m` the shapes at `+g`, `0`, `-g`, projected on
the shear direction:

    R_fwd = (p - z)/g        R_bwd = (z - m)/g        R_anti = (p - m)/(2g)

and therefore, EXACTLY and row by row,

    R_anti == (R_fwd + R_bwd) / 2

So on identical rows the two conventions cannot disagree by anything except the ASYMMETRY between
the two forward legs -- and that asymmetry is the QUADRATIC term in the response. There is no third
possibility. Either the quadratic term is big enough to explain the recorded gap, or the gap is not
a convention effect on identical rows and something else (different sims, different populations,
different estimators) was carrying it.

WHAT THIS BUYS FOR V2.1. The open item is that the flow's closure residual CHANGES SIGN with
resolution (-1.82% well-resolved, +2.86% poorly-resolved). If the extraction nonlinearity also grows
toward small sizes, the two are candidates for the same effect. If it is flat, the convention is
cleared and the sign flip belongs to the flow.

CAVEAT ON |g|. constgold is at |g| = 0.02. A quadratic term scales as g^2, so the half-shear legs at
|g| = 0.05 and 0.2 carry 6x and 100x more of it. This measures the coefficient here and says what it
implies there; it does not measure the 0.2 leg.

FIREWALL. constgold is EVALUATION ONLY. Nothing here trains, tunes, promotes, or corrects anything.
No model is involved: this is sim-vs-sim within one catalogue, so no `m` is reported and the 16-seed
rule does not apply.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from sbs_shear import domain as sbs_domain

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
G0 = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/g0_lookup_c0-99.feather"

COLS = ["case", "input_index", "shear_angle", "applied_g1", "applied_g2", "et_plus",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "Re_input_p", sbs_domain.MAG_COLUMN]


def load(min_case, max_case):
    parts = []
    with ipc.open_file(CG) as r:
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(COLS).to_pandas()
            b = b[(b["case"] >= min_case) & (b["case"] <= max_case)]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True)


def project(df, extra=()):
    """Project every shape onto the applied shear direction, with constgold's own sign fix."""
    sa = df["shear_angle"].to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    p1, p2 = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    yp = p1 * c2 + p2 * s2
    st = df["et_plus"].to_numpy(float)
    ok = np.isfinite(yp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((yp * st)[ok]) > 0 else -1.0
    out = {"p": sign * yp,
           "m": sign * (df["measured_e1_minus"].to_numpy(float) * c2
                        + df["measured_e2_minus"].to_numpy(float) * s2)}
    for name, (a, b) in extra:
        out[name] = sign * (df[a].to_numpy(float) * c2 + df[b].to_numpy(float) * s2)
    return out


def stats(x):
    """Mean and its sem, on the rows given."""
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def report(label, p, z, m, g, n_expect=None):
    r_fwd, r_bwd = (p - z) / g, (z - m) / g
    r_ant = (p - m) / (2 * g)
    fa, sfa = stats(r_fwd)
    ba, sba = stats(r_bwd)
    aa, saa = stats(r_ant)
    d = r_fwd - r_ant                       # == (R_fwd - R_bwd)/2, the quadratic term
    da, sda = stats(d)
    print(f"\n[{label}]  N = {len(p):,}"
          + (f"   ({100*len(p)/n_expect:.1f}% of matched)" if n_expect else ""))
    print(f"  R_fwd  (0 -> +g)      = {fa:.5f} +- {sfa:.5f}")
    print(f"  R_bwd  (0 -> -g)      = {ba:.5f} +- {sba:.5f}")
    print(f"  R_anti (+-g)          = {aa:.5f} +- {saa:.5f}")
    print(f"  fwd - anti            = {da:+.5f} +- {sda:.5f}  ({abs(da)/sda:5.1f} sigma)"
          f"   = {100*da/aa:+.2f}% of R_anti")
    return aa, da, sda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=99)
    args = ap.parse_args()

    cg = load(args.min_case, args.max_case)
    g0 = pf.read_table(G0).to_pandas()
    n_cg = len(cg)

    # Baseline BEFORE the join: R_anti needs no g=0 leg, so it can be formed on every constgold row.
    # Comparing it against R_anti on the matched subset says whether the join is population-selective
    # -- the trap AGENTS.md records four separate times.
    pr_all = project(cg)
    fin_all = np.isfinite(pr_all["p"]) & np.isfinite(pr_all["m"])
    g_all = np.hypot(cg["applied_g1"].to_numpy(float), cg["applied_g2"].to_numpy(float))
    r_all, s_all = stats(((pr_all["p"] - pr_all["m"]) / (2 * g_all))[fin_all])

    df = cg.merge(g0, on=["case", "input_index"])
    print(f"constgold cases {args.min_case}-{args.max_case}: {n_cg:,}   "
          f"matched to the g=0 lookup: {len(df):,} ({len(df)/max(n_cg,1):.2%})")

    pr = project(df, extra=[("z", ("ngmix0_g1", "ngmix0_g2"))])
    p, z, m = pr["p"], pr["z"], pr["m"]
    g = np.hypot(df["applied_g1"].to_numpy(float), df["applied_g2"].to_numpy(float))
    fin = np.isfinite(p) & np.isfinite(z) & np.isfinite(m) & (g > 0)
    p, z, m, g = p[fin], z[fin], m[fin], g[fin]
    mag = df[sbs_domain.MAG_COLUMN].to_numpy(float)[fin]
    re_ = df["Re_input_p"].to_numpy(float)[fin]
    print(f"finite on all three legs: {len(p):,}    |g| = {g.min():.4f} to {g.max():.4f}")

    r_match, s_match = stats((p - m) / (2 * g))
    dsel = r_match - r_all
    print(f"\nJOIN SELECTIVITY CHECK (R_anti needs no g=0 leg, so it exists on both sets)")
    print(f"  all constgold rows : {r_all:.5f} +- {s_all:.5f}   (N = {int(fin_all.sum()):,})")
    print(f"  matched subset     : {r_match:.5f} +- {s_match:.5f}   (N = {len(p):,})")
    print(f"  difference         : {dsel:+.5f}  -- the matched half is "
          f"{'REPRESENTATIVE' if abs(dsel) < 3*np.hypot(s_all, s_match) else 'SELECTIVE, read on with care'}")

    print(f"\n{'='*100}\nEXTRACTION CONVENTION, SAME ROWS, SAME ESTIMATOR\n{'='*100}")
    print("R_anti == (R_fwd + R_bwd)/2 EXACTLY, row by row, so `fwd - anti` IS the quadratic term.")
    r_ref, _, _ = report("ALL matched rows", p, z, m, g)

    dom = sbs_domain.in_domain(mag, re_)
    n = len(p)
    report("V2.1 domain (well-resolved)", p[dom], z[dom], m[dom], g[dom], n)
    report("COMPLEMENT (poorly-resolved)", p[~dom], z[~dom], m[~dom], g[~dom], n)

    print(f"\n{'='*100}\nDOES THE NONLINEARITY TRACK RESOLUTION?\n{'='*100}")
    print(f"  {'Re_input_p':>18}{'R_anti':>10}{'fwd-anti':>11}{'+-':>9}{'sig':>7}{'% of R':>9}{'N':>13}")
    edges = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.5]
    for i in range(len(edges) - 1):
        k = (re_ >= edges[i]) & (re_ < edges[i + 1])
        if k.sum() < 2000:
            continue
        a, _ = stats((p[k] - m[k]) / (2 * g[k]))
        d, sd = stats((p[k] - z[k]) / g[k] - (p[k] - m[k]) / (2 * g[k]))
        print(f"  [{edges[i]:>7.2f},{edges[i+1]:>6.2f}){a:>10.4f}{d:>+11.5f}{sd:>9.5f}"
              f"{abs(d)/sd:>6.1f}s{100*d/a:>+8.2f}%{k.sum():>13,}")

    print(f"\n  {'true mag':>18}{'R_anti':>10}{'fwd-anti':>11}{'+-':>9}{'sig':>7}{'% of R':>9}{'N':>13}")
    medges = [18, 22, 23, 24, 24.5, 25, 25.5, 26, 27, 30]
    for i in range(len(medges) - 1):
        k = (mag >= medges[i]) & (mag < medges[i + 1])
        if k.sum() < 2000:
            continue
        a, _ = stats((p[k] - m[k]) / (2 * g[k]))
        d, sd = stats((p[k] - z[k]) / g[k] - (p[k] - m[k]) / (2 * g[k]))
        print(f"  [{medges[i]:>7.1f},{medges[i+1]:>6.1f}){a:>10.4f}{d:>+11.5f}{sd:>9.5f}"
              f"{abs(d)/sd:>6.1f}s{100*d/a:>+8.2f}%{k.sum():>13,}")

    print(f"\n{'='*100}\nWHAT IT WOULD TAKE TO EXPLAIN THE RECORDED 0.49-vs-0.60 GAP\n{'='*100}")
    print("That gap is 22% of the smaller value. A quadratic term scales as g^2, so a coefficient")
    print(f"measured here at |g| = {np.median(g):.3f} implies 6.2x more at |g| = 0.05 and 100x at 0.2:")
    d, sd = stats((p - z) / g - (p - m) / (2 * g))
    for gl in (0.02, 0.05, 0.2):
        sc = (gl / np.median(g)) ** 2
        print(f"  at |g| = {gl:.2f}:  fwd - anti = {100*d*sc/r_ref:+7.2f}% of R"
              f"   (+- {100*sd*sc/r_ref:.2f})")
    print("\nEXTRACTION_CONVENTION_DONE", flush=True)


if __name__ == "__main__":
    main()
