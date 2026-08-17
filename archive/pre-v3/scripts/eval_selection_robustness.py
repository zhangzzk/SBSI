#!/usr/bin/env python -B
"""Acceptance harness: worst-case |m| of the parameter-free estimator under ARBITRARY selections.

Goal metric for the weekend milestone: keep |m| <= 3% under any (reasonable) selection of the
detected sample.  This script fixes a selection family A PRIORI and reports the worst-case |m|
over it, with case-bootstrap error bars so a noise spike is never mistaken for non-closure.

m under a selection S:   m_S = <r_sim>_S / (<R_flow>_S + <R_blend>_S) - 1
computed on the CONSTANT-shear certification per-object dump (cases 40-139, +/-0.02 CRN),
which is exactly the population the certified global m=+0.245% is harvested from.

The selection family (FIXED HERE, never tuned to shrink |m| -- firewall):
  * global (sanity: should reproduce the certified global m)
  * deciles of true magnitude r_input_p
  * neighbored in {0,1}
  * quartiles of distance / nbr_flux_{near,far,max} / ood_flux_{bright,faint}  (blended only)
  * 2D cells: mag-decile x neighbored ; mag-quintile x nbr_flux_max-quartile (blended)
  * 300 RANDOM linear cuts over the standardized feature vector (fixed seed), top/bottom 30%
    -> genuinely "arbitrary" adversarial sub-populations, not hand-picked axes

Head-to-head support: --rflow-override / --rblend-override take an npz with (case, input_index,
value) to swap in a different R_flow (e.g. the joint-flow harvest) or R_blend (e.g. a separate
scene model), keeping everything else identical.  Default = the certified dump columns.

DIAGNOSTIC / EVALUATION ONLY.  Reads existing r_sim/R_flow/R_blend; tunes nothing.
"""
import argparse, os, time, json
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf

DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
CB = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
OUTNPZ = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selrobust_{tag}.npz"
OUTTXT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selrobust_{tag}.txt"

t0 = time.time(); _o = []
def emit(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _o.append(s)
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def load(dump):
    cols = ["case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend", "neighbored", "distance"]
    parts = {c: [] for c in cols}
    with pa.memory_map(dump, "r") as src:
        r = ipc.open_file(src)
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            for c in cols:
                parts[c].append(b.column(c).to_numpy(zero_copy_only=False))
    df = pd.DataFrame({c: np.concatenate(parts[c]) for c in cols})
    cf = pf.read_table(CB + "crowd_flux_conc_c0-199.feather", memory_map=True).to_pandas()[
        ["case", "input_index", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]]
    df = df.merge(cf, on=["case", "input_index"], how="left"); del cf
    od = pf.read_table(CB + "ood_split_c40-139.feather", memory_map=True).to_pandas()[
        ["case", "input_index", "ood_flux_bright", "ood_flux_faint"]]
    df = df.merge(od, on=["case", "input_index"], how="left"); del od
    for c in ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max", "ood_flux_bright", "ood_flux_faint"]:
        df[c] = df[c].fillna(0.0)
    szp = CB + "true_size_lookup_c40-139.feather"
    if os.path.exists(szp):                              # true-size (Re) for realistic SIZE cuts
        sz = pf.read_table(szp, memory_map=True).to_pandas()[["case", "input_index", "Re_input_p"]]
        df = df.merge(sz, on=["case", "input_index"], how="left"); del sz
    return df


MEAS_COLS = ["measured_mag_auto", "measured_flux_radius", "measured_flux_auto",
             "measured_fwhm_image", "measured_isoarea_image", "measured_class_star"]


def join_measured(df):
    """Join per-object MEASURED (shear-dependent) primary observables -- the quantities a real
    survey selects on. Enables the realistic 'arbitrary selections' test (measured-observable
    cuts have a shear-dependent boundary -> exercise the selection response)."""
    mp = pf.read_table(CB + "meas_prim_lookup_c0-139.feather", memory_map=True).to_pandas()[
        ["case", "input_index"] + MEAS_COLS]
    n0 = len(df)
    df = df.merge(mp, on=["case", "input_index"], how="left"); del mp
    matched = int(df[MEAS_COLS[0]].notna().sum())
    print(f"  measured join: {matched:,}/{n0:,} ({100*matched/n0:.1f}%) have measured observables", flush=True)
    return df


def apply_override(df, path, col):
    z = np.load(path)
    ov = pd.DataFrame({"case": z["case"].astype(np.int64),
                       "input_index": z["input_index"].astype(np.int64),
                       "_ov": z["value"].astype(float)})
    n0 = len(df)
    df = df.merge(ov, on=["case", "input_index"], how="left")
    miss = int(df["_ov"].isna().sum())
    log(f"override {col}: matched {n0-miss:,}/{n0:,} ({100*(n0-miss)/n0:.1f}%), {miss:,} unmatched keep original")
    df[col] = np.where(df["_ov"].notna(), df["_ov"], df[col])
    return df.drop(columns="_ov")


def selections(df, n_dir=300, seed=0, with_measured=False):
    """Yield (label, kind, mask) for the FIXED a-priori family."""
    mag = df.r_input_p.to_numpy(float)
    ngh = df.neighbored.to_numpy().astype(int)
    dist = df.distance.to_numpy(float)
    bl = (ngh == 1) & np.isfinite(dist)
    feats = {k: df[k].to_numpy(float) for k in
             ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max", "ood_flux_bright", "ood_flux_faint"]}
    allrows = np.ones(len(mag), bool)

    def qslabs(x, mask, nq, name, kind):
        good = mask & np.isfinite(x)
        e = np.quantile(x[good], np.linspace(0, 1, nq + 1)); e[0] -= 1e-9; e[-1] += 1e-9
        b = np.digitize(x, e[1:-1])
        for k in range(nq):
            yield (f"{name}_q{k+1}/{nq}", kind, good & (b == k))

    yield ("GLOBAL", "global", np.ones(len(mag), bool))
    for s in qslabs(mag, np.ones(len(mag), bool), 10, "mag", "trueprop"): yield s
    yield ("isolated", "trueprop", ngh == 0)
    yield ("blended", "trueprop", ngh == 1)
    for s in qslabs(dist, bl, 4, "dist", "env"): yield s
    for nm in feats:
        for s in qslabs(feats[nm], bl, 4, nm, "flux"): yield s
    # MEASURED-observable selections (shear-dependent boundary; the realistic survey cuts)
    meas = {}
    if with_measured:
        meas = {c: df[c].to_numpy(float) for c in MEAS_COLS}
        for s in qslabs(meas["measured_mag_auto"], allrows, 10, "meas_mag", "measured"): yield s
        for c in ["measured_flux_radius", "measured_flux_auto", "measured_fwhm_image",
                  "measured_isoarea_image"]:
            for s in qslabs(meas[c], allrows, 4, "meas_" + c.replace("measured_", ""), "measured"): yield s
        cs = meas["measured_class_star"]
        yield ("meas_class_star_hi", "measured", np.isfinite(cs) & (cs >= 0.5))
        yield ("meas_class_star_lo", "measured", np.isfinite(cs) & (cs < 0.5))
        # 2D measured cell: measured-mag decile x measured-size quartile
        mm = meas["measured_mag_auto"]; me = np.quantile(mm[np.isfinite(mm)], np.linspace(0, 1, 11))
        me[0] -= 1e-9; me[-1] += 1e-9; mmb = np.digitize(mm, me[1:-1])
        rr = meas["measured_flux_radius"]; good = np.isfinite(rr)
        re_ = np.quantile(rr[good], np.linspace(0, 1, 5)); re_[0] -= 1e-9; re_[-1] += 1e-9
        rrb = np.digitize(rr, re_[1:-1])
        for k in range(10):
            for j in range(4):
                yield (f"meas_mag_d{k+1}xsize_q{j+1}", "measured",
                       np.isfinite(mm) & good & (mmb == k) & (rrb == j))
    # 2D cells
    me = np.quantile(mag, np.linspace(0, 1, 11)); me[0] -= 1e-9; me[-1] += 1e-9
    mb = np.digitize(mag, me[1:-1])
    for k in range(10):
        for g in (0, 1):
            yield (f"mag_d{k+1}xngh{g}", "cell", (mb == k) & (ngh == g))
    me5 = np.quantile(mag, np.linspace(0, 1, 6)); me5[0] -= 1e-9; me5[-1] += 1e-9
    mb5 = np.digitize(mag, me5[1:-1])
    fmax = feats["nbr_flux_max"]
    fe = np.quantile(fmax[bl], np.linspace(0, 1, 5)); fe[0] -= 1e-9; fe[-1] += 1e-9
    fb = np.digitize(fmax, fe[1:-1])
    for k in range(5):
        for j in range(4):
            yield (f"mag_q{k+1}xnfmax_q{j+1}", "cell", bl & (mb5 == k) & (fb == j))
    # random linear "arbitrary" cuts over standardized features (incl. measured obs if requested)
    cols_r = [mag, np.where(np.isfinite(dist), dist, np.nan),
              feats["nbr_flux_near"], feats["nbr_flux_far"], feats["nbr_flux_max"],
              feats["ood_flux_bright"], feats["ood_flux_faint"], ngh.astype(float)]
    if with_measured:
        cols_r += [meas[c] for c in MEAS_COLS]
    fmat = np.column_stack(cols_r)
    mu = np.nanmean(fmat, 0); sd = np.nanstd(fmat, 0); sd[sd == 0] = 1
    fmat = np.where(np.isfinite(fmat), fmat, mu)  # fill iso-distance NaN with mean (neutral)
    Xs = (fmat - mu) / sd
    rng = np.random.default_rng(seed)
    for d in range(n_dir):
        v = rng.standard_normal(Xs.shape[1]); v /= np.linalg.norm(v)
        p = Xs @ v
        lo, hi = np.quantile(p, [0.30, 0.70])
        yield (f"rand{d}_bot30", "random", p <= lo)
        yield (f"rand{d}_top30", "random", p >= hi)


def realistic_selections(df):
    """FIXED a-priori family of REALISTIC / GENTLE survey cuts on TRUE (shear-invariant)
    properties: cumulative bright magnitude cuts, contiguous true-magnitude windows
    (tomographic-like slices), (mag x blend-state) cells, and gentle environment splits.
    These are the cuts a real weak-lensing analysis actually makes -- contrast the arbitrary
    random projections in selections() -- so the acceptance target tightens from 3% to 0.3%.
    TRUE-property only => shear-invariant (R_blend-correctable); NO measured-observable cuts.
    (True-size Re windows are DEFERRED until a case40-139 Re_input_p lookup is joined -- the
    s501 per-object dump carries r_input_p only.)  Firewall: family is fixed here a priori,
    never adjusted by watching |m|."""
    mag = df.r_input_p.to_numpy(float)
    ngh = df.neighbored.to_numpy().astype(int)
    dist = df.distance.to_numpy(float)
    bl = (ngh == 1) & np.isfinite(dist)
    print("  realistic family: r_input_p (true mag) quantiles 5/25/50/75/95 = "
          f"{np.round(np.nanquantile(mag, [.05, .25, .5, .75, .95]), 3).tolist()}", flush=True)

    yield ("GLOBAL", "global", np.ones(len(mag), bool))
    # cumulative bright cuts (r brighter than X) -- the canonical survey depth cut
    for x in [24.0, 24.5, 25.0, 25.5, 26.0, 26.5]:
        yield (f"mag_lt{x:.1f}", "mag_cum", np.isfinite(mag) & (mag < x))
    # contiguous true-magnitude windows (tomographic-like slices)
    for lo, hi in [(23.0, 24.0), (24.0, 25.0), (25.0, 26.0), (26.0, 27.0),
                   (24.0, 26.0), (24.5, 26.5), (23.0, 25.0), (25.0, 27.0)]:
        yield (f"mag{lo:.1f}-{hi:.1f}", "mag_win", np.isfinite(mag) & (mag >= lo) & (mag < hi))
    # (mag window x blend state): where blending-driven residuals would surface
    for lo, hi in [(24.0, 26.0), (24.0, 25.0), (25.0, 26.0), (26.0, 27.0)]:
        mm = np.isfinite(mag) & (mag >= lo) & (mag < hi)
        yield (f"mag{lo:.0f}-{hi:.0f}_iso",   "mag_x_blend", mm & (ngh == 0))
        yield (f"mag{lo:.0f}-{hi:.0f}_blend", "mag_x_blend", mm & (ngh == 1))
    # gentle environment splits
    yield ("isolated", "env", ngh == 0)
    yield ("blended",  "env", ngh == 1)
    dmed = float(np.nanmedian(dist[bl]))
    yield ("blend_close", "env", bl & (dist < dmed))
    yield ("blend_wide",  "env", bl & (dist >= dmed))
    # --- contiguous true-SIZE (Re_input_p) windows + mag x size cells (if the size lookup joined) ---
    if "Re_input_p" in df.columns:
        size = df.Re_input_p.to_numpy(float)
        print("  realistic family: Re_input_p (true size) q5/50/95 = "
              f"{np.round(np.nanquantile(size, [.05, .5, .95]), 3).tolist()}", flush=True)
        for lo, hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (0.5, 1.5)]:
            yield (f"size{lo:.1f}-{hi:.1f}", "size_win", np.isfinite(size) & (size >= lo) & (size < hi))
        for x in [0.3, 0.5, 1.0]:
            yield (f"size_gt{x:.1f}", "size_cum", np.isfinite(size) & (size >= x))
        for (mlo, mhi) in [(24.0, 26.0), (25.0, 26.0)]:
            mm = np.isfinite(mag) & (mag >= mlo) & (mag < mhi)
            yield (f"mag{mlo:.0f}-{mhi:.0f}_sizeGt0.5", "mag_x_size", mm & np.isfinite(size) & (size >= 0.5))
            yield (f"mag{mlo:.0f}-{mhi:.0f}_sizeLt0.3", "mag_x_size", mm & np.isfinite(size) & (size < 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--rflow-override", default=None)
    ap.add_argument("--rblend-override", default=None)
    ap.add_argument("--n-dir", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=50000)
    ap.add_argument("--min-cases", type=int, default=30)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.03)
    ap.add_argument("--with-measured", action="store_true",
                    help="add measured-observable selections (realistic shear-dependent survey cuts)")
    ap.add_argument("--realistic", action="store_true",
                    help="use the REALISTIC/gentle survey-cut family (contiguous true-mag windows + "
                         "mag x blend cells + env splits) instead of the arbitrary-projection family; "
                         "pair with --target 0.003 for the 0.3%% acceptance question")
    ap.add_argument("--cases-from-override", action="store_true",
                    help="restrict the dump to the cases present in --rflow-override before evaluating "
                         "(use when the override covers a case subset, e.g. an OOS eval range, so certified "
                         "fallback values on other cases don't contaminate the table)")
    ap.add_argument("--resp-floor", type=float, default=0.05,
                    help="mean-total-response floor below which m is treated as degenerate (a priori)")
    args = ap.parse_args()

    log("loading dump + joins")
    df = load(args.dump)
    if args.with_measured:
        df = join_measured(df)
    log(f"loaded {len(df):,} rows, cases {int(df.case.min())}-{int(df.case.max())}")
    if args.cases_from_override and args.rflow_override:
        ovcases = np.unique(np.load(args.rflow_override)["case"].astype(np.int64))
        df = df[df.case.isin(ovcases)].reset_index(drop=True)
        log(f"restricted dump to {len(ovcases)} override cases {ovcases.min()}-{ovcases.max()}: {len(df):,} rows")
    if args.rflow_override:
        df = apply_override(df, args.rflow_override, "R_flow")
    if args.rblend_override:
        df = apply_override(df, args.rblend_override, "R_blend")

    rsim = df.r_sim.to_numpy(float); rflow = df.R_flow.to_numpy(float); rbl = df.R_blend.to_numpy(float)
    case = df.case.to_numpy(np.int64)
    ucase, cidx = np.unique(case, return_inverse=True); ncase = len(ucase)
    rng = np.random.default_rng(7)
    pick = rng.integers(0, ncase, size=(args.nboot, ncase))

    rows = []
    n_sel = n_skip = 0
    fam = (realistic_selections(df) if args.realistic
           else selections(df, n_dir=args.n_dir, with_measured=args.with_measured))
    for label, kind, mask in fam:
        n = int(mask.sum())
        if n < args.min_n:
            n_skip += 1; continue
        ci = cidx[mask]
        sc_s = np.bincount(ci, weights=rsim[mask], minlength=ncase)
        sc_f = np.bincount(ci, weights=rflow[mask], minlength=ncase)
        sc_b = np.bincount(ci, weights=rbl[mask], minlength=ncase)
        sc_n = np.bincount(ci, minlength=ncase)
        if int((sc_n > 0).sum()) < args.min_cases:
            n_skip += 1; continue
        denom = sc_f.sum() + sc_b.sum()
        m = sc_s.sum() / denom - 1 if denom else np.nan
        bs = sc_s[pick].sum(1); bf = sc_f[pick].sum(1); bb = sc_b[pick].sum(1)
        bm = bs / (bf + bb) - 1
        sem = float(bm.std())
        rows.append(dict(label=label, kind=kind, n=n, m=float(m), merr=sem,
                         z=float(abs(m) / sem) if sem > 0 else np.nan,
                         resp=float(denom / n)))   # mean total response per object (m ill-conditioned as ->0)
        n_sel += 1
    log(f"evaluated {n_sel} selections ({n_skip} skipped for n<{args.min_n} or <{args.min_cases} cases)")

    R = pd.DataFrame(rows)
    Rs = R.reindex(R.m.abs().sort_values(ascending=False).index)
    sig = R[(R.z >= 3) & (R.label != "GLOBAL")]
    sig_s = sig.reindex(sig.m.abs().sort_values(ascending=False).index)

    tgt = args.target
    emit("=" * 100)
    emit(f"SELECTION-ROBUSTNESS ACCEPTANCE HARNESS  [tag={args.tag}]  target |m| <= {100*tgt:g}%")
    emit(f"  R_flow override : {args.rflow_override or '(dump: certified)'}")
    emit(f"  R_blend override: {args.rblend_override or '(dump: certified density lookup)'}")
    emit(f"  dump={args.dump.split('/')[-1]}  ncase={ncase}  n_selections={n_sel}")
    emit("=" * 100)
    gm = R[R.label == "GLOBAL"].iloc[0]
    emit(f"GLOBAL m = {100*gm.m:+.3f}% +/- {100*gm.merr:.3f}%   (sanity vs certified +0.245%)")
    emit("")
    emit(f"WORST-CASE |m| over the family        : {100*Rs.iloc[0].m:+.2f}%  ({Rs.iloc[0].label}, n={int(Rs.iloc[0].n):,})")
    if len(sig_s):
        emit(f"WORST-CASE |m| among SIGNIFICANT (z>=3): {100*sig_s.iloc[0].m:+.2f}%  ({sig_s.iloc[0].label})")
    # denominator guard: m is ill-conditioned when the selected set's mean total response -> 0.
    # Report worst-case among NON-DEGENERATE selections (mean response above an a-priori floor).
    rf = args.resp_floor
    nd = R[R.resp.abs() > rf]
    if len(nd):
        nd_s = nd.reindex(nd.m.abs().sort_values(ascending=False).index)
        ndz = nd[(nd.z >= 3) & (nd.label != "GLOBAL")]
        ndz_s = ndz.reindex(ndz.m.abs().sort_values(ascending=False).index)
        emit(f"WORST-CASE |m| NON-DEGENERATE (mean resp>{rf}, {len(nd)}/{n_sel} sel): "
             f"{100*nd_s.iloc[0].m:+.2f}%  ({nd_s.iloc[0].label}, resp={nd_s.iloc[0].resp:.3f})"
             + (f"   | significant: {100*ndz_s.iloc[0].m:+.2f}% ({ndz_s.iloc[0].label})" if len(ndz_s) else ""))
    absm = R.m.abs().to_numpy()
    emit(f"|m| distribution %%: median={100*np.median(absm):.2f}  90th={100*np.quantile(absm,0.9):.2f}  "
         f"99th={100*np.quantile(absm,0.99):.2f}  max={100*absm.max():.2f}")
    frac_ok = float((absm <= tgt).mean())
    frac_ok_sig = float(((absm <= tgt) | (R.z.to_numpy() < 3)).mean())
    emit(f"fraction of selections with |m| <= {100*tgt:g}%%: {100*frac_ok:.1f}%%  "
         f"(counting only z>=3 as failures: {100*frac_ok_sig:.1f}%%)")
    emit("")
    emit("by KIND: worst |m| and median |m|")
    for k, g in R.groupby("kind"):
        am = g.m.abs()
        emit(f"   {k:9s}: n_sel={len(g):4d}  worst={100*am.max():6.2f}%  median={100*am.median():5.2f}%  "
             f"({g.loc[am.idxmax(),'label']})")
    emit("")
    emit("15 WORST selections (by |m|):")
    emit(f"   {'label':28s} {'kind':8s} {'n':>10s} {'m%':>8s} {'sem%':>7s} {'z':>6s}")
    for _, r in Rs.head(15).iterrows():
        emit(f"   {r.label:28s} {r.kind:8s} {int(r.n):>10,d} {100*r.m:>+8.2f} {100*r.merr:>7.2f} {r.z:>6.1f}")

    outnpz = OUTNPZ.format(tag=args.tag); outtxt = OUTTXT.format(tag=args.tag)
    R.to_json(outtxt.replace(".txt", "_rows.json"), orient="records")
    np.savez(outnpz, labels=R.label.to_numpy(), kinds=R.kind.to_numpy(),
             n=R.n.to_numpy(), m=R.m.to_numpy(), sem=R.merr.to_numpy(), z=R.z.to_numpy(),
             global_m=float(gm.m), worst_abs_m=float(absm.max()),
             worst_sig_m=float(sig_s.iloc[0].m) if len(sig_s) else 0.0,
             frac_ok=frac_ok, target=tgt)
    open(outtxt, "w").write("\n".join(_o) + "\n")
    log(f"wrote {outtxt} and {outnpz}")


if __name__ == "__main__":
    main()
