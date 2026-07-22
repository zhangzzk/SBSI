"""[SUPERSEDED — not the deliverable; see SBSI/STATE_OF_PLAY.md] This pairs-only direct-regression
R_flow track (self_response_catalogue = 0% isolated) is a NON-CIRCULAR cross-check, kept for the
record. Its cont.68-73 "sub-percent NOT feasible" verdict was WITHDRAWN (cont.74): the canonical
winner is the production-recipe joint R_flow (_prod, isolated-inclusive), GLOBAL +0.074% / realistic
median 0.76%. Do NOT base conclusions on this script's floor — it lacks the isolated self-anchor.

HONEST feasibility test: does a per-pair blend model (trained on the half-shear BLEND sim),
summed COHERENTLY over each constgold object's aperture neighbours, close the faint-end deficit?

  R_sim(constgold)  ?=  R_flow(shape self, from SELF sim)  +  sum_j R_blend_pair(O,N_j)

- R_flow  : <delta_et1>/0.05 grid over (flux x size) from self_response_catalogue (SELF sim).
- R_bpair : HistGBR on [dmag=r_s-r_p, log10 distance, Re_p, Re_s, r_p] -> delta_et1/0.2, from
            response_catalogue (BLEND sim). This is the per-NEIGHBOUR leakage coefficient.
- scene   : for each detected constgold object, KDTree all input galaxies within r_max arcsec,
            predict R_bpair per neighbour, SUM (linearity of E[leakage] over neighbours).
- r_sim   : constgold cert per-object antithetic response (VALIDATION ONLY; never fit).

Everything trained OOS-by-case (train cases disjoint from the constgold eval cases). Constgold
supplies only object coordinates + neighbour geometry, never its measured response, to R_flow/R_blend.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS as CUTS, source_select_selection  # noqa

SIM = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/"
CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"


def read_cases(path, cols, cases, maxb=100000):
    rows = []
    cmax = max(cases)
    got = False
    with ipc.open_file(path) as r:
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            bmin = int(b.case.min())
            sel = b[b.case.isin(cases)]
            if len(sel):
                rows.append(sel); got = True
            if got and bmin > cmax:      # cases stored in order -> stop once past the requested range
                break
            if bi > maxb:
                break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cases", default="0-19")
    ap.add_argument("--eval-cases", default="40-49")
    ap.add_argument("--r-max", type=float, default=10.0, help="neighbour aperture (arcsec)")
    ap.add_argument("--flow-model", choices=["grid", "smooth"], default="grid",
                    help="grid=6x10 flux-quantile x size mean; smooth=HistGBR(flux,size) on the self sim")
    ap.add_argument("--blend-baseline-sub", action="store_true",
                    help="enforce leakage->0 at the aperture edge: subtract the blend-sim far-pair (8-10\") "
                         "prediction floor from every per-pair prediction (clip>=0). a-priori, blend-sim-only.")
    ap.add_argument("--n-ensemble", type=int, default=1,
                    help="bag N HistGBR members (bootstrap-resampled rows, distinct seeds) for R_flow(smooth) "
                         "and R_blend, average their predictions. 1 = single model. Separates model-fit "
                         "variance (which a case-bootstrap z cannot see) from true bias.")
    ap.add_argument("--harvest-flow", default=None, help="write per-object R_flow override npz")
    ap.add_argument("--harvest-blend", default=None, help="write per-object R_blend_scene override npz")
    args = ap.parse_args()
    def parse(s):
        # comma-separated ranges, e.g. "0-39,140-199" -> [0..39, 140..199]
        out = []
        for part in s.split(","):
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        return sorted(out)
    TR, EV = parse(args.train_cases), parse(args.eval_cases)
    print(f"train(blend/self)={args.train_cases} ({len(TR)} cases)  eval(constgold)={EV[0]}-{EV[-1]}  r_max={args.r_max}\"")

    def read_sharded(base, cols, cases):
        """Read cases from the 0_99 and/or 100_199 shards (self/blend catalogues span 0-199)."""
        lo = [c for c in cases if c < 100]; hi = [c for c in cases if c >= 100]
        parts = []
        if lo:
            parts.append(read_cases(base.format("0_99"), cols, lo))
        if hi:
            parts.append(read_cases(base.format("100_199"), cols, hi))
        return pd.concat(parts, ignore_index=True) if parts else parts[0]

    # ---- R_flow: self-response grid (flux x size) ----
    sc = read_sharded(SIM + "self_response_catalogue_train_cases{}.feather",
                    ["case", "input_index", "delta_et1", "r_input_p", "Re_input_p", "distance", "neighbored"], TR)
    sc = source_select_selection(sc, cuts=CUTS)
    sc["Rself"] = sc.delta_et1 / 0.05
    fe = np.quantile(sc.r_input_p, np.linspace(0, 1, 7)); fe[0] -= 1e-6; fe[-1] += 1e-6
    se = np.array([0.0999, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5001])
    def cell2d(f, s):
        return (np.clip(np.digitize(f, fe) - 1, 0, len(fe) - 2),
                np.clip(np.digitize(s, se) - 1, 0, len(se) - 2))
    fi, si = cell2d(sc.r_input_p.values, sc.Re_input_p.values)
    RG = np.full((len(fe) - 1, len(se) - 1), sc.Rself.mean())
    for a in range(RG.shape[0]):
        for b in range(RG.shape[1]):
            m = (fi == a) & (si == b)
            if m.sum() > 100:
                RG[a, b] = sc.Rself.values[m].mean()
    def bag_fit(X, y, n, mk):
        """Bag n members: member 0 on the full data, members 1..n-1 on bootstrap resamples w/ distinct seeds."""
        out = []
        for k in range(max(1, n)):
            idx = np.arange(len(y)) if k == 0 else np.random.RandomState(1000 + k).randint(0, len(y), len(y))
            reg_ = mk(k); reg_.fit(X[idx], y[idx]); out.append(reg_)
        return out
    flow_regs = None
    if args.flow_model == "smooth":
        # continuous self-response in (flux, size); removes flux-quantile bin-edge steps that
        # alias onto magnitude-decile selections. Trained on the self sim only (OOS by case).
        flow_regs = bag_fit(np.column_stack([sc.r_input_p.values, sc.Re_input_p.values]), sc.Rself.values,
                            args.n_ensemble,
                            lambda k: HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=31,
                                learning_rate=0.05, min_samples_leaf=500, l2_regularization=1.0, random_state=k))
    def flow_predict(rp_, rep_):
        if flow_regs is not None:
            X = np.column_stack([rp_, rep_])
            return np.mean([r.predict(X) for r in flow_regs], axis=0)
        a_, b_ = cell2d(rp_, rep_)
        return RG[a_, b_]
    print(f"self {args.flow_model} model built on {len(sc):,} objs "
          f"({args.n_ensemble if flow_regs else 1} member(s)), global Rself={sc.Rself.mean():.4f}")

    # ---- R_blend_pair: HistGBR on the blend sim ----
    bl = read_sharded(SIM + "response_catalogue_train_cases{}.feather",
                    ["case", "input_index", "delta_et1", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "distance"], TR)
    bl = bl[(bl.r_input_p > CUTS[1][0]) & (bl.r_input_p < CUTS[1][1]) &
            (bl.Re_input_p > CUTS[3][0]) & (bl.Re_input_p < CUTS[3][1])]
    yb = (bl.delta_et1 / 0.2).values
    def feats(rp, rs, dist, rep, res):
        return np.column_stack([rs - rp, np.log10(np.clip(dist, 1e-3, None)), rep, res, rp])
    Xb = feats(bl.r_input_p.values, bl.r_input_s.values, bl.distance.values, bl.Re_input_p.values, bl.Re_input_s.values)
    blend_regs = bag_fit(Xb, yb, args.n_ensemble,
                         lambda k: HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63,
                             learning_rate=0.06, min_samples_leaf=200, l2_regularization=1.0, random_state=k))
    def blend_predict(X):
        return np.mean([r.predict(X) for r in blend_regs], axis=0)
    # far-pair prediction floor (physical BC: a neighbour at the aperture edge gives ~0 shear leakage;
    # the small residual is a sim/measurement floor that spuriously SUMS over many far neighbours).
    faredge = (bl.distance.values >= 8.0) & (bl.distance.values <= 10.0)
    b0 = float(blend_predict(Xb[faredge]).mean()) if faredge.sum() > 1000 else 0.0
    if not args.blend_baseline_sub:
        b0 = 0.0
    print(f"blend per-pair model trained on {len(bl):,} pairs ({len(blend_regs)} member(s), dmag range "
          f"[{(bl.r_input_s-bl.r_input_p).min():.1f},{(bl.r_input_s-bl.r_input_p).max():.1f}]); "
          f"far-pair floor b0={b0:.5f} (baseline_sub={args.blend_baseline_sub})")

    # ---- constgold cert: read ONCE for all eval cases (NO selection cut: harvest EVERY detected
    #      object so the override covers the whole dump; the acceptance harness applies selections) ----
    CERT = read_cases(CDIR + "constant_response_catalogue_c40-139.feather",
                      ["case", "input_index", "response", "r_input_p", "Re_input_p", "distance", "neighbored"], EV)
    print(f"constgold cert (all eval cases): {len(CERT):,} detected objs")

    # ---- apply on constgold eval scene ----
    rmax_deg = args.r_max / 3600.0
    allrows = []
    for c in EV:
        g = glob.glob(CDIR + f"gals{c}_0.02.feather")
        if not g:
            continue
        gg = pf.read_table(g[0], columns=["index", "RA", "DEC", "Re", "r"]).to_pandas()
        cd = np.cos(np.deg2rad(gg.DEC.values))
        XY = np.column_stack([gg.RA.values * cd, gg.DEC.values])  # approx tangent-plane deg
        tree = cKDTree(XY)
        cert = CERT[CERT.case == c].reset_index(drop=True)
        if len(cert) == 0:
            continue
        # map each cert object to its input row
        idx_of = pd.Series(np.arange(len(gg)), index=gg["index"].values)
        oi = idx_of.reindex(cert.input_index.values).values
        ok = np.isfinite(oi); oi = oi[ok].astype(int); cert = cert[ok].reset_index(drop=True)
        rp = gg.r.values[oi]; rep = gg.Re.values[oi]
        RA = gg.RA.values; DEC = gg.DEC.values; RR = gg.r.values; RE = gg.Re.values
        # neighbour sum: flatten ALL (object,neighbour) pairs, predict ONCE, scatter-add per object
        nb = tree.query_ball_point(XY[oi], rmax_deg)
        lens = np.fromiter((len(l) for l in nb), int, len(nb))
        obj_ptr = np.repeat(np.arange(len(oi)), lens)
        nbr_j = np.concatenate([np.asarray(l, int) for l in nb]) if lens.sum() else np.zeros(0, int)
        keep = nbr_j != oi[obj_ptr]                      # drop the self match
        obj_ptr, nbr_j = obj_ptr[keep], nbr_j[keep]
        o_ra = RA[oi[obj_ptr]]; o_dec = DEC[oi[obj_ptr]]; o_cd = cd[oi[obj_ptr]]
        dist = np.hypot((RA[nbr_j] - o_ra) * o_cd, DEC[nbr_j] - o_dec) * 3600.0
        X = feats(rp[obj_ptr], RR[nbr_j], dist, rep[obj_ptr], RE[nbr_j])
        # NO clip: per-pair tangential leakage delta_et1/0.2 can legitimately be negative and the
        # coherent neighbour SUM needs those negatives (clipping at 0 inflates R_blend on faint objects,
        # which have many near-zero/slightly-negative faint neighbours). b0=0 unless --blend-baseline-sub.
        pred = blend_predict(X) - b0
        Rb = np.zeros(len(oi))
        np.add.at(Rb, obj_ptr, pred)
        Rb_far = np.zeros(len(oi))                        # diagnostic: contribution from >4" neighbours
        np.add.at(Rb_far, obj_ptr, np.where(dist >= 4.0, pred, 0.0))
        Rf = flow_predict(rp, rep)
        allrows.append(pd.DataFrame(dict(case=c, input_index=cert.input_index.values,
                                         r_input_p=rp, Re_input_p=rep,
                                         r_sim=cert.response.values, R_flow=Rf, R_blend=Rb, R_blend_far=Rb_far,
                                         neighbored=cert.neighbored.values)))
        print(f"  case {c}: {len(oi):,} objs, <r_sim>={cert.response.mean():.3f} <Rf>={Rf.mean():.3f} <Rb>={Rb.mean():.3f}", flush=True)
    df = pd.concat(allrows, ignore_index=True)
    if args.harvest_flow:
        np.savez(args.harvest_flow, case=df.case.values.astype(np.int64),
                 input_index=df.input_index.values.astype(np.int64), value=df.R_flow.values.astype(float))
        print(f"wrote R_flow override {args.harvest_flow} ({len(df):,} objs)")
    if args.harvest_blend:
        np.savez(args.harvest_blend, case=df.case.values.astype(np.int64),
                 input_index=df.input_index.values.astype(np.int64), value=df.R_blend.values.astype(float))
        print(f"wrote R_blend override {args.harvest_blend} ({len(df):,} objs)")

    def m(mask):
        s = df[mask]
        if len(s) < 50:
            return None
        Ss, Sf, Sb = s.r_sim.sum(), s.R_flow.sum(), s.R_blend.sum()
        return dict(n=len(s), m=100 * (Ss / (Sf + Sb) - 1), rsim=s.r_sim.mean(), Rf=s.R_flow.mean(),
                    Rb=s.R_blend.mean(), Rbfar=s.R_blend_far.mean(), pred=(Sf + Sb) / len(s))
    print("\nCLOSURE  m = <r_sim>/(<R_flow>+<R_blend_scene>) - 1")
    print(f"{'sel':16s} {'n':>7s} {'m%':>8s} | {'rsim':>6s} {'Rf':>6s} {'Rb':>6s} {'Rbfar':>6s} {'pred':>6s}")
    mag = df.r_input_p.values; nb = df.neighbored.values.astype(bool)
    sels = [("ALL", np.ones(len(df), bool)), ("ISOLATED", ~nb), ("BLENDED", nb),
            ("mag<24", mag < 24), ("mag24-25", (mag >= 24) & (mag < 25)),
            ("mag25-26", (mag >= 25) & (mag < 26)), ("mag26-27", (mag >= 26) & (mag < 27))]
    for lbl, mk in sels:
        r = m(mk)
        if r:
            print(f"{lbl:16s} {r['n']:7d} {r['m']:+8.2f} | {r['rsim']:6.3f} {r['Rf']:6.3f} {r['Rb']:6.3f} {r['Rbfar']:6.3f} {r['pred']:6.3f}")


if __name__ == "__main__":
    main()
