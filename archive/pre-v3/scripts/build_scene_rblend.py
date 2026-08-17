#!/usr/bin/env python -B
"""Build a SEPARATE scene-conditioned R_blend, out-of-sample by case (honest CV).

The parameter-free estimator is  m = R_sim/(R_flow + R_blend) - 1  with R_blend a SEPARATE
additive term (user's constraint: R_blend stays separate from the flow).  R_blend must be the
conditional mean of the blend response  (r_sim - R_flow)  given the physical SCENE features.
cont.56 showed the certified density-keyed R_blend omits the joint neighbour-FLUX structure,
which is what re-exposes non-closure under flux/environment selections.

This produces an upgraded scene R_blend = E[ r_sim - R_flow | phi ], phi FIXED a priori:
    phi = [r_input_p, neighbored, distance, nbr_flux_near, nbr_flux_far, nbr_flux_max,
           ood_flux_bright, ood_flux_faint]
Trained with K-fold cross-validation BY CASE (no case leaks train->test), so every object gets
an OUT-OF-SAMPLE R_blend prediction -> the downstream m eval is not in-sample-optimistic.

--rflow-override lets the target use the joint-flow R_flow (self-consistent decomposition) so
the pair (joint R_flow, scene R_blend) is what eval_selection_robustness.py scores.  Default =
the certified dump R_flow (upgrades ONLY R_blend, holding R_flow fixed).

Features are fixed a priori; the model is trained on the physical response target, never on |m|
(firewall).  This is the diagnostic/feasibility R_blend; a production R_blend would train on
independent blend sims rather than OOS-CV on the cert population -- noted, not hidden.
"""
import argparse, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
from sklearn.ensemble import HistGradientBoostingRegressor

DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
CB = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
PHI = ["r_input_p", "neighbored", "distance", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
       "ood_flux_bright", "ood_flux_faint"]

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def load(extra_size=False):
    cols = ["case", "input_index", "r_input_p", "r_sim", "R_flow", "neighbored", "distance"]
    parts = {c: [] for c in cols}
    with pa.memory_map(DUMP, "r") as src:
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
    if extra_size:  # opt-in: add true SIZE so R_blend can model the size-dependent selection response
        sz = pf.read_table(CB + "true_size_lookup_c40-139.feather", memory_map=True).to_pandas()[
            ["case", "input_index", "Re_input_p"]]
        df = df.merge(sz, on=["case", "input_index"], how="left"); del sz
        log(f"extra-size: joined Re_input_p, missing={int(df.Re_input_p.isna().sum()):,}/{len(df):,} "
            f"(NaN routed natively by HistGB)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rflow-override", default=None, help="npz(case,input_index,value) R_flow for the target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--train-sub", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-size", action="store_true",
                    help="append true Re_input_p to phi so R_blend models the size-dependent selection "
                         "response (fixes size-cut non-closure). Opt-in; default phi is unchanged.")
    # GB capacity knobs (defaults reproduce the certified fit byte-for-byte). Raising capacity /
    # lowering regularization gives the sparse size/mag TAILS their own leaves instead of shrinking
    # them toward the bulk -- the mechanism behind the residual size_gt1.0 non-closure.
    ap.add_argument("--gb-iter", type=int, default=400)
    ap.add_argument("--gb-leaves", type=int, default=127)
    ap.add_argument("--gb-min-leaf", type=int, default=200)
    ap.add_argument("--gb-l2", type=float, default=1.0)
    args = ap.parse_args()

    phi = PHI + (["Re_input_p"] if args.extra_size else [])
    log(f"loading dump + joins  (phi={phi})")
    df = load(extra_size=args.extra_size)
    rflow = df.R_flow.to_numpy(float)
    if args.rflow_override:
        z = np.load(args.rflow_override)
        ov = pd.DataFrame({"case": z["case"].astype(np.int64), "input_index": z["input_index"].astype(np.int64),
                           "_ov": z["value"].astype(float)})
        m = df.merge(ov, on=["case", "input_index"], how="left")
        miss = int(m["_ov"].isna().sum())
        log(f"R_flow override matched {len(df)-miss:,}/{len(df):,} ({100*(len(df)-miss)/len(df):.1f}%)")
        rflow = np.where(m["_ov"].notna(), m["_ov"], rflow)
    target = df.r_sim.to_numpy(float) - rflow
    log(f"target=(r_sim - R_flow): mean={target.mean():+.4f} std={target.std():.4f}")

    X = df[phi].to_numpy(float)
    di = phi.index("distance")
    X[:, di] = np.where(np.isfinite(X[:, di]), X[:, di], -1.0)   # isolated: sentinel distance

    case = df.case.to_numpy(np.int64)
    ucase = np.unique(case)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ucase))
    fold_of_case = {int(ucase[perm[i]]): i % args.nfolds for i in range(len(ucase))}
    fold = np.array([fold_of_case[int(c)] for c in case], dtype=int)

    log(f"GB config: iter={args.gb_iter} leaves={args.gb_leaves} min_leaf={args.gb_min_leaf} "
        f"l2={args.gb_l2} train_sub={args.train_sub:,} nfolds={args.nfolds}")
    oos = np.full(len(df), np.nan)
    for f in range(args.nfolds):
        te = fold == f; tr = ~te
        tr_idx = np.where(tr)[0]
        sub = rng.choice(tr_idx, size=min(args.train_sub, tr_idx.size), replace=False)
        gb = HistGradientBoostingRegressor(max_iter=args.gb_iter, learning_rate=0.05,
                                           max_leaf_nodes=args.gb_leaves,
                                           min_samples_leaf=args.gb_min_leaf, l2_regularization=args.gb_l2,
                                           early_stopping=True, validation_fraction=0.1, random_state=0)
        gb.fit(X[sub], target[sub])
        oos[te] = gb.predict(X[te])
        log(f"fold {f}: train_sub={sub.size:,} test={int(te.sum()):,}  "
            f"test<R_blend>={oos[te].mean():+.4f} vs <target>={target[te].mean():+.4f}")
    assert np.isfinite(oos).all(), "some objects got no OOS prediction"
    log(f"OOS scene R_blend: mean={oos.mean():+.4f} std={oos.std():.4f}  "
        f"(old density R_blend mean={df.R_flow.to_numpy().mean()*0:+.4f} n/a)")

    np.savez(args.out, case=case, input_index=df.input_index.to_numpy(np.int64), value=oos)
    log(f"wrote {args.out}")
    print("BUILD_SCENE_RBLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
