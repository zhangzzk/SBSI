"""Is the in-domain m just a POPULATION-WEIGHTING mismatch between the pin and the metric?

THE HYPOTHESIS, AND WHY IT IS THE PRE-REGISTERED ONE
----------------------------------------------------
The response pin in `train_measurement_model_swa_s1_truecond.py::epoch_response` minimises

    resp = sum_b  cnt_b * (Rmodel_b - Rsim_b)^2  /  sum_b cnt_b

where `cnt_b` is the count of TRAINING rows in response-target cell b. The trainer's own comment
(line ~484) states the consequence out loud: absolute error "drives global m~0 only for the TRAINING
population weighting and leaves a crowding tilt that survives any population reweighting
(constant-gold, survey depth)". WORKLOG 2026-07-2x line 1894 then names the fix as the recommended
next framework: "importance-reweight training to the constgold TRUE (mag,size) distribution to
attack the population mismatch directly (firewall-clean: uses the sample's property distribution,
not r_sim)".

That has never been measured. It is measurable with ZERO training, because m is a ratio of means and
we already have every per-object ingredient on disk:

    m(w) = sum_i w_i r_sim_i / ( sum_i w_i R_flow_i + sum_i w_i R_blend_i ) - 1

Evaluate the SAME dumps under two weightings:
  * w = 1                    -> the constgold population. Must reproduce the published m.
  * w = cnt_train(cell(i)) / n_cg(cell(i))   -> reweights constgold to the TRAINING population's
                                               cell occupancy, cell-for-cell on the target's own grid.

If m goes to ~0 under the training weighting, the residual IS the weighting mismatch and importance
reweighting the pin is the fix. If m stays at -0.5%, the mismatch is NOT the mechanism and the
reweighting framework should not be built -- that negative result is just as valuable, and much
cheaper than discovering it after 8 training runs.

The reverse direction is what the fix would actually do, so it is reported too: the per-cell pin
residual dR_b = <R_flow + R_blend - r_sim>_b is a property of the trained model, and

    predicted m after reweighting ~ -sum_b w_b^cg dR_b / (<R_flow> + <R_blend>)

is already the measured m -- the point of the comparison is the size of the gap between the two
weightings, which bounds how much reweighting can possibly buy.

FIREWALL. Reads constgold per-object dumps (evaluation data) and a response-target npz built from
the half-shear det_meas catalogue. Trains nothing, selects nothing. Note that the *fix* this
motivates uses only the constgold TRUE-PROPERTY OCCUPANCY (a property of the input catalogue and of
the cut definition, not of any measured shear response), which is the sense in which WORKLOG 1894
calls it firewall-clean. The r_sim column is used here ONLY to report the diagnostic, never to pick
a cell weighting.
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
TARGET = ("/home/z/Zekang.Zhang/SBSI/results/"
          "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz")


def seed_of(p):
    mo = re.search(r"_s(\d+)(?:_|\.)", os.path.basename(p))
    return int(mo.group(1)) if mo else -1


def cell_ids(mag, size, crowd, ef, es, ec):
    """EXACTLY the trainer's _bin_id: digitize-1, clipped into range, row-major (fi*ns+si)*nb+di."""
    nf, ns, nb = len(ef) - 1, len(es) - 1, len(ec) - 1
    fi = np.clip(np.digitize(mag, ef) - 1, 0, nf - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
    di = np.clip(np.digitize(crowd, ec) - 1, 0, nb - 1)
    return ((fi * ns + si) * nb + di).astype(np.int64), (nf, ns, nb)


def m_of(w, rs, rf, rb):
    """m under an arbitrary per-object weighting. Exact, no binning involved."""
    sw = w.sum()
    return 100.0 * ((w * rs).sum() / sw) / (((w * rf).sum() / sw) + ((w * rb).sum() / sw)) - 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DOMDUMPS)
    ap.add_argument("--tag", default="ablate_s2c_lt500_dom6x6")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--catalogue", default=CONSTCAT)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--crowd-from", default="R_blend",
                    help="dump column standing in for the target's crowd_col (r_blend)")
    ap.add_argument("--save-weights", default=None,
                    help="write the DELIVERABLE population's per-cell occupancy to this npz as "
                         "`pop_w`, for --response-pop-weight-npz in the trainer's global anchor. "
                         "Occupancy only: no response, no m, so training stays firewall-clean.")
    args = ap.parse_args()

    z = np.load(args.target, allow_pickle=True)
    ef, es, ec = z["edges_flux"], z["edges_size"], z["edges_crowd"]
    cnt_tr = np.asarray(z["counts"], dtype=float).reshape(-1)
    Rsim_tgt = np.asarray(z["Rsim"], dtype=float).reshape(-1)
    print(f"target {os.path.basename(args.target)}  crowd_col={z['crowd_col'].item()}  "
          f"grid {z['Rsim'].shape}  training rows {cnt_tr.sum():,.0f}")

    truth = pf.read_table(args.catalogue,
                          columns=["case", "input_index", "r_input_p", "Re_input_p"],
                          memory_map=True).to_pandas().drop_duplicates(["case", "input_index"])
    sel = ((truth["r_input_p"].to_numpy(float) < args.true_mag_max)
           & (truth["Re_input_p"].to_numpy(float) > args.true_re_min))
    # take ONLY Re_input_p from the catalogue -- the dump already carries r_input_p, and merging
    # both would collide into r_input_p_x/_y
    keep = truth[sel][["case", "input_index", "Re_input_p"]]
    print(f"in-domain cut: {len(truth):,} -> {len(keep):,} ({100 * len(keep) / len(truth):.1f}%)")

    paths = sorted(glob.glob(os.path.join(args.dump_dir, f"{args.tag}_perobj_s*.feather")))
    if not paths:
        raise SystemExit(f"no dumps for {args.tag} in {args.dump_dir}")
    print(f"{len(paths)} dumps, seeds {[seed_of(p) for p in paths]}\n")

    rows = []
    cell_acc = None
    n_cg_ref = None
    for p in paths:
        d = pf.read_table(p, memory_map=True).to_pandas().merge(
            keep, on=["case", "input_index"], how="inner")
        cid, (nf, ns, nb) = cell_ids(d["r_input_p"].to_numpy(float),
                                     d["Re_input_p"].to_numpy(float),
                                     d[args.crowd_from].to_numpy(float), ef, es, ec)
        ncell = nf * ns * nb
        rs = d["r_sim"].to_numpy(float)
        rf = d["R_flow"].to_numpy(float)
        rb = d["R_blend"].to_numpy(float)

        n_cg = np.bincount(cid, minlength=ncell).astype(float)
        # importance weight: make constgold's cell occupancy look like the TRAINING population's.
        # Cells the training population never occupied get weight 0 (the pin never constrained them).
        with np.errstate(divide="ignore", invalid="ignore"):
            per_cell_w = np.where(n_cg > 0, cnt_tr / np.maximum(n_cg, 1.0), 0.0)
        w_tr = per_cell_w[cid]
        w_1 = np.ones_like(rs)

        m_cg = m_of(w_1, rs, rf, rb)
        m_train = m_of(w_tr, rs, rf, rb)
        rows.append(dict(seed=seed_of(p), m_cg=m_cg, m_train=m_train,
                         Rf=rf.mean(), Rb=rb.mean(), Rs=rs.mean(),
                         cover=100.0 * (w_tr > 0).mean()))
        acc = np.stack([np.bincount(cid, weights=x, minlength=ncell) for x in (rs, rf, rb)])
        cell_acc = acc if cell_acc is None else cell_acc + acc
        if n_cg_ref is None:      # identical rows every seed, so occupancy is seed-independent
            n_cg_ref = n_cg
        del d, cid, rs, rf, rb

    t = pd.DataFrame(rows)
    print("per-seed m (%) under the two weightings")
    print(f"{'seed':>6} {'m constgold':>12} {'m TRAINING-wt':>14} {'shift':>8}")
    for _, r in t.iterrows():
        print(f"{int(r['seed']):>6} {r['m_cg']:>11.3f}% {r['m_train']:>13.3f}% "
              f"{r['m_train'] - r['m_cg']:>+7.3f}")
    n = len(t)
    mc, mt = t["m_cg"].mean(), t["m_train"].mean()
    sc, st = t["m_cg"].std(ddof=1), t["m_train"].std(ddof=1)
    dpair = (t["m_train"] - t["m_cg"])
    print("\n" + "=" * 76)
    print(f"m under the CONSTGOLD population (the published number) = {mc:+.3f}%  "
          f"(sd {sc:.3f}, sem {sc / np.sqrt(n):.3f})")
    print(f"m under the TRAINING population weighting               = {mt:+.3f}%  "
          f"(sd {st:.3f}, sem {st / np.sqrt(n):.3f})")
    print(f"PAIRED shift (same seeds, same rows)                    = {dpair.mean():+.3f} "
          f"+- {dpair.std(ddof=1) / np.sqrt(n):.3f} points")
    print(f"training-weight coverage of constgold rows: {t['cover'].mean():.2f}%")
    print("=" * 76)

    if abs(mt) < abs(mc) - 2 * dpair.std(ddof=1) / np.sqrt(n):
        print("\n=> The pin closes MUCH better under its own training weighting. The residual is a\n"
              "   POPULATION-WEIGHTING mismatch, and importance-reweighting the response pin to the\n"
              "   deliverable population is the indicated fix. The paired shift above BOUNDS the gain.")
    else:
        print("\n=> The weighting mismatch does NOT explain the residual: m is comparable under both\n"
              "   weightings. Importance-reweighting the pin would move m by roughly the paired shift\n"
              "   above and no more, so it should NOT be built on this evidence.")

    # ---- where the two populations disagree, and where the pin residual is ----
    S_rs, S_rf, S_rb = cell_acc / n     # seed-averaged per-cell sums
    n_cg = n_cg_ref

    if args.save_weights:
        shp = (len(ef) - 1, len(es) - 1, len(ec) - 1)
        np.savez(args.save_weights, pop_w=n_cg.reshape(shp),
                 edges_flux=ef, edges_size=es, edges_crowd=ec,
                 crowd_col=str(z["crowd_col"].item()),
                 source=f"constgold in-domain occupancy, mag<{args.true_mag_max} "
                        f"Re>{args.true_re_min}, grid of {os.path.basename(args.target)}")
        print(f"\nwrote deliverable-population cell weights -> {args.save_weights}\n"
              f"  {int((n_cg > 0).sum())}/{n_cg.size} cells occupied, {n_cg.sum():,.0f} rows")
    occ = (n_cg > 0) & (cnt_tr > 0)
    w_cg = n_cg / n_cg.sum()
    w_tr_n = np.where(occ, cnt_tr, 0.0) / max(cnt_tr[occ].sum(), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dR = np.where(n_cg > 0, (S_rf + S_rb - S_rs) / np.maximum(n_cg, 1.0), np.nan)
    denom = (S_rf.sum() + S_rb.sum()) / n_cg.sum()
    contrib_cg = -100.0 * w_cg * dR / denom
    contrib_tr = -100.0 * w_tr_n * dR / denom

    print("\n--- the 10 cells that move m most under constgold weighting -----------------")
    print(f"{'cell(f,s,c)':>13} {'w_cg':>7} {'w_train':>8} {'dR':>9} {'dm_cg':>8} {'dm_train':>9}")
    nfs = (len(es) - 1), (len(ec) - 1)
    order = np.argsort(np.nan_to_num(np.abs(contrib_cg)))[::-1][:10]
    for b in order:
        fi = b // (nfs[0] * nfs[1]); si = (b // nfs[1]) % nfs[0]; ci = b % nfs[1]
        print(f"{f'({fi},{si},{ci})':>13} {w_cg[b]:>7.4f} {w_tr_n[b]:>8.4f} {dR[b]:>+9.4f} "
              f"{contrib_cg[b]:>+7.3f}% {contrib_tr[b]:>+8.3f}%")
    print(f"\nSUM over all cells: dm_cg={np.nansum(contrib_cg):+.3f}%  "
          f"dm_train={np.nansum(contrib_tr):+.3f}%")

    print("\n--- marginal population comparison (constgold in-domain vs training) --------")
    for name, axis, edges in (("TRUE MAG", 0, ef), ("TRUE SIZE", 1, es), ("CROWD r_blend", 2, ec)):
        shp = (len(ef) - 1, len(es) - 1, len(ec) - 1)
        a_cg = n_cg.reshape(shp).sum(axis=tuple(i for i in range(3) if i != axis))
        a_tr = cnt_tr.reshape(shp).sum(axis=tuple(i for i in range(3) if i != axis))
        a_cg = a_cg / a_cg.sum(); a_tr = a_tr / a_tr.sum()
        print(f"  {name}:")
        for i in range(len(a_cg)):
            print(f"    [{edges[i]:>8.4f},{edges[i + 1]:>8.4f})  constgold {a_cg[i]:6.3f}   "
                  f"training {a_tr[i]:6.3f}   ratio {a_cg[i] / max(a_tr[i], 1e-9):6.2f}")


if __name__ == "__main__":
    main()
