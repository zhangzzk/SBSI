"""AUDIT of the `analytic-control` probe's Table D ("bank support vs bank rows").

Two things the original could not separate, both fixed here:

  (1) GALAXY SAMPLE.  `--dedupe-bank` in `diag5c_analytic_control.py` drops duplicates from
      `rows` BEFORE the bank/galaxy split, so the deduped run also changes the galaxies (and
      `fills`, `scales`, the detection intercept `a0`).  Here the galaxy rows, the xhat draw,
      the units and `a0` are IDENTICAL across every bank.

  (2) ROWS vs SUPPORT vs EFFECTIVE-K.  At `d2`/`d4` every emission dim is a PRIMARY property,
      and rows sharing `input_index` are EXACT duplicates (verified: within-group ptp of
      (e1,e2,mag,logT) is 0.0).  So a raw K-row bank is mathematically the SAME mixture as a
      (K/8)-component bank with multiplicity weights, and its row-ESS is inflated by exactly
      the multiplicity.  The banks compared here are therefore:

        raw10k    rows [0,10000)              10000 rows, 1221 distinct primaries
        dedup1k   unique(input_index) of the same rows   1221 rows, the SAME 1221 primaries
        dedup10k  first 10000 distinct primaries         10000 rows, 10000 distinct

      raw10k vs dedup1k isolates "does adding exact-duplicate rows do ANYTHING?"  If they
      agree, Table D's 15.0 -> 2.29 is a change in the number of DISTINCT mixture components
      (plain effective-K), not a discovery that rows and support differ.

Also printed, and omitted from the original summary: the ORACLE row at g_true = 0.05.  The
Bartlett identity is exact only at g = 0; the (5.9b) drift is O(g * Var_i(I_i)/E[I]) and
therefore grows like sigma_e^-2, so at small sigma_e the EXACT one-node mixture is itself far
from ratio 1 and its ghat(5.9) collapses toward 0.  Any g != 0 row must be read against it.

Usage:
    python scripts/diag5c_anactl_audit.py --n-gal 4000 --sigma-e 0.3,0.1,0.03
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import load_rows  # noqa: E402
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency,
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)

from diag5c_analytic_control import (  # noqa: E402
    CAT, DIMSETS, TRUE_CUT, detection, emission, fill_and_scale, phi_analytic,
    scene_arrays,
)


def ratio_ci(s, info, s_sel, i_sel, nboot=400, seed=7):
    rng = np.random.default_rng(seed)
    n = s.size
    out = np.empty(nboot)
    for b in range(nboot):
        j = rng.integers(0, n, n)
        ld = float(np.mean(info[j]) - i_sel)
        out[b] = float(np.mean((s[j] - s_sel) ** 2)) / ld if ld != 0 else np.nan
    return float(np.nanpercentile(out, 16)), float(np.nanpercentile(out, 84))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--raw-rows", type=int, default=400_000)
    ap.add_argument("--n-gal", type=int, default=4000)
    ap.add_argument("--gal-row", type=int, default=140_000, help="in-domain row offset for galaxies")
    ap.add_argument("--k-raw", type=int, default=10_000)
    ap.add_argument("--k-dedup", type=int, default=10_000)
    ap.add_argument("--dimset", default="d2")
    ap.add_argument("--sigma-e", default="0.3,0.1,0.03")
    ap.add_argument("--sigma-nuis", type=float, default=1.0)
    ap.add_argument("--gammas", default="0.0,0.05")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--det-b", type=float, default=1.0)
    ap.add_argument("--det-c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    names = DIMSETS[args.dimset]
    sige_list = [float(v) for v in args.sigma_e.split(",")]
    gam_list = [float(v) for v in args.gammas.split(",")]

    rows = load_rows(args.catalogue, args.raw_rows)
    keep = ((rows["Re_input_p"].to_numpy(float) > TRUE_CUT[0])
            & (rows["r_input_p"].to_numpy(float) < TRUE_CUT[1]))
    rows = rows[keep].reset_index(drop=True)
    print(f"in-domain rows: {len(rows):,}")

    # ---- the three banks, all as index arrays into the SAME in-domain frame -------------
    ii = rows["input_index"].to_numpy()
    raw_idx = np.arange(args.k_raw)
    _, first_local = np.unique(ii[raw_idx], return_index=True)
    dedup_local_idx = np.sort(raw_idx[first_local])
    _, first_glob = np.unique(ii, return_index=True)
    glob_order = np.sort(first_glob)
    dedup_glob_idx = glob_order[:args.k_dedup]
    gal_idx = np.arange(args.gal_row, args.gal_row + args.n_gal)
    if gal_idx[-1] >= len(rows):
        raise SystemExit("not enough in-domain rows for the galaxy block")

    banks = {
        "raw10k": raw_idx,
        "dedup1k": dedup_local_idx,
        "dedup10k": dedup_glob_idx,
    }
    for nm, idx in banks.items():
        nd = len(np.unique(ii[idx]))
        print(f"  bank {nm:9s}: {len(idx):6,} rows, {nd:6,} distinct primaries "
              f"(mean multiplicity {len(idx)/nd:.2f}), max row {idx.max():,}")
    gal_pids = set(np.unique(ii[gal_idx]).tolist())
    for nm, idx in banks.items():
        ov = len(gal_pids & set(np.unique(ii[idx]).tolist()))
        print(f"  galaxy/bank primary overlap with {nm}: {ov}")

    # ---- units, fills, detection: computed ONCE from raw10k, shared by every config -----
    ref = scene_arrays(rows.iloc[raw_idx].reset_index(drop=True))
    fills = {}
    for k, v in ref.items():
        m = np.isfinite(v)
        fills[k] = float(v[m].mean()) if m.any() else 0.0
    ref = fill_and_scale(ref, fills)
    scales = {}
    for nm in set(sum(DIMSETS.values(), [])):
        if nm in ("e1", "e2"):
            continue
        scales[nm] = float(np.std(ref["logT0" if nm == "logT" else nm]))
    a0 = args.det_b * float(np.median(ref["mag"])) - args.det_c * float(np.median(ref["logT0"]))
    print(f"detection intercept a0={a0:.4f} (shared by ALL banks)")

    bank_sc = {nm: fill_and_scale(scene_arrays(rows.iloc[idx].reset_index(drop=True)), fills)
               for nm, idx in banks.items()}
    gal_sc = fill_and_scale(scene_arrays(rows.iloc[gal_idx].reset_index(drop=True)), fills)

    # population drift induced by dedupe: is it even the same p0?
    print("\npopulation of the emission dims, per bank (mean / std):")
    for nm in ("e1_0", "e2_0", "mag", "logT0"):
        line = f"  {nm:7s}"
        for bn in banks:
            v = bank_sc[bn][nm]
            line += f"  {bn}={np.mean(v):+.4f}/{np.std(v):.4f}"
        v = gal_sc[nm]
        line += f"  GAL={np.mean(v):+.4f}/{np.std(v):.4f}"
        print(line)

    sig = np.asarray([1.0 if nm in ("e1", "e2") else args.sigma_nuis * max(scales[nm], 1e-8)
                      for nm in names], float)
    ctr = emission({k: v[:2000] for k, v in ref.items()}, 0.0, names, sig)[0].mean(axis=0)

    def pop_terms(sc_bank, dl):
        lp = {}
        for t in (0.0, +dl, -dl):
            P, _, _, _ = detection(sc_bank, t, a0, args.det_b, args.det_c)
            lp[t] = float(np.log(P.mean()))
        return selection_terms(lp[0.0], (lp[+dl] - lp[-dl]) / (2 * dl),
                               (lp[+dl] - 2 * lp[0.0] + lp[-dl]) / dl ** 2)

    hdr = (f"{'sig_e':>6} {'g_true':>7} {'bank':>9} {'Krow':>6} {'Kdist':>6} {'ESSrow':>8} "
           f"{'ESSdist':>8} {'Var(s-c)':>12} {'<I>-Isel':>11} {'RATIO':>10} {'[16,84]%':>19} "
           f"{'ghat5.8':>9} {'ghat5.9':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for sige in sige_list:
        s_ = sig.copy()
        for j, nm in enumerate(names):
            if nm in ("e1", "e2"):
                s_[j] = sige
        for gtrue in gam_list:
            rng = np.random.default_rng(args.seed + int(round(1000 * gtrue)))
            Mg, _, _ = emission(gal_sc, gtrue, names, s_)
            xall = Mg + rng.normal(size=Mg.shape) * s_[None, :]
            Pg, _, _, _ = detection(gal_sc, gtrue, a0, args.det_b, args.det_c)
            det = rng.random(len(Pg)) < Pg
            xhat = xall[det]

            for bn, idx in banks.items():
                sc = bank_sc[bn]
                s_sel, i_sel = pop_terms(sc, args.delta)
                M0, dM0, ddM0 = emission(sc, 0.0, names, s_)
                _, lp0, dlp0, ddlp0 = detection(sc, 0.0, a0, args.det_b, args.det_c)
                phi0, d1, d2 = phi_analytic(xhat, M0, dM0, ddM0, s_, lp0, dlp0, ddlp0, ctr=ctr)
                s, info = score_and_information(phi0, d1, d2)
                bd, ld, ratio = denominator_consistency(s, info, s_sel, i_sel)
                lo, hi = ratio_ci(s, info, s_sel, i_sel)
                w = posterior_weights(phi0)
                ess_row = float(np.mean(1.0 / np.sum(w ** 2, axis=1)))
                # ESS over DISTINCT primaries: aggregate the weights within input_index
                pid = ii[idx]
                _, inv = np.unique(pid, return_inverse=True)
                nd = inv.max() + 1
                wagg = np.zeros((w.shape[0], nd))
                np.add.at(wagg.T, inv, w.T)
                ess_dist = float(np.mean(1.0 / np.sum(wagg ** 2, axis=1)))
                print(f"{sige:>6.3f} {gtrue:>7.3f} {bn:>9} {len(idx):>6d} {nd:>6d} "
                      f"{ess_row:>8.1f} {ess_dist:>8.1f} {bd:>12.3f} {ld:>11.3f} "
                      f"{ratio:>+10.3f} [{lo:>+8.3f},{hi:>+8.3f}] "
                      f"{shear_estimate_louis(s, info, s_sel, i_sel):>+9.5f} "
                      f"{shear_estimate_bartlett(s, s_sel):>+9.5f}")
                del phi0, d1, d2, w, wagg

            # ORACLE, printed at EVERY gamma (the original summary showed only g=0)
            gsel = {k: v[det] for k, v in gal_sc.items()}
            Mo, dMo, ddMo = emission(gsel, 0.0, names, s_)
            _, lpo, dlpo, ddlpo = detection(gsel, 0.0, a0, args.det_b, args.det_c)
            zz = (xhat - Mo) / s_
            dmu, ddmu = dMo / s_, ddMo / s_
            s_o = (zz * dmu).sum(1) + dlpo
            info_o = -((zz * ddmu).sum(1) - (dmu ** 2).sum(1) + ddlpo)
            s_sel_o, i_sel_o = pop_terms(bank_sc["raw10k"], args.delta)
            bd, ld, ratio = denominator_consistency(s_o, info_o, s_sel_o, i_sel_o)
            lo, hi = ratio_ci(s_o, info_o, s_sel_o, i_sel_o)
            print(f"{sige:>6.3f} {gtrue:>7.3f} {'ORACLE':>9} {int(det.sum()):>6d} "
                  f"{int(det.sum()):>6d} {1.0:>8.1f} {1.0:>8.1f} {bd:>12.3f} {ld:>11.3f} "
                  f"{ratio:>+10.3f} [{lo:>+8.3f},{hi:>+8.3f}] "
                  f"{shear_estimate_louis(s_o, info_o, s_sel_o, i_sel_o):>+9.5f} "
                  f"{shear_estimate_bartlett(s_o, s_sel_o):>+9.5f}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
