"""Turn the per-PAIR ruler into a per-PRIMARY SUMMED R_blend, binned by measured properties.

WHY. `m` uses `R_blend` = the emulator's response SUMMED over a primary's neighbours
(`scripts/build_blend_lookup.py`: `reg.groupby(index_input_p)['response'].sum()`), not a single pair.
The ruler measures pairs. The two are linked because the per-pair truth
`truth_j = de . ghat_j / |g|` is an UNBIASED estimator of `R_blend(j)` (the other neighbours enter
only through `cos 2(theta_i - theta_j)`, which has zero mean over their independent directions), so

    S_truth(primary) = sum_j truth_j     estimates    sum_j R_blend(j)

i.e. exactly the summed quantity, inside whatever aperture the catalogue annotates (7" for the ap7
legs). Errors are the scatter of S over PRIMARIES, so the within-primary correlation is handled by
construction.

Input: the npz written by `scripts/eval_rblend_gap_measured.py --all-neighbours`.
FIREWALL: derived entirely from the half-shear ruler; no constgold quantity is read or applied.
"""
from __future__ import annotations

import argparse

import numpy as np


def per_primary(pid, npid, vals):
    return np.bincount(pid, weights=vals, minlength=npid)


def show(label, name, key, edges, S, Sp, Sn, k, fmt="{:.2f}", tags=()):
    print(f"\n[{label}]")
    hdr = f"  {name:<22} {'S_truth':>9} {'sem':>8}"
    for t in tags:
        hdr += f" {t[-14:]:>9} {'rel%':>8} {'+-':>7}"
    hdr += f" {'S_null':>9} {'nsem':>8} {'<k>':>5} {'Nprim':>11}"
    print(hdr)

    def row(nm, m):
        n = int(m.sum())
        if n < 200:
            return
        t, sem = S[m].mean(), S[m].std(ddof=1) / np.sqrt(n)
        nl, nsem = Sn[m].mean(), Sn[m].std(ddof=1) / np.sqrt(n)
        line = f"  {nm:<22} {t:>9.4f} {sem:>8.4f}"
        for p in Sp:
            b = p[m].mean()
            line += f" {b:>9.4f} {(b/t-1)*100:>+8.2f} {abs(b/t)*(sem/abs(t))*100:>7.2f}"
        print(line + f" {nl:>+9.4f} {nsem:>8.4f} {k[m].mean():>5.2f} {n:>11,}")

    row("ALL", np.ones(len(S), bool))
    for i in range(len(edges) - 1):
        row(f"[{fmt.format(edges[i])},{fmt.format(edges[i+1])})",
            (key >= edges[i]) & (key < edges[i + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
                                     "ablation/eval/rblend_gap_measured_ap7.npz")
    ap.add_argument("--sep-max", type=float, default=None,
                    help="restrict the sum to neighbours closer than this (arcsec)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    tags = [str(t) for t in d["tags"]]
    truth, null, pid = d["truth"], d["null"], d["pid"].astype(np.int64)
    preds = [d[f"pred_{t}"] for t in tags]
    ok = np.isfinite(truth) & np.isfinite(null)
    for p in preds:
        ok &= np.isfinite(p)
    if args.sep_max is not None:
        ok &= d["dist"] <= args.sep_max
    npid = int(pid.max()) + 1
    print(f"rows={ok.sum():,} of {len(truth):,}   primaries={npid:,}"
          + (f"   [neighbours within {args.sep_max}\" only]" if args.sep_max else ""))

    pidk = pid[ok]
    k = per_primary(pidk, npid, np.ones(ok.sum()))
    S = per_primary(pidk, npid, truth[ok])
    Sn = per_primary(pidk, npid, null[ok])
    Sp = [per_primary(pidk, npid, p[ok]) for p in preds]

    # primary-level properties: constant within a primary, so take any row of it
    first = np.zeros(npid, np.int64)
    first[pid[::-1]] = np.arange(len(pid))[::-1]
    mag0, sz0, sn0, tmag = (d["mag0"][first], d["sz0"][first], d["sn0"][first], d["tmag"][first])
    # TRUE size of the primary. Added 2026-08-02: this is the axis on which the SUMMED R_blend was
    # claimed to be "3.1x too flat", a claim inferred from constgold via
    # `required_blend = r_sim - R_flow`. That inference assumes additivity, charges the flow's own
    # size error to the blend term, and uses a target carrying a size-structured leg-matching bias.
    # Here the same axis is read off the FIREWALL-CLEAN ruler, where none of those three apply.
    tre = d["tre"][first]

    have = k > 0
    S, Sn, Sp, k = S[have], Sn[have], [p[have] for p in Sp], k[have]
    mag0, sz0, sn0, tmag, tre = mag0[have], sz0[have], sn0[have], tmag[have], tre[have]
    print(f"primaries with >=1 usable neighbour: {have.sum():,}   "
          f"<neighbours per primary>={k.mean():.3f}")
    print(f"<S_truth>={S.mean():.5f} +- {S.std(ddof=1)/np.sqrt(len(S)):.5f}   "
          + "   ".join(f"<S_{t}>={p.mean():.5f}" for t, p in zip(tags, Sp)))

    show("SUMMED R_blend by PRIMARY MEASURED MAG (g=0 leg)", "measured_mag_auto(0)", mag0,
         [18, 24, 25, 25.5, 26, 26.5, 27, 32], S, Sp, Sn, k, tags=tags)
    show("SUMMED R_blend by PRIMARY MEASURED SIZE (arcsec)", "R_meas(0) [\"]", sz0,
         [0.0, 0.60, 0.70, 0.80, 1.00, 10.0], S, Sp, Sn, k, tags=tags)
    show("SUMMED R_blend by PRIMARY MEASURED S/N", "S/N(0)", sn0,
         [0, 10, 15, 20, 30, 50, 1e9], S, Sp, Sn, k, fmt="{:.0f}", tags=tags)
    show("SUMMED R_blend by PRIMARY TRUE MAG (continuity)", "r_input_p", tmag,
         [18, 23, 24, 25, 26], S, Sp, Sn, k, tags=tags)
    show("SUMMED R_blend by PRIMARY TRUE SIZE  <-- PHASE 2 PREMISE", "Re_input_p [\"]", tre,
         [0.30, 0.38, 0.46, 0.56, 0.70, 0.95, 1.50], S, Sp, Sn, k, tags=tags)
    print("\n  HOW TO READ THE TRUE-SIZE TABLE. The Phase 2 claim is that the emulator's summed")
    print("  R_blend is '3.1x too flat' in true size: predicted spanning 0.116-0.145 where the")
    print("  constgold-required value spans 0.071-0.160. On THIS table the test is whether the")
    print("  emulator's span across true-size bins is ~3x narrower than the RULER TRUTH's span.")
    print("  If truth and emulator span comparable ranges here, the flatness is not a property of")
    print("  the emulator -- it is an artefact of the constgold `r_sim - R_flow` inference, and")
    print("  Phase 2 is aimed at a defect the firewall-clean instrument does not see.")
    for t, p in zip(tags, Sp):
        import numpy as _np
        bs = _np.digitize(tre, [0.30, 0.38, 0.46, 0.56, 0.70, 0.95, 1.50]) - 1
        mt = _np.array([S[bs == b].mean() if (bs == b).sum() > 200 else _np.nan for b in range(6)])
        mp = _np.array([p[bs == b].mean() if (bs == b).sum() > 200 else _np.nan for b in range(6)])
        rt = _np.nanmax(mt) - _np.nanmin(mt)
        rp = _np.nanmax(mp) - _np.nanmin(mp)
        print(f"    {t}: truth span {rt:.4f}, emulator span {rp:.4f} -> emulator is "
              f"{rt/rp if rp else _np.nan:.2f}x flatter than truth "
              f"(the constgold-inferred claim was 3.1x)")

    # the two constgold-relevant cells, as single rows
    print("\n[CELLS]")
    for nm, m in (("measured mag < 26", mag0 < 26),
                  ("measured mag >= 26 (SHELL)", mag0 >= 26),
                  ("measured size < 0.60\"", sz0 < 0.60),
                  ("measured size >= 0.60\"", sz0 >= 0.60)):
        n = int(m.sum())
        t, sem = S[m].mean(), S[m].std(ddof=1) / np.sqrt(n)
        nl, nsem = Sn[m].mean(), Sn[m].std(ddof=1) / np.sqrt(n)
        cells = "  ".join(f"{t2}: {p[m].mean():.5f} ({(p[m].mean()/t-1)*100:+.1f}% "
                          f"+- {abs(p[m].mean()/t)*(sem/abs(t))*100:.1f})"
                          for t2, p in zip(tags, Sp))
        print(f"  {nm:<28} S_truth={t:.5f} +- {sem:.5f}   {cells}   "
              f"null={nl:+.5f} +- {nsem:.5f}   N={n:,}")
    print("SUMMED_DONE")


if __name__ == "__main__":
    main()
