"""PHASE 0a -- does the model work where the EMULATOR contributes nothing?

WHY THIS EXISTS. The 2026-08-01s attribution (flux axis -> FLOW, size axis -> EMULATOR, blend axis ->
BOTH and anti-correlated) rests on two things that were never checked:

  1. sim additivity, `R_total = R_self + R_blend`. Every per-bin "required blend"
     (`r_sim - R_flow`) is only the emulator's target IF that identity holds.
  2. the flow's own error on constgold's OWN extraction. fig5 measures the flow against HALF-SHEAR
     with FORWARD extraction; constgold is ANTITHETIC. That extraction difference is a known real
     effect (+0.49 vs +0.60 at the faint end), so the fig5 column pasted next to constgold numbers
     carries an unremoved confounder.

ISOLATED ROWS CLOSE BOTH AT ONCE -- IF "ISOLATED" IS DEFINED CORRECTLY. On rows with no neighbour
the blend term is zero and the model reduces to `R_flow` alone, so

    r_sim(isolated)  vs  R_flow(isolated)

is a clean flow-only comparison, on constgold, in constgold's own extraction -- no emulator, no
cross-population pairing, no forward-vs-antithetic gap.

*** CORRECTED 2026-08-02, FIRST VERSION OF THIS SCRIPT WAS WRONG (job 15477290). ***
The first version used `neighbored = False` as the isolation criterion and concluded the flow was
7.8% low on isolated rows with a 9.21 pt size-axis rms -- a headline "the attribution is invalid"
result. THAT CONCLUSION WAS AN ARTEFACT OF THE FLAG. `neighbored` marks only whether the catalogue
annotated a PAIR SECONDARY within 3"; it says nothing about the rest of the field. Two independent
facts settle it:

  * `scripts/build_crowding_lookup.py` sets `NEAR, FAR = 3.0, 7.0` arcsec -- so `nbr_flux_far` exists
    precisely to carry neighbours BEYOND the 3" pair radius, and it is non-zero on these rows.
  * `scripts/build_blend_lookup.py` sums the emulator "over its sheared neighbours (SECONDARY, drawn
    from the FULL FIELD)" -- not over the annotated pair.

So `neighbored = False` rows genuinely blend, the emulator's `<R_blend> = +0.095` on them is not a
bug, and the `+0.071` required blend there is real blending rather than a flow deficit. This is the
SAME trap that produced one retraction already in this project (np7's 7" build flag read against the
ruler's 3"); it is recorded here so the third occurrence is caught faster.

WHAT THIS VERSION DOES INSTEAD: an ISOLATION LADDER. Isolation is not a flag, it is a radius, so the
script prints every available definition side by side and runs the per-bin test on the strictest one:

    pair3      `neighbored = False`            no annotated secondary within 3"  -- NOT isolation
    near0      `nbr_flux_near == 0`            no neighbour flux inside 3"
    near0far0  near and far both zero          no neighbour flux inside 7"
    emu0       `R_blend < eps`                 the emulator itself sees nothing to respond to
    CLEAN      near0far0 AND emu0              what the flow-only test actually needs

`nbr_flux_* = log10(1 + F_shell/rms)` is exactly zero iff the shell flux is zero, so `== 0` is an
exact statement about the sim, not a threshold choice.

WHAT EACH OUTCOME MEANS. Let `resid = R_model/r_sim - 1` per bin.

  isolated resid small AND flat      -> the flow is right where it is unaided. Structure on blended
                                        rows is then the emulator or additivity, and the 2026-08-01s
                                        attribution stands.
  isolated resid large / structured  -> the flow is ALREADY wrong with no emulator involved. Whatever
                                        that structure is, it was being charged to the emulator by
                                        the `required_blend = r_sim - R_flow` construction, because
                                        that construction assigns 100% of the flow's error to the
                                        blend term. RE-DERIVE THE ATTRIBUTION.

THE SIZE AXIS IS THE ONE THAT MATTERS. Phase 2 exists because `R_blend` is 3.1x too flat in true
size. If the SIZE structure is already present in the isolated residual, Phase 2 is aimed at the
wrong component. This script prints the isolated-row size trend first for that reason.

A SEPARATE, WEAKER CONTINUITY TEST is printed too: residual binned by neighbour flux and by pair
distance, isolated and blended side by side. Under additivity the blended residual should approach
the isolated one as the neighbour becomes faint/distant; a discontinuity at the isolated/blended
boundary that the emulator's own prediction does not explain is evidence against additivity.

LIMITS -- state them, do not let the reader infer them.
  * This does NOT test additivity in the strong sense. A genuine test needs the same objects rendered
    with and without their neighbours; constgold has no such pair. Isolated and blended rows are
    DIFFERENT galaxies, so any isolated-vs-blended difference confounds additivity with population.
    What it does test is the premise `R_blend = 0 => R_model = R_flow`, which is the assumption the
    per-bin required-blend construction actually leans on.
  * Even the CLEAN set only removes neighbours out to 7". The emulator sums over the full field, so a
    faint far-field contribution survives; `<R_blend>` on the CLEAN set is printed as the residual
    scale of that leakage, and it bounds how much of any deficit could still be blending.
  * Selecting on `R_blend < eps` conditions on the emulator's OWN output. Used alone that would be
    circular; it is used only in AND with the sim-side `near0far0`, which is a statement about the
    rendered field and cannot be argued away.
  * No cuts, no tuning, no fit. Diagnostic read of existing dumps; constgold is evaluation-only
    (the R_blend firewall). Nothing here can promote or demote a model.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plotting.plot_fid_flow_figures import (  # noqa: E402
    CONST_CAT, CROWD, _read_key_table, load_dumps)

# (column, label, log-x, x floor, needs-CROWD)
AXES = [("Re_input_p", "primary TRUE size  Re [arcsec]", False, None),
        ("S/N_plus", "primary flux  S/N", True, None),
        ("r_input_p", "primary TRUE magnitude", False, None),
        ("nbr_flux_near", "neighbour / blend flux", True, 1e-3)]


def edges_for(x, nb, logx, xmin):
    m = np.isfinite(x)
    if xmin is not None:
        m &= x > xmin
    xx = np.log10(x[m]) if logx else x[m]
    return np.linspace(np.nanpercentile(xx, 0.5), np.nanpercentile(xx, 99.5), nb + 1), m


def per_bin(x, rs, rf, rb, nb, logx, xmin, use_blend):
    """Bin and return rows of (centre, n, sim, flow, blend, model, resid%)."""
    ed, m = edges_for(x, nb, logx, xmin)
    m = m & np.isfinite(rs) & np.isfinite(rf).all(0) & np.isfinite(rb)
    if m.sum() < 30:
        return np.zeros((0, 7)), ed
    xx = np.log10(x[m]) if logx else x[m]
    idx = np.digitize(xx, ed) - 1
    rs_, rf_, rb_ = rs[m], rf[:, m], rb[m]
    rows = []
    for b in range(nb):
        s = idx == b
        n = int(s.sum())
        if n < 30:
            continue
        cx = 10 ** (0.5 * (ed[b] + ed[b + 1])) if logx else 0.5 * (ed[b] + ed[b + 1])
        st = float(np.mean(rs_[s]))
        fl = float(rf_[:, s].mean())
        bl = float(np.mean(rb_[s])) if use_blend else 0.0
        tot = fl + bl
        rows.append((cx, n, st, fl, bl, tot, (tot / st - 1.0) * 100.0 if st else np.nan))
    return np.array(rows, float), ed


def show(tag, a):
    if not len(a):
        print(f"  {tag}: too few rows")
        return
    print(f"  {'bin x':>10}{'n':>12}{'r_sim':>9}{'R_flow':>9}{'R_blend':>9}"
          f"{'R_model':>9}{'resid%':>9}")
    for r in a:
        print(f"  {r[0]:>10.4g}{int(r[1]):>12,}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>9.4f}"
              f"{r[5]:>9.4f}{r[6]:>+9.2f}")
    res = a[:, 6]
    print(f"  -> {tag}: rms {np.sqrt(np.mean(res**2)):.2f} pt | "
          f"min {res.min():+.2f} max {res.max():+.2f} | span {res.max()-res.min():.2f} pt")
    return float(np.sqrt(np.mean(res**2))), float(res.max() - res.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--save-npz", default="results/additivity_isolated.npz")
    ap.add_argument("--min-clean", type=int, default=5000,
                    help="below this many truly-isolated rows the per-bin flow test is reported "
                         "BLOCKED rather than run on a sample too small to mean anything")
    args = ap.parse_args()

    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    key = ["case", "input_index"]
    cases = set(ref["case"].unique().tolist())
    ref = ref.merge(_read_key_table(
        CONST_CAT, key + ["Re_input_p", "S/N_plus", "r_input_p", "neighbored", "distance"], cases),
        on=key, how="left")
    ref = ref.merge(_read_key_table(
        CROWD, key + ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"], cases), on=key, how="left")

    rsim = ref["r_sim"].to_numpy(float)
    rb = ref["R_blend"].to_numpy(float)
    rf = flows.astype(np.float64)
    nb_flag = ref["neighbored"].to_numpy()
    dist = ref["distance"].to_numpy(float)
    fnear = ref["nbr_flux_near"].to_numpy(float)
    ffar = ref["nbr_flux_far"].to_numpy(float)

    # ---- ISOLATION LADDER: isolation is a radius, not a flag. Print every definition. -------
    pair3 = (nb_flag == 0)
    near0 = np.isfinite(fnear) & (fnear <= 0.0)
    far0 = np.isfinite(ffar) & (ffar <= 0.0)
    near0far0 = near0 & far0
    emu0 = np.isfinite(rb) & (np.abs(rb) < 0.005)
    clean = near0far0 & emu0
    ladder = [("pair3      neighbored=False (NOT isolation)", pair3),
              ("near0      no neighbour flux within 3\"", near0),
              ("near0far0  no neighbour flux within 7\"", near0far0),
              ("emu0       emulator |R_blend| < 0.005", emu0),
              ("CLEAN      near0far0 AND emu0", clean)]
    print(f"\n{'='*112}\nISOLATION LADDER -- how many rows each definition calls isolated, and what "
          f"the emulator says there\n{'='*112}")
    print(f"  {'definition':<44}{'N':>12}{'frac':>8}{'<R_blend>':>12}{'<nbr_far>':>11}"
          f"{'<r_sim>':>9}{'<R_flow>':>10}{'flow-only resid':>24}")
    for name, s in ladder:
        n = int(s.sum())
        if not n:
            print(f"  {name:<44}{n:>12,}   EMPTY")
            continue
        st, fl = float(np.nanmean(rsim[s])), float(np.nanmean(rf[:, s]))
        # Per-object response scatter is std ~5 against a mean ~0.86, so a small isolated set has a
        # huge error. WITHOUT this bar a -13% on 1,855 rows reads as a result; it is noise.
        sd = float(np.nanstd(rsim[s]))
        sem = sd / np.sqrt(max(n, 1))
        rel = 100.0 * sem / abs(st) if st else np.nan
        print(f"  {name:<44}{n:>12,}{100*n/len(ref):>7.2f}%{np.nanmean(rb[s]):>12.5f}"
              f"{np.nanmean(ffar[s]):>11.4f}{st:>9.4f}{fl:>10.4f}"
              f"{(fl/st - 1)*100:>+15.3f}% +-{rel:>5.2f}%")
    print("\n  READ THIS ROW BY ROW. `pair3` is the definition the first version of this script used;"
          "\n  its non-zero <R_blend> and non-zero <nbr_far> are exactly why it is NOT isolation.")

    # distance is only defined for annotated pairs -- print its support so the 3" radius is visible
    d = dist[np.isfinite(dist)]
    if len(d):
        q = np.percentile(d, [0, 50, 99, 100])
        print(f"  `distance` (annotated pair separation) is finite on {len(d):,} rows: "
              f"min {q[0]:.3f}  med {q[1]:.3f}  p99 {q[2]:.3f}  MAX {q[3]:.3f} arcsec "
              f"<- the pair-build radius")

    if int(clean.sum()) < args.min_clean:
        n = int(clean.sum())
        st = float(np.nanmean(rsim[clean])) if n else np.nan
        sem = float(np.nanstd(rsim[clean])) / np.sqrt(max(n, 1)) if n else np.nan
        print(f"\n{'='*112}\nPHASE 0a VERDICT: **BLOCKED** -- not decidable on constgold\n{'='*112}")
        print(f"  The CLEAN isolated set has {n:,} rows ({100*n/len(ref):.3f}% of the population).")
        print(f"  Per-object response scatter is std {float(np.nanstd(rsim[clean])):.2f} against a "
              f"mean of {st:.3f}, so the error on its mean response alone is "
              f"+-{100*sem/abs(st):.1f}%.")
        print("  A test with a +-{:.0f}% error cannot adjudicate a 2-3 pt per-bin effect, and "
              "splitting\n  it into 12 size bins makes it worse by another factor of ~3.5."
              .format(100 * sem / abs(st)))
        print("\n  WHY, physically: the constgold field is dense. Requiring NO neighbour flux within")
        print("  7\" keeps 0.03% of rows. There is no isolated subpopulation in this sim to run a")
        print("  flow-only test on -- the premise of Phase 0a as written in PLAN_resolution.md")
        print("  ('for ISOLATED galaxies R_blend = 0 by construction') does not hold for any")
        print("  workable sample size.")
        print("\n  WHAT IS NOT CONCLUDED. This does NOT show additivity fails, and it does NOT show")
        print("  the flow is fine. It shows this particular route to the question is unavailable.")
        print("  Do not substitute the `pair3`/`near0` number (-7.822%) for it: those rows have")
        print("  <nbr_far> = 1.71 and a real blend response, which is exactly the error the first")
        print("  version of this script made.")
        print("\n  THE REMAINING ROUTE is the half-shear ghat_p projection (fig5 /")
        print("  scripts/dump_halfshear_selfresp.py), which isolates the SELF response by averaging")
        print("  neighbours away instead of by selecting isolated galaxies, and so works on the full")
        print("  2.36M-row population. Its confounder is the forward-vs-antithetic extraction gap,")
        print("  which is what Phase 0b measures.")
        np.savez(args.save_npz, blocked=True, n_clean=n,
                 ladder_n=np.array([int(s.sum()) for _, s in ladder]))
        print(f"\nsaved -> {args.save_npz} (blocked marker + ladder counts)")
        return

    iso, bld = clean, ~near0far0
    print(f"\n{'='*112}\nGLOBAL, CLEAN-ISOLATED vs BLENDED\n{'='*112}")
    for tag, s, ub in (("CLEAN isolated (model = R_flow only)", iso, False),
                       ("CLEAN isolated (+ emulator leakage, for reference)", iso, True),
                       ("blended, i.e. any flux within 7\" (R_flow + emulator)", bld, True)):
        st, fl = float(np.nanmean(rsim[s])), float(np.nanmean(rf[:, s]))
        bl = float(np.nanmean(rb[s])) if ub else 0.0
        print(f"  {tag:<54} r_sim {st:+.4f}  R_flow {fl:+.4f}  R_blend {bl:+.4f}  "
              f"-> resid {((fl+bl)/st - 1)*100:+.3f}%")

    store, gates = {}, {}
    for col, lab, logx, xmin in AXES:
        x = ref[col].to_numpy(float)
        print(f"\n{'='*100}\n{lab}\n{'='*100}")
        print("  CLEAN-ISOLATED rows -- model is R_flow ALONE, no emulator, constgold extraction:")
        a_iso, _ = per_bin(x[iso], rsim[iso], rf[:, iso], rb[iso], args.nbins, logx, xmin, False)
        g_iso = show("isolated (flow only)", a_iso)
        print("\n  BLENDED rows -- model is R_flow + emulator R_blend:")
        a_bld, _ = per_bin(x[bld], rsim[bld], rf[:, bld], rb[bld], args.nbins, logx, xmin, True)
        g_bld = show("blended (flow+emulator)", a_bld)
        store[f"iso_{col}"], store[f"bld_{col}"] = a_iso, a_bld
        if g_iso and g_bld:
            gates[col] = (g_iso[0], g_bld[0])
            print(f"\n  -> SHARE: clean-isolated rms is {100*g_iso[0]/g_bld[0]:.0f}% of the "
                  f"blended-row rms on this axis. Whatever the clean-isolated part is, it is NOT "
                  f"the emulator -- there is nothing for the emulator to respond to.")

    # ---- verdict ---------------------------------------------------------------------------
    print(f"\n{'='*100}\nPHASE 0a GATE\n{'='*100}")
    print("  Gate as written in PLAN_resolution.md: 'if isolated-row |R_flow/R_sim - 1| is small and")
    print("  flat, additivity is supported and the 2026-08-01s attribution stands. If it is large and")
    print("  structured, stop and re-derive the attribution.'")
    for col, (ri, rbl) in gates.items():
        print(f"    {col:>16}: clean-isolated rms {ri:5.2f} pt | blended rms {rbl:5.2f} pt")
    if "Re_input_p" in gates:
        ri, rbl = gates["Re_input_p"]
        print(f"\n  SIZE AXIS IS THE DECIDING ONE (Phase 2 is aimed at the emulator's size flatness).")
        print(f"    clean-isolated rms on true size = {ri:.2f} pt, with NO emulator involved.")
        print(f"    If that is a large fraction of the {rbl:.2f} pt blended-row rms, then the size")
        print(f"    structure is at least partly the FLOW, and `required_blend = r_sim - R_flow`")
        print(f"    has been charging the flow's own size error to the emulator.")
    np.savez(args.save_npz, **store, seeds=np.array(seeds))
    print(f"\nsaved -> {args.save_npz}")
    print("\nCAVEATS, repeated deliberately:")
    print("  1. Clean-isolated and blended rows are DIFFERENT GALAXIES. This tests the premise "
          "'no neighbour => model is R_flow', not additivity in the strong sense, which would need "
          "the same object rendered both ways.")
    print("  2. The clean set is isolated only out to 7\". Read its <R_blend> in the ladder above as "
          "the residual far-field leakage; a deficit smaller than that is not resolved by this test.")
    print("  3. The clean set is a BIASED subsample of the population (isolation correlates with "
          "field density and hence with magnitude). Compare its per-bin residual against the blended "
          "one bin by bin, never its global mean against the global m.")


if __name__ == "__main__":
    main()
