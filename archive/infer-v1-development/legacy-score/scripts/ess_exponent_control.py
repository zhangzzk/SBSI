"""Archived finite-M catalogue-draw control retained to reproduce cont.191.

This is not part of the current finite-scene catalogue likelihood pipeline.

THE CONTROL: is ESS doing anything a plain fitted EXPONENT on M cannot?

ESS ~ c*M^p, so much of what the 1/ESS abscissa buys may be nothing but a changed exponent
-- a one-parameter change with no effective-sample-size content.  Two tiers of control,
weakest assumption first:

  UNIVERSAL p   one exponent shared by every configuration.  The strongest form of the
                control: if even the best universal power law matches ESS, ESS adds nothing.
  PER-CONFIG p  three free exponents, one per configuration.  This separates the two claims
                bundled in "the configuration-specific part of ESS is what works":

                  * if it reproduces most of ESS's improvement, ESS's content reduces to
                    "each configuration has its own effective exponent" -- real, but three
                    numbers, and the per-rung departures are decoration;
                  * if it falls short, the per-rung structure does work no power law can do.

                Note the asymmetry that makes this sharp: the per-config power law has THREE
                free parameters and ESS has NONE -- ESS is measured, not fitted.  A
                zero-parameter model beating a three-parameter one is a far stronger result
                than the universal comparison alone.

DEGENERACY, AND WHY THE OBVIOUS DIAGNOSTIC DOES NOT DETECT IT.  As p -> 0 the regressor
M^-p -> 1 for every rung, the slope stops being identifiable and its error blows up.  chi2
then falls FOR WANT OF POWER and the two-config sigma falls with it, while the span
explodes -- so BOTH the chi2 and the sigma columns are FLATTERED by degeneracy and only the
span flags it.  Struck rows are therefore struck from all three columns, not discounted in
one.

The tempting test -- fractional error on the fitted slope, sd/|a| -- CANNOT DO THIS JOB, and
the reason is structural rather than a matter of threshold.  Shrinking the regressor's range
inflates the slope and its error IN LOCKSTEP, because both are the same ratio against the
same shrinking abscissa: measured here, nbr runs a = -8.15% +/- 0.724% at p = 1 and
a = -40.96% +/- 4.034% at p = 0.05 -- a 5x inflation of both, leaving sd/|a| at 8.9% against
9.8%, essentially unmoved across the entire degenerate range.  Meanwhile a configuration
whose slope is genuinely CONSISTENT WITH ZERO (struct b=0.8, a = +3.3% +/- 2.3%) shows ~70%
fractional error at EVERY p, and a fractional-error cut strikes it everywhere for a physical
result rather than a degeneracy.  Applied to these three configurations the cut strikes all
rows and keeps none.

What does track the degeneracy is the slope error measured AGAINST ITS OWN BEST VALUE over
the scan -- the operational statement that this parametrisation costs precision.  A row is
struck when any configuration's slope error exceeds `DEGEN_INFLATE` times the smallest that
configuration achieves anywhere on the grid.  The regressor's dynamic range, printed
alongside, is the same fact stated geometrically.
"""
import glob, pathlib, sys
import numpy as np
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/scripts")
import combine_catprior as cc

TABLES = {}     # blocks doc/WORKLOG.md carries verbatim; see EMITTED BLOCK
POOL = [0, 0]   # inversions / draws pooled over both brackets
QUOTED = {}   # every number doc/WORKLOG.md is allowed to quote; see EMITTED BLOCK below

D_ = "/project/ls-gruen/users/zekang.zhang/sbsi_catprior/"
files = sorted(glob.glob(D_ + "catprior_*_d777_*_irow.npz"))

# a parametrisation that inflates the slope error by half again over the best available
# has spent a third of the ladder's constraining power on the choice of abscissa alone
DEGEN_INFLATE = 1.50
# ...and the OPPOSITE limit.  As p grows, M^-p collapses onto an indicator of the M=1 rung
# (2^-p at the second rung), so the "power law" stops being one: it becomes a two-point step
# that says only "M=1 differs from the rest".  A row is struck once the second rung carries
# less than STEP_MIN of the first, because past that the control is no longer testing the
# hypothesis it was built to test.  The error-inflation test above CANNOT see this -- the
# step fit is perfectly well conditioned, it is just not a power law.
STEP_MIN = 0.05
PGRID = [round(x, 3) for x in np.arange(0.05, 12.0001, 0.025)]  # THE GRID TOP IS A MEASURED
# CHOICE.  struct b=0.8 has a slope consistent with zero, so no exponent is identified for
# it and the tuned best-case optimum runs to whatever edge the grid provides -- a bound that
# stops at its own edge is not a bound.  Scanning the top: 3.0 -> 1.49x, 6.0 -> 1.43x,
# 12.0 -> 1.40x, 24.0 -> 1.39x, so it converges and 12.0 is inside the plateau.  The honest
# per-config span is flat from 6.0 onward (133.7%).


def prep(run):
    """(D, reps, ladder) for one configuration -- the expensive part, done once."""
    g = np.array([float(run["cfg"]["closure_g"]), float(run["cfg"]["closure_g2"])])
    gn = float(np.hypot(*g))
    st = cc.stats(run, g / gn, gn)
    ladder = list(run["ladder"])
    D = np.array([[st[m][0] - st["pin"][0] for m in ladder]])
    reps = np.stack([st[m][1] - st["pin"][1] for m in ladder], axis=1)
    return D, reps, ladder


def slope(pre, x=None):
    """(a, sigma_a, chi2, dof) for one configuration on abscissa `x` (default 1/M)."""
    D, reps, ladder = pre
    fits, _ = cc.fit_ladder(D, reps, ladder, x=x)
    for name, bh, cov, c2, mask, order in fits:
        if order == 1 and int(mask.sum()) == len(ladder):
            return (float(bh[1]), float(np.sqrt(cov[1, 1])), float(c2),
                    int(mask.sum()) - 2)
    return None, None, None, None


def ess_x(run, pre):
    """Information-weighted <1/ESS> per rung -- MEASURED, not fitted, zero free parameters."""
    if run.get("ess_rung") is None:
        return None
    g = np.array([float(run["cfg"]["closure_g"]), float(run["cfg"]["closure_g2"])])
    gdir = g / float(np.hypot(*g))
    ladder = pre[2]
    w = cc.row_weights(run, ladder, gdir, source="pin")
    inv = cc.ess_abscissa([run["ess_rung"]], ladder,
                          wr=([w] if w is not None else None))[0]
    return np.array([inv[m] for m in ladder]), (w is not None)


def common_chi2(vals, errs):
    v, e = np.asarray(vals, float), np.asarray(errs, float)
    w = 1.0 / e ** 2
    c = (v * w).sum() / w.sum()
    return float((w * (v - c) ** 2).sum()), float(c), float(v.max() - v.min())


def rel_span(vals, errs):
    """span / |common a| -- the DIMENSIONLESS version of the span.

    WHY BOTH ARE REPORTED.  Under a uniform rescale x -> c*x of the abscissa, every fitted
    slope and its error scale by 1/c together.  chi2 and the two-config sigma are therefore
    INVARIANT (weights scale by c^2 against residuals scaling by 1/c^2; a difference over its
    own error), and so is this ratio -- but the raw span is not: it scales by 1/c.  Of the
    four summaries only the raw span carries units.

    That does NOT make the raw span meaningless here, because there is no free c: EVERY
    abscissa in this script equals exactly 1 at M = 1 -- ESS because one draw has effective
    sample size one by construction, M^-p trivially -- so the scale is anchored, not chosen,
    and verified to 1e-10 in `tests/test_catprior_ladder_fit.py`.  What the abscissae do NOT
    share is their RANGE (ESS falls to 0.118 at nbr's top rung but only to 0.396 at struct's,
    while M^-1 falls to 0.063 and 0.016), so the anchoring pins the origin without making the
    units identical.  Raw span is kept for what it did well -- flagging the p -> 0 degeneracy,
    which it alone caught -- and this ratio is what an effect size should be quoted on.
    """
    c2, c, span = common_chi2(vals, errs)
    return span / abs(c) if c else float("inf")


def two_cfg_sigma(vals, errs):
    return abs(vals[0] - vals[1]) / float(np.hypot(errs[0], errs[1]))


runs = [cc.load(f) for f in files]
pres = [prep(r) for r in runs]
names = [f.split("/")[-1].replace("catprior_", "").replace("_irow.npz", "") for f in files]
print("configurations:")
for n, (_, _, lad) in zip(names, pres):
    print(f"  {n}   ladder {lad}")

# ---------------------------------------------------------------------------------------
# the ESS reference, COMPUTED from ess_rung rather than pasted in
# ---------------------------------------------------------------------------------------
ex = [ess_x(r, p) for r, p in zip(runs, pres)]
if any(e is None for e in ex):
    raise SystemExit("ESS reference needs `ess_rung` in every run; re-run the missing ones.")
weighted = all(e[1] for e in ex)
ess_v, ess_e = [], []
for pre, (x, _) in zip(pres, ex):
    a, sa, _, _ = slope(pre, x=x)
    ess_v.append(a * 100); ess_e.append(sa * 100)
ESS_C2, ESS_A, ESS_SPAN = common_chi2(ess_v, ess_e)
ESS_SIG = two_cfg_sigma(ess_v, ess_e)
ESS_RSPAN = rel_span(ess_v, ess_e)
print(f"\nESS reference (zero fitted parameters; abscissa "
      f"{'information-weighted' if weighted else 'FLAT ROW MEAN -- unbounded shift'}):")
print(f"  common a = {ESS_A:.3f}%, span {ESS_SPAN:.2f}% (span/|a| = {ESS_RSPAN:.3f}), "
      f"chi2 = {ESS_C2:.2f}, two-config {ESS_SIG:.1f} sigma")

# ---------------------------------------------------------------------------------------
# precompute every (configuration, p) slope once
# ---------------------------------------------------------------------------------------
TAB = {}
for p in PGRID:
    for i, pre in enumerate(pres):
        x = np.array([float(m) ** (-p) for m in pre[2]])
        TAB[(i, p)] = slope(pre, x=x)


SD_MIN = {i: min(TAB[(i, p)][1] for p in PGRID) for i in range(len(pres))}
XRANGE = {(i, p): float(np.ptp(np.array([float(m) ** (-p) for m in pres[i][2]])))
          for i in range(len(pres)) for p in PGRID}


def flat_degenerate(i, p, infl=None):
    """Slope too weakly identified: error inflated against the configuration's OWN best.

    Not sd/|a| -- see the module docstring for why the fractional error is blind to this.
    """
    return TAB[(i, p)][1] > (DEGEN_INFLATE if infl is None else infl) * SD_MIN[i]


def step_degenerate(p):
    """Regressor has collapsed onto an M=1 indicator, so it is no longer a power law."""
    return 2.0 ** (-p) < STEP_MIN


def degenerate(i, p, infl=None, step=True):
    return flat_degenerate(i, p, infl) or (step and step_degenerate(p))


# ---------------------------------------------------------------------------------------
# TIER 1 -- one universal exponent
# ---------------------------------------------------------------------------------------
print(f"\n{'=' * 88}\nTIER 1: UNIVERSAL exponent (one p for all configurations)")
print(f"{'p':>7} {'common a':>11} {'chi2':>10} {'span':>10} {'span/|a|':>9} {'2-cfg':>7}  "
      f"{'worst sd/best':>13} {'x range':>8}")
best = None
SHOW = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.825, 1.00]
for p in SHOW:
    vals = [TAB[(i, p)][0] * 100 for i in range(len(pres))]
    errs = [TAB[(i, p)][1] * 100 for i in range(len(pres))]
    c2, c, span = common_chi2(vals, errs)
    sig = two_cfg_sigma(vals, errs)
    infl = max(TAB[(i, p)][1] / SD_MIN[i] for i in range(len(pres)))
    xr = min(XRANGE[(i, p)] for i in range(len(pres)))
    deg = any(degenerate(i, p) for i in range(len(pres)))
    why = ("FLAT (slope unidentified)" if any(flat_degenerate(i, p) for i in range(len(pres)))
           else "STEP (not a power law)")
    tag = f"  <-- struck: {why}" if deg else ""
    print(f"{p:>7.3f} {c:>10.3f}% {c2:>10.2f} {span:>9.2f}% {span / abs(c):>9.3f} "
          f"{sig:>7.1f}  {infl:>12.2f}x {xr:>8.2f}{tag}")
# EACH MEASURE GETS ITS OWN BEST POWER LAW.  Selecting one exponent and then quoting four
# summaries at it confounds "how good can this family be" with "which criterion picked the
# exponent" -- and the criteria disagree sharply: raw span has an interior minimum, while
# span/|a| rises MONOTONICALLY across the whole non-degenerate range (5.40 at p = 0.05 to
# 12.47 at p = 1.00), so selecting on it would drive the choice straight back into the
# degenerate region it is meant to guard against.  Giving the power law its best exponent
# separately on every measure removes the selection from the comparison entirely and is the
# most generous reading available to it.
OK = [p for p in PGRID if not any(degenerate(i, p) for i in range(len(pres)))]
MEAS = {}
for p in OK:
    vals = [TAB[(i, p)][0] * 100 for i in range(len(pres))]
    errs = [TAB[(i, p)][1] * 100 for i in range(len(pres))]
    c2, c, span = common_chi2(vals, errs)
    MEAS[p] = dict(chi2=c2, sigma=two_cfg_sigma(vals, errs), span=span,
                   rspan=rel_span(vals, errs))
ESSREF = dict(chi2=ESS_C2, sigma=ESS_SIG, span=ESS_SPAN, rspan=ESS_RSPAN)
LABEL = dict(chi2="chi2      [invariant]", sigma="two-cfg sigma  [invariant]",
             rspan="span/|a|  [invariant]", span="raw span  [CARRIES UNITS]")
print(f"\nBEST NON-DEGENERATE universal power law, chosen SEPARATELY FOR EACH MEASURE\n"
      f"(the most generous reading of the power-law family):")
_hdr = f"  {'measure':<28} {'best p':>7} {'power law':>11} {'ESS':>11} {'ESS better by':>14}"
_rows = [_hdr]
for k in ("chi2", "sigma", "rspan", "span"):
    bp = min(OK, key=lambda q: MEAS[q][k])
    v = MEAS[bp][k]
    _rows.append(f"  {LABEL[k]:<28} {bp:>7.3f} {v:>11.2f} {ESSREF[k]:>11.2f} "
                 f"{v / ESSREF[k]:>13.2f}x")
TABLES["per_measure"] = "\n".join(_rows)
print(TABLES["per_measure"])
best = (min(OK, key=lambda q: MEAS[q]["span"]),)
print("\n  Read them together.  On the FULL grid the chi2 and sigma minima both\n"
      "  sit in the degenerate region -- p = 0.05 gives chi2 651.75 and 23.2 sigma against\n"
      "  ESS's 22.0, which quoted alone would read as 'barely distinguishable from ESS'\n"
      "  while the span at that same p is 305%, over four times ESS's.  That row is an\n"
      "  unidentified slope, not a competitive model.  The three INVARIANT measures put\n"
      "  ESS ahead by 1.33x to 1.67x here; the raw span agrees in direction but is the only\n"
      "  one of the four carrying units, so it is quoted last and never alone.")

# ---------------------------------------------------------------------------------------
# TIER 2 -- three free exponents, one per configuration
# ---------------------------------------------------------------------------------------
print(f"\n{'=' * 88}\nTIER 2: PER-CONFIGURATION exponents (three free parameters vs ESS's zero)")
own = []
for i, n in enumerate(names):
    cand = [(TAB[(i, p)][2], p) for p in PGRID if not degenerate(i, p)]
    c2i, pi = min(cand)
    own.append(pi)
    a, sa, c2, dof = TAB[(i, pi)]
    print(f"  {n:<34} own-best p = {pi:.3f}   ladder chi2/dof = {c2 / max(dof, 1):.2f}   "
          f"a = {a * 100:+.3f}% +/- {sa * 100:.3f}%")
vals = [TAB[(i, own[i])][0] * 100 for i in range(len(pres))]
errs = [TAB[(i, own[i])][1] * 100 for i in range(len(pres))]
c2o, co, spano = common_chi2(vals, errs)
sigo = two_cfg_sigma(vals, errs)
rso = rel_span(vals, errs)
print(f"\n  HONEST per-config (each p from that configuration's OWN ladder fit, blind to the\n"
      f"  cross-configuration statistic being reported):")
print(f"    span {spano:.2f}% (span/|a| = {rso:.3f}), chi2 = {c2o:.2f}, "
      f"two-config {sigo:.1f} sigma")
print(f"    ESS beats it by:  chi2 {c2o / ESS_C2:.2f}x [inv]   sigma {sigo / ESS_SIG:.2f}x "
      f"[inv]   span/|a| {rso / ESS_RSPAN:.2f}x [inv]   raw span {spano / ESS_SPAN:.2f}x "
      f"[units]")

# The BEST CASE for the per-config power law: exponents chosen to minimise the very statistic
# being compared.  This is NOT a legitimate fit -- it tunes to the answer -- and is reported
# only as an UPPER BOUND on what per-configuration exponents could possibly achieve.  If even
# this loses to ESS, no choice of three exponents can win.
grid = [[p for p in PGRID if not degenerate(i, p)] for i in range(len(pres))]
# EXACT minimum-span search, not a triple loop.  The span is max(a) - min(a) over one choice
# per configuration, so the optimum is the narrowest window on the merged, sorted list of
# candidate slopes that still covers every configuration at least once -- O(N log N) instead
# of O(N^3), which matters because the grid has to run far enough that the optimum stops
# sitting on its edge.
cand = sorted((TAB[(i, pp)][0] * 100, i, pp)
              for i in range(len(pres)) for pp in grid[i])
need, have, lo, bb = len(pres), {}, 0, None
for hi in range(len(cand)):
    have[cand[hi][1]] = have.get(cand[hi][1], 0) + 1
    while len(have) == need:
        width = cand[hi][0] - cand[lo][0]
        if bb is None or width < bb[1]:
            pick = {}
            for j in range(lo, hi + 1):
                pick.setdefault(cand[j][1], cand[j][2])
            ps = tuple(pick[i] for i in range(len(pres)))
            v = [TAB[(i, ps[i])][0] * 100 for i in range(len(pres))]
            e = [TAB[(i, ps[i])][1] * 100 for i in range(len(pres))]
            c2b, _, spanb = common_chi2(v, e)
            bb = (ps, spanb, c2b, two_cfg_sigma(v, e), rel_span(v, e))
        have[cand[lo][1]] -= 1
        if not have[cand[lo][1]]:
            del have[cand[lo][1]]
        lo += 1
edge = [i for i in range(len(pres)) if bb[0][i] >= max(PGRID) - 1e-9]
print(f"\n  BEST CASE, exponents tuned to minimise the reported span (an UPPER BOUND on what\n"
      f"  three exponents can do, not a legitimate fit -- it tunes to the answer):")
print(f"    p = {bb[0]}, span {bb[1]:.2f}% (span/|a| = {bb[4]:.3f}), chi2 = {bb[2]:.2f}, "
      f"two-config {bb[3]:.1f} sigma")
if bb[1] > ESS_SPAN:
    print(f"    ESS still beats it by:  chi2 {bb[2] / ESS_C2:.2f}x [inv]   sigma "
          f"{bb[3] / ESS_SIG:.2f}x [inv]   span/|a| {bb[4] / ESS_RSPAN:.2f}x [inv]   "
          f"raw span {bb[1] / ESS_SPAN:.2f}x [units]")
else:
    print(f"    and this BEATS ESS on span ({bb[1]:.2f}% vs {ESS_SPAN:.2f}%) -- but only by "
          f"tuning\n    three parameters to the statistic itself, so it bounds rather than "
          f"describes.")
if edge:
    print(f"    NOTE: the optimum sits ON THE GRID EDGE (p = {max(PGRID)}) for "
          f"{', '.join(names[i] for i in edge)}.\n"
          f"    The exponent is not identified for it -- struct b=0.8's slope is consistent\n"
          f"    with zero outright -- so the scan runs to whatever edge it is given and the\n"
          f"    bound is edge-limited.  Sensitivity measured, not assumed: see the grid-top\n"
          f"    scan recorded in the worklog.")

# and the weakest bound available: ignore the step flag entirely and let the "power law"
# degenerate into an M=1 indicator.  Reported because ESS wins under BOTH readings, which is
# a stronger statement than picking whichever range flatters the conclusion.
gridU = [[pp for pp in PGRID if not flat_degenerate(i, pp)] for i in range(len(pres))]
candU = sorted((TAB[(i, pp)][0] * 100, i, pp)
               for i in range(len(pres)) for pp in gridU[i])
have, lo, bu = {}, 0, None
for hi in range(len(candU)):
    have[candU[hi][1]] = have.get(candU[hi][1], 0) + 1
    while len(have) == len(pres):
        if bu is None or candU[hi][0] - candU[lo][0] < bu[1]:
            pick = {}
            for j in range(lo, hi + 1):
                pick.setdefault(candU[j][1], candU[j][2])
            ps = tuple(pick[i] for i in range(len(pres)))
            v = [TAB[(i, ps[i])][0] * 100 for i in range(len(pres))]
            e = [TAB[(i, ps[i])][1] * 100 for i in range(len(pres))]
            c2u, _, spanu = common_chi2(v, e)
            bu = (ps, spanu, c2u, two_cfg_sigma(v, e), rel_span(v, e))
        have[candU[lo][1]] -= 1
        if not have[candU[lo][1]]:
            del have[candU[lo][1]]
        lo += 1
print(f"\n  WEAKEST BOUND, step-degenerate exponents ALLOWED (p up to {max(PGRID)}, where the\n"
      f"  regressor is an M=1 indicator rather than a power law):")
print(f"    p = {bu[0]}, span {bu[1]:.2f}% (span/|a| = {bu[4]:.3f}), chi2 = {bu[2]:.2f}, "
      f"two-config {bu[3]:.1f} sigma")
print(f"    ESS beats even this by:  chi2 {bu[2] / ESS_C2:.2f}x [inv]   sigma "
      f"{bu[3] / ESS_SIG:.2f}x [inv]   span/|a| {bu[4] / ESS_RSPAN:.2f}x [inv]   "
      f"raw span {bu[1] / ESS_SPAN:.2f}x [units]")

# =======================================================================================
# TIER 3 -- how much of the answer is the THRESHOLDS?
#
# Every tier-1 optimum sits ON a flag boundary rather than at an interior minimum: chi2 and
# sigma optimise at the step cut, span/|a| at the error-inflation cut.  Each headline is
# therefore the value AT a threshold that was chosen, not measured, so the thresholds have
# to be swept rather than defended.  span/|a| is the most exposed of the four -- it is
# monotone in p (see `rel_span`), so its reported value is set almost entirely by where the
# admissible region starts.
# =======================================================================================
T2 = [("per-config p, honest (own ladder chi2)", c2o, sigo, rso, spano),
      ("per-config p, tuned to the span", bb[2], bb[3], bb[4], bb[1]),
      ("per-config p, step-degeneracy allowed", bu[2], bu[3], bu[4], bu[1])]
TABLES["tier2"] = "\n".join(
    [f"    {'tier':<38} {'chi2':>6} {'sigma':>7} {'span/|a|':>9} {'raw span':>9}"]
    + [f"    {nm:<38} {c / ESS_C2:>5.2f}x {sg / ESS_SIG:>6.2f}x {rs / ESS_RSPAN:>8.2f}x "
       f"{sp / ESS_SPAN:>8.2f}x" for nm, c, sg, rs, sp in T2])
print(f"\n  TIER 2, ALL FOUR MEASURES, ESS's advantage:")
print(TABLES["tier2"])

print(f"\n{'=' * 88}\nTIER 3: THRESHOLD SENSITIVITY (ESS's advantage, per measure)")
SWEEP = [f"  {'inflation cut':>14} {'step flag':>10} {'chi2':>8} {'sigma':>8} {'span/|a|':>9} "
         f"{'raw span':>9}   {'admissible p from':>18}"]
for infl in (1.2, 1.5, 2.0, 1e9):
    for step in (True, False):
        ok = [q for q in PGRID
              if not any(degenerate(i, q, infl, step) for i in range(len(pres)))]
        if not ok:
            continue
        row = {}
        for q in ok:
            v = [TAB[(i, q)][0] * 100 for i in range(len(pres))]
            e = [TAB[(i, q)][1] * 100 for i in range(len(pres))]
            c2, c, sp = common_chi2(v, e)
            row[q] = dict(chi2=c2, sigma=two_cfg_sigma(v, e), span=sp, rspan=rel_span(v, e))
        r = {k: min(row[q][k] for q in ok) / ESSREF[k]
             for k in ("chi2", "sigma", "rspan", "span")}
        lab = "none" if infl > 1e8 else f"{infl:.1f}x"
        SWEEP.append(f"  {lab:>14} {('on' if step else 'OFF'):>10} {r['chi2']:>7.2f}x "
              f"{r['sigma']:>7.2f}x {r['rspan']:>8.2f}x {r['span']:>8.2f}x   "
              f"{min(ok):>18.3f}")
TABLES["sweep"] = "\n".join(SWEEP)
print(TABLES["sweep"])
# the inflation of the unidentified winner, emitted rather than transcribed
_p0 = min(PGRID)
QUOTED["worst_infl_at_min_p"] = f"{max(TAB[(i, _p0)][1] / SD_MIN[i] for i in range(len(pres))):.1f}x"
print("  span/|a| runs 0.87x -> 2.52x across these cuts while chi2 holds 1.24x-1.63x and\n"
      "  sigma 1.06x-1.33x.  It is by far the most threshold-exposed of the four, NOT the\n"
      "  most robust, and 1.17x and 1.67x are the SAME calculation at two cut positions.\n"
      "  AND THE LAST ROW MATTERS: with no inflation cut at all, span/|a| INVERTS to 0.87x --\n"
      "  ESS loses on that measure — and sigma falls to a 1.06x near-tie.  The winner there\n"
      f"  is p = 0.05, whose slope is inflated {QUOTED['worst_infl_at_min_p']} over its own best and is not identified.\n"
      "  So the direction is NOT threshold-free: it holds at every cut that excludes\n"
      "  unidentified slopes, and fails only when they are admitted.  That is an argument\n"
      "  FOR the flag, not evidence the conclusion survives without one -- and it is the\n"
      "  reason the flag has to be justified on its own terms (§1) rather than by outcome.")

# =======================================================================================
# TIER 4 -- a sampling bar on the ratio itself
#
# None of the ratios above carries an error, and they are not noise-free: the slopes are
# measurements, so each common-slope chi2 is a noncentral statistic with real scatter.  The
# two chi2 CANNOT be perturbed independently -- they are two fits of the SAME d(m) data --
# so the bar is taken by perturbing at the DATA level and refitting both abscissae from the
# same draw, which carries the correlation through exactly.  The GLS slope is linear in the
# data, so this is a matrix product per draw and not a refit.
# =======================================================================================
def gls_slope_weights(C, xv, nblk):
    """(w, sigma_a) with slope = w . yv -- the GLS estimator written as a linear functional."""
    A = np.stack([xv ** k for k in range(2)], axis=1)
    Ci = np.linalg.pinv(C)
    k = len(xv)
    if nblk - k - 2 > 0:                                   # Hartlap, as in fit_ladder
        Ci = Ci * (nblk - k - 2) / (nblk - 1)
    cov = np.linalg.pinv(A.T @ Ci @ A)
    return (cov @ A.T @ Ci)[1], float(np.sqrt(cov[1, 1]))


print(f"\n{'=' * 88}\nTIER 4: SAMPLING BAR on the ratio (parametric bootstrap at the DATA level)")
BOOT, rng = 20000, np.random.default_rng(20260819)
P_C2 = min(OK, key=lambda q: MEAS[q]["chi2"])
P_RS = min(OK, key=lambda q: MEAS[q]["rspan"])
Wess, Wc2, Wrs, SD, L, base = [], [], [], [], [], []
for i, (D, reps, lad) in enumerate(pres):
    C = cc.jk_cov(reps)
    yv = D.mean(axis=0)
    xe = ex[i][0]
    we, sde = gls_slope_weights(C, xe, len(reps))
    wc, _ = gls_slope_weights(C, np.array([float(m) ** (-P_C2) for m in lad]), len(reps))
    wr, _ = gls_slope_weights(C, np.array([float(m) ** (-P_RS) for m in lad]), len(reps))
    Wess.append(we); Wc2.append(wc); Wrs.append(wr); SD.append(sde)
    L.append(np.linalg.cholesky(C + 1e-18 * np.eye(len(C)))); base.append(yv)
    # the linear form must reproduce the fitted slope it stands in for
    assert abs(float(we @ yv) - ess_v[i] / 100) < 1e-9, "GLS linear form does not match the fit"

T4, T4ROWS = {}, [f"    {'measure':<11} {'point':>5} {'bootstrap median':>19}"
                      f"   {'68% interval':>14} {'P(ESS worse)':>14}"]
E_ess = [TAB[(i, P_C2)][1] * 100 for i in range(len(pres))]      # errors are data-independent
rat_c2, rat_rs = [], []
for _ in range(BOOT):
    ae, ac, ar = [], [], []
    for i in range(len(pres)):
        z = rng.standard_normal(len(base[i]))
        yv = base[i] + L[i] @ z
        ae.append(float(Wess[i] @ yv) * 100)
        ac.append(float(Wc2[i] @ yv) * 100)
        ar.append(float(Wrs[i] @ yv) * 100)
    ee = [SD[i] * 100 for i in range(len(pres))]
    rat_c2.append(common_chi2(ac, [TAB[(i, P_C2)][1] * 100 for i in range(len(pres))])[0]
                  / max(common_chi2(ae, ee)[0], 1e-12))
    rat_rs.append(rel_span(ar, [TAB[(i, P_RS)][1] * 100 for i in range(len(pres))])
                  / max(rel_span(ae, ee), 1e-12))
for nm, r, pt in (("chi2", np.array(rat_c2), MEAS[P_C2]["chi2"] / ESS_C2),
                  ("span/|a|", np.array(rat_rs), MEAS[P_RS]["rspan"] / ESS_RSPAN)):
    lo, hi = np.percentile(r, [16, 84])
    nworse = int((r < 1).sum())
    pw4 = f"{nworse / len(r):.4f}" if nworse else f"< {1.0 / len(r):.5f}"
    T4[nm] = (f"{pt:.2f}x", f"[{lo:.2f}, {hi:.2f}]", f"{np.median(r):.2f}x", pw4,
              f"{(hi - lo) / 2:.3f}")
    T4ROWS.append(f"    {nm:<11} {pt:.2f}x {np.median(r):>13.2f}x        "
                  f"[{lo:.2f}, {hi:.2f}]  {pw4:>13}")
    print(f"  {nm:<9} point {pt:.2f}x   bootstrap median {np.median(r):.2f}x   "
          f"68% [{lo:.2f}, {hi:.2f}]   P(ESS worse) = {pw4}")
TABLES["tier4"] = "\n".join(T4ROWS)
QUOTED["t4_chi2_halfwidth"] = T4["chi2"][4]
QUOTED["t4_rspan_halfwidth"] = T4["span/|a|"][4]
QUOTED["t4_boot_draws"] = f"{BOOT:,}"
print(f"  {BOOT:,} draws, perturbing d(m) with the jackknife covariance and refitting BOTH\n"
      f"  abscissae from the SAME draw.  Perturbing the two chi2 independently would give a\n"
      f"  visibly wider and WRONG bar -- they share the data, and the correlation is most of\n"
      f"  the reason the ratio is better determined than either chi2 alone.")

# =======================================================================================
# TIER 5 -- the covariance's OWN error, the one unpropagated channel that can move the cut
#
# THE OBJECTION THIS PRE-EMPTS FIRST.  The admissible-p boundary looks like a data-dependent
# selection that ought to be re-derived inside every bootstrap draw of Tier 4.  It is not:
# for GLS with a fixed C, Var(a_hat) = (A' C^-1 A)^-1 depends on the DESIGN and the
# COVARIANCE only, never on the data values.  So the inflation ratio sd(p)/min_p sd(p) is
# identical in every Tier-4 draw and holding the boundary fixed there is exact, not a
# shortcut.  Pinned in `tests/test_catprior_ladder_fit.py`.
#
# BUT IT DOES DEPEND ON C, AND C IS ESTIMATED.  ~200 jackknife blocks give C roughly 10%
# per-element noise, which moves the slope errors, which moves the boundary -- and the Tier-3
# sweep shows the boundary is what the answer is most sensitive to.  A GLOBAL rescale of C
# cancels (it scales every sd equally and leaves the ratio alone), so only C's SHAPE noise
# matters.  That is measured here rather than named: draw C* ~ Wishart(C, df)/df, refit
# every slope and error from C*, re-derive the boundary from C*, and re-read the ratios.
# df = 199 is the generous count (jackknife replicates are not independent, so the true
# effective df is smaller and this UNDERSTATES the channel); df = 50 brackets it from the
# pessimistic side.
# =======================================================================================
def wishart(C, df, rng):
    """One draw of Wishart(df, C/df), mean C, via the Gaussian outer-product construction.

    Written out rather than taken from scipy: `sims1`'s scipy.stats cannot import on the
    login node (its `_highs` extension wants a newer GLIBCXX), and this needs one line.
    """
    A = np.linalg.cholesky(C / df + 1e-18 * np.eye(len(C)))
    Z = rng.standard_normal((len(C), df))
    return A @ Z @ Z.T @ A.T


print(f"\n{'=' * 88}\nTIER 5: COVARIANCE noise, propagated through the BOUNDARY")
PSUB = [q for q in PGRID if q <= 5.0]
CJ = [cc.jk_cov(pre[1]) for pre in pres]
YV = [pre[0].mean(axis=0) for pre in pres]
NB = [len(pre[1]) for pre in pres]
HW = {}
print(f"  {'df':>6} {'draws':>6} {'chi2 ratio':>22} {'span/|a| ratio':>24} {'cut p':>14}"
      f" {'P(ESS worse)':>14}")
for df, nd in ((199, 150), (50, 150)):
    rc, rr, cuts = [], [], []
    for _ in range(nd):
        Cs = [wishart(CJ[i], df, rng) for i in range(len(pres))]
        tab, sdmin = {}, {}
        for i in range(len(pres)):
            for q in PSUB:
                xq = np.array([float(m) ** (-q) for m in pres[i][2]])
                w, sd = gls_slope_weights(Cs[i], xq, NB[i])
                tab[(i, q)] = (float(w @ YV[i]), sd)
            sdmin[i] = min(tab[(i, q)][1] for q in PSUB)
        ok = [q for q in PSUB
              if not step_degenerate(q)
              and all(tab[(i, q)][1] <= DEGEN_INFLATE * sdmin[i] for i in range(len(pres)))]
        if not ok:
            continue
        # the ESS side must be refitted from the SAME C* -- otherwise the ratio mixes a
        # perturbed numerator with an unperturbed denominator and the spread is meaningless
        av = [float(gls_slope_weights(Cs[i], ex[i][0], NB[i])[0] @ YV[i]) * 100
              for i in range(len(pres))]
        ae = [gls_slope_weights(Cs[i], ex[i][0], NB[i])[1] * 100 for i in range(len(pres))]
        e_c2, e_rs = common_chi2(av, ae)[0], rel_span(av, ae)
        best_c2 = min(common_chi2([tab[(i, q)][0] * 100 for i in range(len(pres))],
                                  [tab[(i, q)][1] * 100 for i in range(len(pres))])[0]
                      for q in ok)
        best_rs = min(rel_span([tab[(i, q)][0] * 100 for i in range(len(pres))],
                               [tab[(i, q)][1] * 100 for i in range(len(pres))]) for q in ok)
        rc.append(best_c2 / e_c2); rr.append(best_rs / e_rs); cuts.append(min(ok))
    rc, rr, cuts = np.array(rc), np.array(rr), np.array(cuts)
    inv = int(((rc < 1) | (rr < 1)).sum())
    # RESOLUTION.  With n draws a zero count does not measure zero -- it bounds the rate at
    # 1/n.  Printing 0.000 would claim three digits the run cannot produce.
    pw = f"{inv / len(rc):.3f}" if inv else f"< {1.0 / len(rc):.3f}"
    print(f"  {df:>6} {len(rc):>6}   {np.median(rc):>5.2f}x [{np.percentile(rc, 16):.2f}, "
          f"{np.percentile(rc, 84):.2f}]   {np.median(rr):>5.2f}x "
          f"[{np.percentile(rr, 16):.2f}, {np.percentile(rr, 84):.2f}]   "
          f"{np.median(cuts):>5.3f} [{cuts.min():.3f}, {cuts.max():.3f}] {pw:>14}")
    QUOTED[f"t5_df{df}_chi2"] = f"{np.median(rc):.2f}x"
    QUOTED[f"t5_df{df}_chi2_ci"] = f"[{np.percentile(rc, 16):.2f}, {np.percentile(rc, 84):.2f}]"
    QUOTED[f"t5_df{df}_rspan"] = f"{np.median(rr):.2f}x"
    QUOTED[f"t5_df{df}_rspan_ci"] = f"[{np.percentile(rr, 16):.2f}, {np.percentile(rr, 84):.2f}]"
    QUOTED[f"t5_df{df}_cut"] = f"{np.median(cuts):.3f}"
    QUOTED[f"t5_df{df}_cut_range"] = f"[{cuts.min():.3f}, {cuts.max():.3f}]"
    QUOTED[f"t5_df{df}_p_ess_worse"] = pw
    POOL[0] += inv; POOL[1] += len(rc)
    HW[df] = ((np.percentile(rc, 84) - np.percentile(rc, 16)) / 2.0, float(np.median(rc)),
              float(np.percentile(rc, 2.5)), float(np.percentile(rc, 16)),
              float(np.percentile(rc, 84)))
QUOTED["p_ess_worse_pooled"] = (f"{POOL[0] / POOL[1]:.3f}" if POOL[0]
                                else f"< {1.0 / POOL[1]:.3f}")
QUOTED["pooled_draws"] = str(POOL[1])
CUT0 = min(OK)
QUOTED["fixedC_chi2"] = f"{MEAS[P_C2]['chi2'] / ESS_C2:.2f}x"
QUOTED["fixedC_rspan"] = f"{MEAS[P_RS]['rspan'] / ESS_RSPAN:.2f}x"
QUOTED["fixedC_cut"] = f"{CUT0:.3f}"
QUOTED["t4_chi2_ci"] = T4["chi2"][1]
QUOTED["t4_rspan_ci"] = T4["span/|a|"][1]
# hand-carried literals here would be the SAME failure class one level up from the worklog
print(f"  Read against the FIXED-C values (chi2 {QUOTED['fixedC_chi2']}, span/|a| "
      f"{QUOTED['fixedC_rspan']}, cut {QUOTED['fixedC_cut']}).  Two things\n"
      "  come out of this, and the second overturns a Tier-4 reading:\n"
      "    * the CUT ITSELF is what moves -- median ~0.61, spanning 0.275 to 1.450 across\n"
      "      the two brackets -- so C's shape noise is exactly the channel Tier 3 predicted;\n"
      "    * span/|a|'s Tier-4 bar of +/-2% is therefore ILLUSORY.  Propagating C widens it\n"
      f"      to {QUOTED['t5_df199_rspan_ci']} at df = 199, about ten times wider, because Tier 4 holds the cut\n"
      "      fixed and span/|a| is the measure most sensitive to where the cut falls.  chi2\n"
      f"      widens far less ({QUOTED['t4_chi2_ci']} -> {QUOTED['t5_df199_chi2_ci']}), which is the same\n"
      "      threshold-stability that makes it the right headline.\n"
      "  NOT additive with Tier 4: both perturb the same fit, one through the data and one\n"
      "  through the covariance, so they are quoted separately rather than combined.")
# HOW BAD WOULD THE EFFECTIVE df HAVE TO BE?  There is no principled effective df for a
# jackknife covariance (the replicates are correlated, so 199 is optimistic), but it does not
# have to be derived -- it can be bounded out of the way.  The CI half-width scales as
# 1/sqrt(df), so extrapolate to where the 68% interval would first touch 1.0x.
print("\n  EFFECTIVE df: BOUNDED RATHER THAN DERIVED, UNDER BOTH EXTRAPOLATION CONVENTIONS.")
print("    The 1/sqrt(df) law fixes how the interval GROWS but not what it grows ON.  Additive\n"
      "    (linear) and multiplicative (log) readings of the same band give different answers,\n"
      "    and the choice is not free -- the skew measured below is evidence about which.")
DFB = {}
for df, (h, med, lo2, p16, p84) in HW.items():
    hlog = (np.log(p84) - np.log(p16)) / 2.0
    lin, lg = df * (h / (med - 1.0)) ** 2, df * (hlog / np.log(med)) ** 2
    DFB[df] = (lin, lg, med - 2 * h, med * np.exp(-2 * hlog), lo2)
    QUOTED[f"df{df}_touch_linear"] = f"{lin:.0f}"
    QUOTED[f"df{df}_touch_log"] = f"{lg:.0f}"
    QUOTED[f"df{df}_lo2_empirical"] = f"{lo2:.2f}x"
    QUOTED[f"df{df}_lo2_sym_linear"] = f"{med - 2 * h:.2f}x"
    QUOTED[f"df{df}_lo2_sym_log"] = f"{med * np.exp(-2 * hlog):.2f}x"
    print(f"    from df = {df:>3}: chi2 median {med:.2f}x, half-width {h:.3f} additive /"
          f" {hlog:.3f} in log;\n"
          f"      the 68% band first touches 1.0x at df ~ {lin:.0f} (linear)"
          f" / df ~ {lg:.0f} (log)")
print("    Both conventions are internally consistent across the two brackets"
      f" ({DFB[199][0]:.0f} vs {DFB[50][0]:.0f}\n"
      f"    linear, {DFB[199][1]:.0f} vs {DFB[50][1]:.0f} log), so neither is failing"
      " -- they simply differ in the assumed shape.")
print("\n    WHICH CONVENTION?  THE SKEW ANSWERS IT.  Three readings of the same 2-sigma lower\n"
      "    edge, and the empirical one is the arbiter because it assumes no shape at all:")
for df, (lin, lg, sym_lin, sym_log, emp) in DFB.items():
    print(f"      df = {df:>3}:  symmetric-linear {sym_lin:.2f}x   symmetric-log {sym_log:.2f}x"
          f"   EMPIRICAL PERCENTILE {emp:.2f}x")
print("    The percentile sits ABOVE both symmetric forms and much nearer the log one at each\n"
      "    bracket, which is what right-skew implies: a distribution right-skewed on a linear\n"
      "    scale is closer to symmetric on a log scale.  So the MULTIPLICATIVE extrapolation is\n"
      "    the better-motivated one, and quoting the linear ~17 alone would be a\n"
      "    conservative-LOOKING choice this run\'s own skew evidence does not support.\n"
      "    Adopted from the peer, whose log figures reproduce here exactly; the earlier\n"
      "    fractional-half-width arithmetic they first offered was a separate, real slip.")
print("    Scaling check: the two half-widths differ by "
      f"{HW[50][0] / HW[199][0]:.2f}x against the sqrt(199/50) = 1.99 the 1/sqrt(df) law\n"
      "    predicts, so the extrapolation is running on a law the two brackets confirm.")
DF_LO, DF_HI = min(DFB[199][1], DFB[50][1]), max(DFB[199][0], DFB[50][0])
QUOTED["df_bound_range"] = f"~{DF_LO:.0f}-{DF_HI:.0f}"
QUOTED["df_degradation_range"] = f"~{200 / DF_HI:.0f}-{200 / DF_LO:.0f}"
QUOTED["hw_scaling_ratio"] = f"{HW[50][0] / HW[199][0]:.2f}x"
print(f"    SO THE ADVANTAGE SURVIVES unless 200 jackknife blocks carry less information\n"
      f"    about the covariance than ~{DF_LO:.0f}-{DF_HI:.0f} independent realisations would -- log and\n"
      f"    linear conventions respectively, a factor ~{200 / DF_HI:.0f}-{200 / DF_LO:.0f} degradation.  THE CONCLUSION IS\n"
      f"    UNCHANGED ACROSS THE WHOLE RANGE, so nothing rests on picking one, and the range\n"
      f"    is what gets quoted.  Correlated replicates make 199 optimistic, but not by that\n"
      f"    much.  This converts the missing effective df from an open limitation into a\n"
      f"    bound nobody has to resolve.\n"
      f"    THE 2-SIGMA READING IS LESS COMFORTABLE and is quoted rather than buried: at the\n"
      f"    pessimistic bracket the advantage is 1-sigma secure but only marginally 2-sigma\n"
      f"    secure.  The ORDERING is robust; '1.6x' as a MAGNITUDE is not 2-sigma robust\n"
      f"    there, which is the same conclusion as everywhere else -- do not quote the\n"
      f"    partial to two figures.")
print("\n  ENDPOINT NOISE.  At 150 draws the 68% endpoints themselves carry ~6%, so comparing\n"
      "  this chi2 interval against the fixed-cut one is a comparison INSIDE that noise and\n"
      "  nothing should be read into the small difference.  The finding that survives it is\n"
      "  the span/|a| widening, which is a factor of ten.")
print("\n  UNPROPAGATED, AND THE DIRECTION IS KNOWN.  <1/ESS> is measured with noise while M^-p\n"
      "  is exact, since M is known.  Noise in a REGRESSOR adds scatter to the fitted relation\n"
      "  and inflates chi2, so propagating it would LOWER ESS's chi2 and leave the power law's\n"
      "  untouched -- widening ESS's lead.  Omitting it is therefore conservative for the\n"
      "  claim being made, which turns this limitation into a one-sided bound rather than an\n"
      "  open risk.  The frozen-weight choice is bounded separately in cont.190 §4.")

# ============================================================================================
# THE REJECTION, WHICH IS THE FIRST-ORDER RESULT AND NEEDS NONE OF THE SCAFFOLDING ABOVE.
# Adopted from the peer.  Everything from Tier 3 on characterises a MARGIN between two models
# that are BOTH refuted -- and the margin is what needed thresholds, conventions, bootstraps
# and a covariance channel to pin down.  The rejection needed none of them: it is the same
# under every convention tested, because chi2 on 2 dof of this size is not a close call.
# Prominence should follow what is load-bearing, not where the work went.
# ============================================================================================
# scattered figures section 4 quotes, emitted rather than transcribed
def measures_at(q):
    v = [TAB[(i, q)][0] * 100 for i in range(len(pres))]
    e = [TAB[(i, q)][1] * 100 for i in range(len(pres))]
    c2, _, sp = common_chi2(v, e)
    return dict(chi2=c2, sigma=two_cfg_sigma(v, e), span=sp, rspan=rel_span(v, e))

QUOTED["abscissa_min_ess_nbr"] = f"{min(ex[0][0]):.3f}"
QUOTED["abscissa_min_ess_struct"] = f"{min(ex[1][0]):.3f}"
QUOTED["abscissa_min_invM"] = f"{min(float(m) ** -1.0 for m in pres[0][2]):.3f}"
# the peer's proposed restriction p >= 0.3, evaluated over the FULL grid (not just the
# admissible set) because the point of quoting it is that it ADMITS degenerate exponents
_r30 = [q for q in PGRID if q >= 0.3 - 1e-9]
_M30 = {q: measures_at(q) for q in _r30}
QUOTED["p30_chi2"] = f"{min(_M30[q]['chi2'] for q in _r30) / ESS_C2:.2f}x"
QUOTED["p30_sigma"] = f"{min(_M30[q]['sigma'] for q in _r30) / ESS_SIG:.2f}x"
QUOTED["p30_span"] = f"{min(_M30[q]['span'] for q in _r30) / ESS_SPAN:.2f}x"
_v030 = measures_at(_r30[0])["rspan"] / ESS_RSPAN
QUOTED["at_p030_rspan"] = f"{_v030:.2f}x"
QUOTED["at_p030_rspan_4dp"] = f"{_v030:.4f}"
QUOTED["at_p030_inflation"] = f"{max(TAB[(i, 0.3)][1] / SD_MIN[i] for i in range(len(pres))):.2f}x"
_marg = ([c / ESS_C2 for _, c, _, _, _ in T2] + [sg / ESS_SIG for _, _, sg, _, _ in T2]
         + [rs / ESS_RSPAN for _, _, _, rs, _ in T2] + [sp / ESS_SPAN for _, _, _, _, sp in T2]
         + [MEAS[P_C2]["chi2"] / ESS_C2, MEAS[P_RS]["rspan"] / ESS_RSPAN])
QUOTED["margin_range"] = f"{min(_marg):.2f}x-{max(_marg):.2f}x"

# --------------------------------------------------------------------------------------
# SECTIONS 1-3 AUDIT.  The peer's point: the covered sections accumulated EIGHT stale
# literals under "care, no check", and sections 1-3 have had exactly that treatment for
# exactly as long -- so the honest prior is that they contain some too.  A one-time audit
# would answer it once; emitting the numbers answers it permanently, at the cost of a few
# lines.  The grid scan behind section 3 is the only expensive part and is gated.
# --------------------------------------------------------------------------------------
_NBR, _SB08 = 0, 2
_pmin, _pmax = min(PGRID), 1.0
for _tag, _q in (("p1", _pmax), ("pmin", _pmin)):
    _a, _sd = TAB[(_NBR, _q)][0], TAB[(_NBR, _q)][1]
    QUOTED[f"s1_nbr_{_tag}_a"] = f"{_a * 100:.2f}%"
    QUOTED[f"s1_nbr_{_tag}_sd"] = f"{_sd * 100:.3f}%"
    QUOTED[f"s1_nbr_{_tag}_frac"] = f"{_sd / abs(_a) * 100:.1f}%"
    QUOTED[f"s1_xrange_{_tag}"] = f"{XRANGE[(_NBR, _q)]:.2f}"
QUOTED["s1_nbr_inflation"] = f"{TAB[(_NBR, _pmin)][1] / TAB[(_NBR, _pmax)][1]:.1f}x"
_a8, _sd8 = TAB[(_SB08, _pmax)][0], TAB[(_SB08, _pmax)][1]
_ae8, _sde8 = slope(pres[_SB08], x=ex[_SB08][0])[0], slope(pres[_SB08], x=ex[_SB08][0])[1]
ESS_SB08_FRAC = _sde8 / abs(_ae8)
QUOTED["s1_sb08_a_ess"] = f"{_ae8 * 100:+.1f}%"
QUOTED["s1_sb08_sd_ess"] = f"{_sde8 * 100:.1f}%"
QUOTED["s1_sb08_sigma_ess"] = f"{abs(_ae8) / _sde8:.1f}"
QUOTED["s1_sb08_a"] = f"{_a8 * 100:+.1f}%"
QUOTED["s1_sb08_sd"] = f"{_sd8 * 100:.1f}%"
QUOTED["s1_sb08_sigma"] = f"{abs(_a8) / _sd8:.1f}"
_fr = np.array([[TAB[(j, q)][1] / abs(TAB[(j, q)][0]) for q in PGRID]
                for j in range(len(pres))])
_ps = np.array(PGRID, dtype=float)
QUOTED["s1_sb08_frac_med_lowp"] = f"{np.median(_fr[_SB08][_ps <= 3.0]) * 100:.0f}%"
# WHAT THE 50% CUT ACTUALLY DOES, on the grid as ORIGINALLY scanned and as it stands now.
# This is the claim section 1 rests on, and it is grid-dependent -- which is exactly how it
# went stale when section 3 extended the top from 3.0 to 12.0.
_struck = (_fr > 0.5).any(axis=0)
_step = np.array([step_degenerate(q) for q in PGRID])
for _top in (3.0, 12.0):
    _m = _ps <= _top + 1e-9
    QUOTED[f"s1_frac_cut_keeps_top{_top:g}"] = str(int((~_struck & _m).sum()))
    QUOTED[f"s1_frac_cut_keeps_nonstep_top{_top:g}"] = str(int((~_struck & _m & ~_step).sum()))
QUOTED["s1_frac_cut_grid_rows"] = str(int(_m.sum()))
_keep = ~_struck & ~_step
QUOTED["s1_frac_cut_window_lo"] = f"{_ps[_keep].min():.3f}"
QUOTED["s1_frac_cut_window_hi"] = f"{_ps[_keep].max():.3f}"
# THE SURVIVORS ARE NOT STRUCK BY THE STEP FLAG -- they sit just BELOW its boundary, so
# section 1 does NOT get to lean on section 2 here.  Recording the per-config errors that
# let them through, and how the count moves with STEP_MIN, since a rescue that depended on
# that threshold would be threshold-shopping.
QUOTED["s1_survivor_worst_frac"] = f"{_fr[:, _keep].max(axis=0).max() * 100:.0f}%"
QUOTED["s1_survivor_sb08_lo"] = f"{_fr[_SB08][_keep].min() * 100:.0f}%"
QUOTED["s1_survivor_sb08_hi"] = f"{_fr[_SB08][_keep].max() * 100:.0f}%"
for _sm in (0.01, 0.10):
    _b = np.log(1.0 / _sm) / np.log(2.0)
    QUOTED[f"s1_survivors_if_stepmin_{_sm:g}"] = str(int((~_struck & (_ps <= _b)).sum()))
QUOTED["s1_sb08_frac_ess"] = f"{ESS_SB08_FRAC * 100:.0f}%"
# FLUSH, NOT MERELY BELOW: the survivor window's top is the LAST grid point under the step
# boundary, so there is no empirical seam in the data to anchor a threshold on -- which is
# why the survivor count reads out the threshold rather than the configurations, and why
# STEP_MIN has to stand definitionally.
QUOTED["s1_grid_step"] = f"{PGRID[1] - PGRID[0]:.3f}"
QUOTED["s1_survivor_gap_to_step"] = f"{np.log(1.0 / STEP_MIN) / np.log(2.0) - _ps[_keep].max():.4f}"
QUOTED["s2_step_p"] = f"{np.log(1.0 / STEP_MIN) / np.log(2.0):.2f}"
QUOTED["s2_step_min_pct"] = f"{STEP_MIN * 100:.0f}%"

REJ = [("ESS (zero fitted parameters)", ESS_C2),
       ("best universal power law", MEAS[P_C2]["chi2"]),
       ("1/M (the nominal expectation)", MEAS[1.0]["chi2"])]
print(f"\n{'=' * 88}\nTHE REJECTION -- the result that survives every convention in this script")
rows = []
for nm, c2 in REJ:
    # chi2 survival on 2 dof is exactly exp(-x/2); no scipy needed, and none available here
    log10p = -c2 / 2.0 / np.log(10.0)
    # NB: not `ex` -- that name holds the ESS abscissae and rebinding it here silently
    # broke every later reader of it.  Caught by the section 1-3 audit, not by a test.
    mant, expo = 10.0 ** (log10p % 1), int(np.floor(log10p))
    rows.append(f"    {nm:<31} chi2 = {c2:8.2f} on 2 dof   p = {mant:.1f}e{expo}")
QUOTED["rej_ess_chi2"] = f"{ESS_C2:.2f}"
QUOTED["rej_powerlaw_chi2"] = f"{MEAS[P_C2]['chi2']:.2f}"
QUOTED["rej_invM_chi2"] = f"{MEAS[1.0]['chi2']:.2f}"
TABLES["rejection"] = "\n".join(rows)
print("\n".join(rows))
print("  EVERY member of the family is rejected, by more than a hundred orders of magnitude.\n"
      "  The DELIVERABLE SENTENCE is therefore the rejection -- no common-slope law in this\n"
      "  family describes the three configurations -- with the margin as SECONDARY\n"
      "  characterisation: ESS is the least-bad member by a factor 1.2-2.3 depending on\n"
      "  analysis choices.  A soft second-order number attached to a hard first-order result\n"
      "  is a normal shape; a soft number carrying the claim alone is not.")

print(f"\n{'=' * 88}\nVERDICT")
print("  HEADLINE ON chi2, NOT ON span/|a|.  chi2 is invariant AND threshold-stable\n"
      "  (1.24x-1.63x across the whole Tier-3 sweep).  span/|a| is invariant but NOT\n"
      "  threshold-stable (0.87x-2.52x), and its bootstrap bar is the TIGHTEST of the four\n"
      "  (+/-2%) -- so the measure that looks most precise is the one whose value is most\n"
      "  determined by a choice.  Quoting '1.67x +/- 0.035' without the sweep beside it\n"
      "  would be among the more misleading true statements available here.  span/|a| is\n"
      "  therefore reported ONLY in the sweep table, where its spread is visible.")
bs = MEAS[best[0]]["span"]
print(f"  universal p:   best non-degenerate span {bs:.2f}% vs ESS {ESS_SPAN:.2f}% "
      f"({bs / ESS_SPAN:.2f}x); the three INVARIANT measures give "
      f"{MEAS[P_C2]['chi2'] / ESS_C2:.2f}x (chi2), "
      f"{min(MEAS[q]['sigma'] for q in OK) / ESS_SIG:.2f}x (sigma), "
      f"{MEAS[P_RS]['rspan'] / ESS_RSPAN:.2f}x (span/|a|)")
print(f"  per-config p:  honest span {spano:.2f}% ({spano / ESS_SPAN:.2f}x), "
      f"best-case bound {bb[1]:.2f}% ({bb[1] / ESS_SPAN:.2f}x),"
      f"\n                 weakest bound {bu[1]:.2f}% ({bu[1] / ESS_SPAN:.2f}x)")
print("  ESS wins under every reading THAT EXCLUDES UNIDENTIFIED SLOPES -- every tier, every\n"
      "  summary, every inflation cut from 1.2x to 2.0x, with or without the step flag.  It\n"
      "  does NOT win with no cut at all (Tier 3's last row), where an unidentified p = 0.05\n"
      "  takes span/|a| by 0.87x.  A THREE-parameter fitted family does not reach a\n"
      "  ZERO-parameter measured quantity: the per-rung structure is doing work no power law\n"
      "  reproduces.  THREE kinds of uncertainty, none substituting for another: the spread\n"
      "  over analysis choices (Tier 3, the widest), the covariance's own error acting\n"
      "  through the cut (Tier 5, comparable), and the data sampling bar at fixed choices\n"
      "  (Tier 4, the narrowest).  Quote the range, not a number with a bar.")
if abs(best[0] - max(OK)) < 1e-9:
    print("\n  AND NOTE WHERE THE OPTIMA SIT.  Both tiers push the exponent right up against\n"
          "  the step boundary and are held there only by the flag: what actually helps the\n"
          "  power-law family is not an exponent at all but permission to treat M=1 as\n"
          "  special.  The family's only route to competitiveness is to stop being a power\n"
          "  law -- which is the conclusion stated a second way, not a separate finding.")



# --------------------------------------------------------------------------------------
# SECTION 3'S GRID-TOP SCAN, RUN RATHER THAN REMEMBERED.  The bound is edge-limited --
# struct b=0.8's exponent is not identified, so the tuned best case runs to whatever edge
# the grid provides -- and the worklog answers that with a scan across grid tops.  Until
# now those four columns were typed once and carried by hand, which is exactly the class of
# statement that broke section 1: the grid top is the very dimension whose change
# invalidated it.  So the scan runs here, on its own grid, and is emitted.
#
# It uses a SEPARATE grid reaching past PGRID's top rather than extending PGRID, because
# extending PGRID would move `max(PGRID)` and with it the weakest-bound region of section
# 4b -- a sensitivity probe must not change the thing it is probing.  Its degeneracy
# reference SD_MIN is recomputed over its own wider grid, since "inflated against this
# configuration's own best" is a statement about the grid it is asked on.  The top = 12.0
# column reproducing the main analysis exactly is asserted below, which is what makes the
# two grids' conventions comparable rather than merely adjacent.
SCAN_TOPS = (3.0, 6.0, 12.0, 24.0)
_STEP = round(PGRID[1] - PGRID[0], 3)
_SCANG = [round(x, 3) for x in np.arange(min(PGRID), max(SCAN_TOPS) + 1e-9, _STEP)]
_ST = {}
for _q in _SCANG:
    for _i in range(len(pres)):
        _ST[(_i, _q)] = (TAB[(_i, _q)] if (_i, _q) in TAB
                         else slope(pres[_i], x=np.array([float(m) ** (-_q)
                                                          for m in pres[_i][2]])))
_SDM = {i: min(_ST[(i, q)][1] for q in _SCANG) for i in range(len(pres))}


def _narrowest_window(gr):
    """Exact minimum-span choice of one exponent per configuration; same search as tier 2."""
    c = sorted((_ST[(i, q)][0] * 100, i, q) for i in range(len(pres)) for q in gr[i])
    have, lo, out = {}, 0, None
    for hi in range(len(c)):
        have[c[hi][1]] = have.get(c[hi][1], 0) + 1
        while len(have) == len(pres):
            w = c[hi][0] - c[lo][0]
            if out is None or w < out[0]:
                pick = {}
                for j in range(lo, hi + 1):
                    pick.setdefault(c[j][1], c[j][2])
                ps = tuple(pick[i] for i in range(len(pres)))
                vv = [_ST[(i, ps[i])][0] * 100 for i in range(len(pres))]
                ee = [_ST[(i, ps[i])][1] * 100 for i in range(len(pres))]
                out = (common_chi2(vv, ee)[2], ps)
            have[c[lo][1]] -= 1
            if not have[c[lo][1]]:
                del have[c[lo][1]]
            lo += 1
    return out


def scan_at(top):
    """(best-case span, honest span) in units of ESS's span, on a grid capped at `top`."""
    gr = [[q for q in _SCANG
           if q <= top + 1e-9 and not step_degenerate(q)
           and _ST[(i, q)][1] <= DEGEN_INFLATE * _SDM[i]]
          for i in range(len(pres))]
    ow = [min((_ST[(i, q)][2], q) for q in gr[i])[1] for i in range(len(pres))]
    vv = [_ST[(i, ow[i])][0] * 100 for i in range(len(pres))]
    ee = [_ST[(i, ow[i])][1] * 100 for i in range(len(pres))]
    return _narrowest_window(gr)[0] / ESS_SPAN, common_chi2(vv, ee)[2] / ESS_SPAN


SCAN = {t: scan_at(t) for t in SCAN_TOPS}
_rows = ["    " + "grid top".ljust(12) + "".join(f"{t:>8.1f}" for t in SCAN_TOPS),
         "    " + "best-case".ljust(12) + "".join(f"{SCAN[t][0]:>7.2f}x" for t in SCAN_TOPS),
         "    " + "honest".ljust(12) + "".join(f"{SCAN[t][1]:>7.2f}x" for t in SCAN_TOPS)]
TABLES["gridtop"] = "\n".join(_rows)
print(f"\n{'=' * 88}\nGRID-TOP SENSITIVITY (the bound is edge-limited; measured, not assumed)")
print(TABLES["gridtop"])
# THE CONVENTION CHECK.  If the scan grid's own degeneracy reference disagreed with the main
# analysis's, the columns would not be comparable to anything else in the entry.  It does not:
_main = (bb[1] / ESS_SPAN, spano / ESS_SPAN)
assert all(abs(a - b) < 5e-3 for a, b in zip(SCAN[12.0], _main)), (SCAN[12.0], _main)
QUOTED["s3_grid_top_used"] = f"{max(PGRID):.1f}"
QUOTED["s3_plateau_spread"] = (f"{abs(SCAN[24.0][0] - SCAN[12.0][0]):.2f}")
print(f"  top = 12.0 reproduces the main analysis ({_main[0]:.2f}x, {_main[1]:.2f}x), so the\n"
      f"  scan's wider grid has not changed the convention it is scanning.  12.0 to 24.0\n"
      f"  moves the best case by {QUOTED['s3_plateau_spread']}x, so 12.0 is inside the plateau.")


# ============================================================================================
# EMITTED BLOCK.  Three times now a statement that was correct WHEN WRITTEN silently stopped
# being true and nothing errored: a grouping key that outlived the dimension it assumed, a
# docstring describing a formula the body no longer computed, and prose literals from a
# 40-draw run surviving into a 150-draw section.  The first was closed permanently by
# assert_group_is_replicates.  This is the PROSE version of the same class, and it is the one
# most likely to reach a reader, because prose is what gets quoted.
#
# So the numbers doc/WORKLOG.md is allowed to quote are EMITTED here rather than transcribed
# by hand, and tests/test_worklog_quotes_current_numbers.py asserts every emitted value still
# appears in the section that quotes it.  A re-run that moves a number then either forces the
# prose to be updated or fails the suite -- instead of leaving a section that reads correctly
# and is wrong.  The check is a tripwire, not a proof: it cannot tell a value used in the
# right place from the same digits appearing elsewhere.  It catches drift, which is the
# failure that actually happened.
# ============================================================================================
EMIT = pathlib.Path(__file__).resolve().parents[1] / "doc" / "generated" / "ess_exponent_control.numbers"
EMIT.parent.mkdir(parents=True, exist_ok=True)
QUOTED["ess_chi2_ratio"] = f"{MEAS[P_C2]['chi2'] / ESS_C2:.2f}x"
QUOTED["ess_rspan_ratio"] = f"{MEAS[P_RS]['rspan'] / ESS_RSPAN:.2f}x"
# THE BLOCKS, NOT JUST THE VALUES.  A bare value check is too weak: "1.49x" is a stale
# Tier-5 median AND a live grid-scan figure elsewhere in the same entry, so searching the
# entry for it passes on drift.  Numbers that travel together are therefore emitted as the
# EXACT lines doc/WORKLOG.md carries, and compared verbatim -- which cannot be fooled by a
# coincidence, because the coincidence would have to be the whole row.
TABLES.update({
    "tier5": "\n".join(
        ["      df   chi2 ratio           span/|a| ratio        cut p                  P(ESS worse)"]
        + [f"     {df:>3}   {QUOTED[f't5_df{df}_chi2']} {QUOTED[f't5_df{df}_chi2_ci']:<12}   "
           f"{QUOTED[f't5_df{df}_rspan']} {QUOTED[f't5_df{df}_rspan_ci']:<12}   "
           f"{QUOTED[f't5_df{df}_cut']} {QUOTED[f't5_df{df}_cut_range']:<16} "
           f"{QUOTED[f't5_df{df}_p_ess_worse']:>7}" for df in (199, 50)]),
    "df_touch": "\n".join(
        ["                        df = 199    df = 50",
         f"    LINEAR (additive)    {QUOTED['df199_touch_linear']:>7}    {QUOTED['df50_touch_linear']:>7}",
         f"    LOG (multiplicative) {QUOTED['df199_touch_log']:>7}    {QUOTED['df50_touch_log']:>7}"]),
    "lo2": "\n".join(
        ["               symmetric-linear   symmetric-log   EMPIRICAL PERCENTILE",
         f"    df = 199        {QUOTED['df199_lo2_sym_linear']}             {QUOTED['df199_lo2_sym_log']}              {QUOTED['df199_lo2_empirical']}",
         f"    df =  50        {QUOTED['df50_lo2_sym_linear']}             {QUOTED['df50_lo2_sym_log']}              {QUOTED['df50_lo2_empirical']}"]),
})
(EMIT.parent / "ess_exponent_control.tables").write_text(
    "# GENERATED by scripts/ess_exponent_control.py -- do not hand-edit.\n"
    "# Each block below appears VERBATIM in doc/WORKLOG.md cont.191; the suite compares them.\n"
    + "".join(f"<<<{k}\n{v}\n>>>\n" for k, v in TABLES.items()))
body = "".join(f"{k} = {v}\n" for k, v in sorted(QUOTED.items()))
EMIT.write_text("# GENERATED by scripts/ess_exponent_control.py -- do not hand-edit.\n"
                "# Every value here is quoted by doc/WORKLOG.md cont.191; the suite checks it.\n"
                + body)
print(f"\n{'=' * 88}\nEMITTED {len(QUOTED)} quotable numbers -> {EMIT}")
print(body, end="")
