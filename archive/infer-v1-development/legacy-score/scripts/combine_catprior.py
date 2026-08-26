#!/usr/bin/env python -B
# Archived Infer V1 development diagnostic.
"""Combine legacy `eval_score_catprior.py` finite-M study runs.

This is a reproducibility tool for the superseded score/quadrature experiment,
not a summarizer for the current finite-scene likelihood.  Current profiles use
``summarize_catalogue_closure.py`` and ``summarize_catalogue_bias.py``.

WHY THIS IS NOT JUST A CONCATENATION.  Inside one run the draw-count ladder is nested, so
consecutive points share every draw the shorter one used.  That pairing is what makes
`d(m)` precise, and it is also what makes a single run's spread across `M` useless as an
uncertainty: the points are not independent, and a smooth-looking curve through them is
guaranteed by construction rather than earned.  `pass_fraction_by_node`'s docstring records
the same trap costing a 34-sigma detection of a quantity that is zero by symmetry.

The fix is the one that always works: repeat the QUADRATURE, not the rows.  Runs that share
`--seed` and `--flow-seed` but differ in `--draw-seed` see identical galaxies, identical
true shapes and identical `xhat`, and differ only in which catalogue rows were drawn.  Their
scatter at fixed `M` is therefore exactly the finite-draw uncertainty, with no galaxy noise
in it at all -- and the `pin` column, which does not depend on the draws, must come back
bit-identical across them, which this script checks rather than assumes.

Usage:
    python scripts/combine_catprior.py /path/to/catprior_*.npz
"""

import argparse
import itertools
import glob
import json
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbsi.score_inference import jackknife_blocks, jackknife_sigma  # noqa: E402


# ======================================================================================
# WHAT MAKES TWO RUNS THE SAME RUN
#
# THIS KEY IS AN ALLOWLIST OF WHAT MAY DIFFER, NOT A LIST OF WHAT MUST MATCH.  The earlier
# form enumerated the fields to include, which made every NEW config field poolable by
# default until a human remembered to add it.  That failed three times in one session, all
# with the same signature: runs that differ get averaged, the output is well-formed, the
# tests pass, and nothing errors --
#   * `pool_half` missing      -> arms A and B averaged into one group, erasing the very
#                                 difference the run exists to measure;
#   * `pool_split_seed` missing-> several splits of one half pooled as if they were draw
#                                 seeds, relabelling pool scatter as seed scatter;
#   * (and the same shape in the output PATH, where arm B overwrote arm A).
# Inverting it makes a new field safe BY DEFAULT: it enters the key automatically, and the
# only way to pool over an axis is to declare it here on purpose.  Instance four is now not
# merely findable but impossible.
REPLICATE_AXES = ("draw_seed",)
# Fields that genuinely do not identify the measurement.  Keep this set as small as it can
# be: everything omitted here is treated as identifying, which is the safe direction.
IDENTITY_IGNORE = ("save",)


def run_identity(run):
    """`(key, axis)`: everything that identifies this run, and its replicate-axis values.

    `key` deliberately spans EVERY config field except the declared replicate axes and
    `IDENTITY_IGNORE`, so a field added to the eval script later is identifying without
    anyone having to remember it here.
    """
    cfg = run["cfg"]
    skip = set(REPLICATE_AXES) | set(IDENTITY_IGNORE)
    key = tuple(sorted((k, str(v)) for k, v in cfg.items() if k not in skip))
    key += (("__features", ",".join(run["features"])), ("__ladder", str(run["ladder"])))
    return key, tuple(str(cfg.get(a)) for a in REPLICATE_AXES)


def assert_group_is_replicates(rs):
    """Fail loudly if a group is not what pooling assumes: replicates along the axis alone.

    Duplicate values on the replicate axis mean the group was formed by an identifying field
    going missing from the key -- the exact failure mode above, whose ONLY visible symptom
    last time was a header reading `draw seeds [99, 99, 99, 99, 99, 99]`.  Six identical
    seeds is not a plausible replicate set, so assert it rather than print it.
    """
    axes = [run_identity(r)[1] for r in rs]
    if len(set(axes)) != len(axes):
        dup = sorted({a for a in axes if axes.count(a) > 1})
        raise SystemExit(
            f"REFUSING TO POOL: {len(rs)} runs in one group share replicate-axis "
            f"value(s) {dup} on {REPLICATE_AXES}.\n"
            f"  Runs that are not distinguished by the replicate axis are NOT replicates.\n"
            f"  Either an identifying config field is missing from the key, or these are "
            f"genuinely the same run.\n  Files:\n    "
            + "\n    ".join(r["path"] for r in rs))


def load(path):
    z = np.load(path, allow_pickle=True)
    cfg = json.loads(str(z["config"]))
    ladder = [int(v) for v in z["ladder"]]
    keys = ["pin"] + ladder
    blk = {k: (z[f"{k}_cnt"], z[f"{k}_ns"], z[f"{k}_ni"]) for k in keys}
    dig = [str(v) for v in z["ehat_digest"]] if "ehat_digest" in z.files else []
    ess_rung = z["ess_rung"] if "ess_rung" in z.files else None
    # Per-row information.  Absent from every run written before 2026-08-19; the readouts
    # that need it say so and fall back rather than silently substituting the block sums,
    # which cannot stand in for it (blocks are interleaved -- see `row_weights`).
    irow = {k: z[f"{k}_i_row"] for k in keys if f"{k}_i_row" in z.files}
    return dict(path=path, cfg=cfg, ladder=ladder, blk=blk, ess=z["ess"],
                ess_rung=ess_rung, irow=irow,
                features=[str(v) for v in z["features"]], digest=dig)


def stats(run, gdir, gn):
    """`m` and the leave-one-block replicates, per ladder key."""
    out = {}
    for k, b in run["blk"].items():
        full, _, reps = jackknife_blocks(*b)
        out[k] = (float(full @ gdir) / gn - 1.0, (reps @ gdir) / gn)
    return out



def jk_cov(mat):
    """Jackknife covariance of a (B, K) block-replicate matrix."""
    r = mat - mat.mean(axis=0, keepdims=True)
    return (len(mat) - 1) / len(mat) * (r.T @ r)


def _wmean(x, w):
    """Mean of `x` over rows, weighted by `w` if given, else flat.  Both (rows, K)."""
    return x.mean(axis=0) if w is None else (x * w).sum(axis=0) / w.sum(axis=0)


def _wsem(x, w):
    """Standard error on `_wmean`, rows independent."""
    if w is None:
        return x.std(axis=0) / np.sqrt(len(x))
    sw = w.sum(axis=0)
    return np.sqrt(((w * (x - (x * w).sum(axis=0) / sw)) ** 2).sum(axis=0)) / sw


def row_weights(run, ladder, gdir, source="pin"):
    """Per-row weight for the abscissa, (rows, K), or None if unavailable.

    WHY ROWS ARE NOT WEIGHTED EQUALLY, AND UNDER WHAT MODEL THIS WEIGHT IS THE RIGHT ONE.
    `ghat = sum_i s_i / sum_i I_i` is a ratio of sums, so how the per-row finite-draw law
    aggregates depends on HOW the draw error enters each row, and the two natural models
    disagree:

      * MULTIPLICATIVE on the response -- `s_i = I_i g (1 + eps_i)`.  Then
        `m = sum_i I_i eps_i / sum_i I_i`, an information-weighted mean of fractional
        errors, and `w_i = g^T I_i g` is EXACTLY right.  This is the standard result for
        multiplicative response biases and is the natural model here: a catalogue draw
        that mismatches a row perturbs that row's RESPONSE, not its score by an additive
        constant.
      * ADDITIVE on the score -- `s_i = I_i g + delta_i`.  Then
        `m = sum_i delta_i / (g sum_i I_i)`, whose numerator is a FLAT sum, and the
        unweighted abscissa is the right one.

    So this is exact under the multiplicative model, not assumption-free, and the flat
    abscissa is not merely a sloppier version of it -- it is the correct statistic under
    the other model.  The two are distinguishable from the per-row data now being saved
    (regress per-row `s_i - I_i g` on `I_i`: multiplicative predicts slope with `I`,
    additive predicts none), which is worth doing only if weighted and flat disagree
    enough to move a conclusion.  Projected on `g`, not traced: `I` is not isotropic and
    the trace is a different statistic.

    Then, exactly,   <1/ESS>_I = <1/ESS> + Cov(I_i, 1/ESS_i) / <I>   -- an identity at any
    sample size, not a leading-order expansion, since a weighted mean minus a flat mean IS
    the covariance over the mean weight.

    WHY THE WEIGHT IS FROZEN ACROSS RUNGS (`source`).  Each rung's own `I` is itself
    finite-draw biased, and that bias is the very quantity the ladder measures.  Recompute
    the weight per rung and it drifts along the ladder BECAUSE of the effect under study,
    producing a rung-dependent shift in the abscissa indistinguishable from the
    rung-dependent shift being looked for -- the `***` warning downstream would fire on an
    artifact of its own construction.  So the default takes the weight from the PIN arm,
    which carries no marginalisation and hence no draw bias at all, on the same rows; the
    frozen vector is then reused at every rung and any rung-dependence that survives comes
    from the ESS side, which is the thing of interest.  `source="deep"` freezes on the
    deepest rung (least biased, but not unbiased) and `source="per_rung"` is the
    unfrozen version, kept so the size of the confound can be MEASURED rather than assumed.

    Runs written before 2026-08-19 do not carry per-row `I` and get None here -- including
    jobs 15845011/12/13, which were already running when the field was added.  The
    block-level sums they DO carry are not a fallback: `eval_score_catprior.py` assigns
    blocks by `pair % jk_blocks`, so blocks are interleaved and near-identical in
    composition by construction, and a between-block covariance reads ~0 whatever the
    row-level truth is.  Reporting that as a bound would be worse than reporting nothing.
    """
    ir = run.get("irow") or {}
    need = ladder if source == "per_rung" else [{"pin": "pin", "deep": ladder[-1]}[source]]
    if not all(m in ir for m in need):
        return None

    def proj(m):
        return np.einsum("a,iab,b->i", gdir, np.asarray(ir[m], dtype=float), gdir)

    if source == "per_rung":
        return np.stack([proj(m) for m in ladder], axis=1)
    return np.repeat(proj(need[0])[:, None], len(ladder), axis=1)


SLOPE_FLOOR_K = 0.27   # = 2 x the measured `bias * K` = 2 x 0.133; the margin is IN here


def slope_floor(n_rungs):
    """Systematic floor on the recovered drift exponent, as a function of LADDER LENGTH.

    Measured by injection, not asserted, and NOT one number.  Injecting true exponents
    1.0/0.7/0.5/0.3 at the production rung sets (4 seeds, 6000 rows each) gives a recovery
    bias that is FLAT in the true exponent -- 0.026/0.025/0.024/0.023 across that range at
    K=5 -- and falls as 1/K in the number of rungs:

        K=4 [1,2,4,8]        bias 0.034      bias*K = 0.136
        K=5 [1,2,4,8,16]     bias 0.026      bias*K = 0.130
        K=7 [1..64]          bias 0.019      bias*K = 0.133

    i.e. `bias ~ K^-1.03`, spread 1.8% about a constant `bias*K`.  So a single floor is
    wrong in the direction that matters: it is too loose on long ladders, throwing away
    sensitivity exactly where the measurement is best.  The earlier fixed 0.10 came from ONE
    injected truth at ONE depth on a single seed and was 3-5x too conservative.

    The returned floor carries a factor-2 margin over the measured bias.  The bias is
    one-directional -- the recovered exponent comes back SHALLOWER than the truth
    (-0.974 for an injected -1.0) -- and shallower is the direction the alarm fires in, so
    the floor has to cover it rather than merely bracket it.

    IT STAYS A FLOOR, NOT A CALIBRATION.  Measured-vs-injected pairs would invert into a
    mapping that could be applied to the central slope, and that mapping is exactly the
    empirical correction AGENTS.md forbids: it would make a reported number agree with an
    expectation by construction and cost the ability to call it a measurement.  Widening the
    threshold costs sensitivity honestly; correcting the value would hide the same
    uncertainty.  Trade taken deliberately.
    """
    # NOT `2 * SLOPE_FLOOR_K / K` -- the factor of 2 is already folded into the
    # constant, and applying it twice gave a 4x margin behind a docstring that
    # promised 2x.  Caught by `test_the_slope_floor_is_a_FUNCTION_of_ladder_length`.
    return SLOPE_FLOOR_K / max(int(n_rungs), 1)


def drift_slope(er, wf, wp, ladder, n_boot=64, seed=0):
    """Power law of the frozen-vs-unfrozen abscissa drift in M, WITH ITS OWN ERROR BAR.

    Returns `(slope, sd, drift)`: the exponent `p` in `drift ~ M^p`, its standard error, and
    the per-rung drift vector.  `(nan, nan, drift)` if fewer than three rungs carry a
    positive drift, since a power law through two points has no residual.

    WHY THE ERROR BAR IS NOT OPTIONAL.  This diagnostic warns when the drift falls MORE
    SLOWLY than 1/M -- the case where the information's draw bias outlives the shear
    estimate's and so escapes the 1/M -> 0 extrapolation.  Without a resolution attached,
    the quiet branch is unreadable: silence would mean "consistent with 1/M" when it may
    only mean "within resolution".  That is the minimum-detectable-offset discipline, and
    the reason for it is the project's own scar -- an unresolved measurement is not a pass.

    The bar is a ROW BOOTSTRAP rather than the seed-to-seed scatter, because there are
    typically two draw seeds and an error estimated from two points is barely an estimate.
    Rows are the independent unit here: each carries its own draws (`draw_indices` gives
    every row an independent index set), so resampling rows resamples exactly the noise
    the slope is subject to.  It does NOT cover the fixed catalogue pool, which is common
    to every row and every seed -- same blind spot as everywhere else in this file.

    AND THE BOOTSTRAP IS NOT THE WHOLE RESOLUTION.  It measures DRAW NOISE.  It does not see
    the estimator's own recovery bias: on injected synthetics a true 1/M drift comes back as
    M^-0.95 while the bootstrap bar is +/- 0.01, i.e. the truth sits 4 sigma from the
    estimate and the bar is blind to it.  The drift is not exactly a power law over a short
    ladder, so fitting one costs a systematic of order `slope_floor(K)` in the exponent.  The
    threshold below is therefore `max(2*sd, slope_floor(K))`, and reporting the bootstrap bar
    alone would have manufactured a 4-sigma alarm out of data that was right.
    """
    rng = np.random.default_rng(seed)

    def slope_of(idx):
        d = []
        for e, f, q in zip(er, wf, wp):
            e, f, q = e[idx], f[idx], q[idx]
            x = 1.0 / np.maximum(e, 1e-12)
            flat = x.mean(axis=0)
            d.append(np.abs((x * q).sum(0) / q.sum(0) - (x * f).sum(0) / f.sum(0)) / flat)
        d = np.mean(d, axis=0)
        if len(ladder) < 3 or not np.all(d > 0):
            return np.nan, d
        return np.polyfit(np.log([float(m) for m in ladder]), np.log(d), 1)[0], d

    n = len(er[0])
    full, drift = slope_of(np.arange(n))
    if not np.isfinite(full):
        return np.nan, np.nan, drift
    boot = [slope_of(rng.integers(0, n, size=n))[0] for _ in range(n_boot)]
    boot = np.array([b for b in boot if np.isfinite(b)])
    return full, (float(boot.std(ddof=1)) if len(boot) > 2 else np.nan), drift


def ess_abscissa(er, ladder, wr=None):
    """Per-rung ESS summaries from a list of per-run `ess_rung` arrays, each (rows, K).

    Returns `(inv, mean, cv, sd)` as dicts keyed by rung.

    THE ABSCISSA IS <1/ESS_i>, NOT 1/<ESS_i>, and that is the whole point of this function.
    The bias law is PER ROW: each row's finite-draw error goes as c_i / ESS_i with that
    row's own effective sample size, and the estimator is a ratio of sums over rows, so the
    aggregate error is a MEAN OF RECIPROCALS.  Building the abscissa as one over the mean
    ESS swaps those two, and by Jensen they differ by the row-to-row spread:

        <1/ESS> * <ESS> = 1 + Var(ESS)/<ESS>^2      to leading order,

    i.e. 4% at CV 0.2, 25% at CV 0.5, 100% at CV 1.0.  Against the sensitivity recorded in
    AMENDMENT 6 -- a 10% abscissa shift moving chi2/dof from 1.20 to 465 -- a CV of 0.5 is
    far outside what the fit can absorb.  AND IT IS A BIAS, NOT A VARIANCE: more rows
    measure it better, they do not make it smaller.  Worse, the gap is rung-DEPENDENT
    because CV(ESS) grows with M, so it distorts the SHAPE rather than rescaling the slope,
    which is exactly what the GLS is most sensitive to.

    `sd` is the uncertainty on <1/ESS>: the larger of the within-run standard error on the
    row mean (rows carry independent draws -- see `draw_indices`, which gives each row its
    own) and the between-seed scatter.  It does NOT cover the fixed POOL: every row of every
    seed draws from the same catalogue, so a pool-level shift moves all rungs coherently and
    no scatter measured at fixed pool can see it.  Same shape as the score-grid bank
    systematic, same remedy (vary the pool).  The row mean is also unweighted, where the
    estimator weights rows by their information; a weighted <1/ESS> would be exact.
    """
    x = [1.0 / np.maximum(e, 1e-12) for e in er]
    w = wr if wr is not None else [None] * len(er)
    inv_run = np.stack([_wmean(xi, wi) for xi, wi in zip(x, w)])
    flat_run = np.stack([xi.mean(axis=0) for xi in x])
    per_run = np.stack([e.mean(axis=0) for e in er])
    within = np.mean([_wsem(xi, wi) for xi, wi in zip(x, w)], axis=0)
    between = (inv_run.std(axis=0, ddof=1) / np.sqrt(len(inv_run))
               if len(inv_run) > 1 else np.zeros_like(within))
    cv = np.mean([e.std(axis=0) / np.maximum(e.mean(axis=0), 1e-12) for e in er], axis=0)
    shift = inv_run.mean(axis=0) / np.maximum(flat_run.mean(axis=0), 1e-300) - 1.0
    return (dict(zip(ladder, inv_run.mean(axis=0))), dict(zip(ladder, per_run.mean(axis=0))),
            dict(zip(ladder, cv)), dict(zip(ladder, np.maximum(within, between))),
            dict(zip(ladder, shift)))


def _jk_sd(reps):
    """Jackknife sd of a (B,) leave-one-block replicate vector."""
    reps = np.asarray(reps, dtype=float)
    r = reps - reps.mean()
    return float(np.sqrt((len(reps) - 1) / len(reps) * (r @ r)))


def irreducible_term(diffs, sd_paired, n_shards=2):
    """`(a_prod, sd, n)`: the finite-catalogue error the ladder cannot remove.

    `diffs` and `sd_paired` are the per-split `d_A - d_B` and its within-split jackknife
    bar, AT ONE RUNG -- use the top rung, where the draw term has died.

    WHY THE SCATTER ACROSS SPLITS AND NOT THE JACKKNIFE BAR.  The jackknife resamples
    GALAXIES, not the pool.  A different pool shifts every row coherently, so that shift
    survives into every leave-one-block replicate identically and the jackknife variance of
    it is ZERO.  The pool term is therefore invisible to `sd_paired` by construction; it
    shows up only as scatter of `d_A - d_B` from split to split.  `sd_paired` is the NOISE
    FLOOR that scatter has to beat, which is why it is subtracted rather than ignored.

    Scaling: with disjoint 1/n shards, each arm's pool is `n_shards` times smaller than
    production, so if `a ~ 1/N_pool` then `var(A-B) = 2 * n_shards * a_prod`.  At n=2 that
    is the familiar `var(A-B)/4`.  THE 1/N SCALING IS AN ASSUMPTION the halves cannot test
    on their own -- run a second shard size (`pool_shard_indices`) to turn it into a
    measurement.

    NOT CLIPPED AT ZERO, DELIBERATELY.  `a` is a difference of two noisy variances, so a
    NEGATIVE draw is ordinary at small `n` -- it is a null result, not an error condition.
    Clipping it would bias a symmetric estimator UPWARD, and would do so hardest in exactly
    the low-signal regime this test is planned to run in, which is the direction that
    manufactures a detection out of noise.  The signed value is returned with its error and
    the caller quotes an interval; `test_a_negative_draw_is_REPORTED_not_clipped` pins it.

    Returns `(a_prod, sd, n)` as a signed VARIANCE and its error, with `(None, None, 1)`
    when fewer than two splits are given: one split carries no scatter to measure.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return None, None, n
    V = float(np.var(d, ddof=1))
    floor = float(np.mean(np.asarray(sd_paired, dtype=float) ** 2))
    a_prod = (V - floor) / (2.0 * int(n_shards))       # debias by the sampling floor
    # the across-split variance dominates the error; a chi2 variance from n samples carries
    # a fractional error sqrt(2/(n-1)), and the floor's own error is smaller by ~sqrt(n)
    sd = V * np.sqrt(2.0 / (n - 1)) / (2.0 * int(n_shards))
    return a_prod, sd, n


def shard_family_term(arm_full, arm_jk):
    """`a` at the FULL pool from k disjoint shard arms, with its error and its dof.

    Generalises `irreducible_term` off the 2-arm difference.  k arms give k-1 degrees of
    freedom from k jobs where halves give 1 from 2, and a k-way arm carries k*a_full because
    the finite-catalogue term goes as 1/N_pool, so the floor matters less.  Both effects
    favour large k -- see `doc/WORKLOG.md` cont.189 §6c for the measured comparison.

    `arm_full` is the (k,) vector of arm values, `arm_jk` the (k, B) leave-one-block-out
    replicates.  The arms score the SAME galaxies, so the common galaxy fluctuation cancels
    out of the across-arm scatter exactly as it does out of a paired difference; the floor
    is therefore the jackknife scale of the CENTRED arms, corrected by k/(k-1) for the
    variance that the centring itself removes.

    Signed, NOT clipped, for the reason in `irreducible_term`.
    """
    arm_full = np.asarray(arm_full, dtype=float)
    arm_jk = np.asarray(arm_jk, dtype=float)
    k, nb = arm_jk.shape
    if k < 2:
        return None, None, 0
    V = float(np.var(arm_full, ddof=1))                    # k*a_full + F_arm
    r = arm_jk - arm_jk.mean(axis=0, keepdims=True)        # drop the shared galaxy term
    jk = (nb - 1) / nb * ((r - r.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    floor = float(jk.mean()) * k / (k - 1)                 # centring removed 1/k of it
    dof = k - 1
    a = (V - floor) / k
    sd = V * np.sqrt(2.0 / dof) / k
    return a, sd, dof


def a_verdict(a_prod, sd, n_sigma=2.0, z=1.645):
    """`(kind, amplitude)`: how `a` must be reported, by a rule fixed BEFORE the data.

    Pre-registered in `doc/WORKLOG.md` cont.189 §6 so the choice between "a value" and "an
    upper limit" cannot be made while looking at the answer:

      * `a_prod > n_sigma * sd`  -> DETECTION; quote the amplitude sqrt(a).
      * otherwise                -> UPPER LIMIT at `a + z*sd` (z = 1.645, one-sided 95%).

    Clipping at zero is legitimate for the LIMIT -- a limit on a variance cannot be negative
    -- and illegitimate for the point estimate, which is why only this branch does it.
    """
    if a_prod is None:
        return "none", None
    if a_prod > n_sigma * sd:
        return "detection", float(np.sqrt(a_prod))
    return "upper limit", float(np.sqrt(max(0.0, a_prod + z * sd)))


def pool_halves_report(sa, sb, ladder):
    """A-vs-B pool-halves comparison: `(rows, pin_gap)`.

    `sa`, `sb` are `stats()` outputs for the two arms.  Both arms score the SAME rows with
    the SAME draw seed and differ only in which disjoint half of the catalogue is drawable,
    so the galaxy contribution to the variance cancels in the difference and the pool
    contribution cannot -- the arms share no atoms by construction.

    THE GAIN IS THE MEASUREMENT, NOT A HEALTH CHECK.  `gain = sd(unpaired)/sd(paired)`
    equals `sqrt(1 + Vgal/Vpool)`, so it reports how the variance splits between the rows
    and the pool.  A modest gain is the EXPECTED outcome here and says the pool term is
    large, not that the pairing broke.  It can only be read against a prediction made
    beforehand.  `doc/WORKLOG.md` cont.189 §2 registers it, and registers what to grade on:
    the MONOTONE RISE with M, which is robust to the unmeasured Vpool/Vdraw ratio, rather
    than the point values, which are not.  A flat or falling gain is a failure; missing a
    point prediction is not.  Expect a rise from ~1.3 that SATURATES -- the saturation level
    measures the irreducible finite-catalogue term, the error more draws cannot remove.
    Read without that prediction the gain is uninterpretable in either direction.

    `pin_gap` is the HARD GATE: the pin arm never reads the pool, so it must be identical
    between the halves.  Any nonzero value means the split perturbed something it should
    not have, and nothing else in the table is worth reading.
    """
    pin_gap = abs(sa["pin"][0] - sb["pin"][0])
    rows = []
    for m in ladder:
        da, db = sa[m][0] - sa["pin"][0], sb[m][0] - sb["pin"][0]
        ra = sa[m][1] - sa["pin"][1]
        rb = sb[m][1] - sb["pin"][1]
        sd_a, sd_b = _jk_sd(ra), _jk_sd(rb)
        sd_paired = _jk_sd(ra - rb)
        sd_unpaired = float(np.hypot(sd_a, sd_b))
        gain = sd_unpaired / sd_paired if sd_paired > 0 else np.inf
        rows.append(dict(M=m, d_a=da, d_b=db, diff=da - db, sd_paired=sd_paired,
                         sd_unpaired=sd_unpaired, gain=gain,
                         nsig=abs(da - db) / sd_paired if sd_paired > 0 else np.inf))
    return rows, pin_gap


def fit_ladder(D, reps, ladder, seed_full_cov=False, x=None, xname="1/M"):
    """GLS fit of `d(M) = d_inf + a/M (+ b/M^2)` against the FULL rung covariance.

    `D` is (S, K): the per-draw-seed `d(m)` for each rung.  `reps` is (B, K): the
    seed-averaged leave-one-block jackknife replicates of the same quantity.

    WHY GLS AND NOT WEIGHTED LEAST SQUARES.  The rungs are nested to the bit -- M=2
    contains M=1's accumulated term -- and every rung is differenced against the SAME pin
    arm, so they are heavily overlapping measurements, not independent ones.  Fitting them
    with independent errors reports a flattering chi2 (mostly confirming that the rungs
    share their noise) and parameter bars that are too small.

    TWO NOISE SOURCES, COMBINING DIFFERENTLY ACROSS SEEDS.  Galaxy noise is COMMON to every
    draw seed (same rows), so averaging seeds does not shrink it.  Draw noise is independent
    per seed and shrinks as 1/S.  The block jackknife on the seed-averaged replicates
    estimates the TOTAL of both -- see the comment in the body for why `C_seed/S` is NOT
    added on top, which an earlier version did and which double counts.

    THE TWO SOURCES ARE PROBED BY DIFFERENT OUTPUTS OF THIS FIT, AND CONFLATING THEM LEADS
    TO A WRONG INFERENCE.  The bar on `d_inf` is set mostly by the COMMON component: a
    shared offset moves the intercept directly and does not shrink when seeds are pooled.
    The chi2 is set by the RESIDUAL after the smooth two-parameter law has been subtracted,
    and the fit absorbs the shared smooth part into `d_inf` and `a` -- so what the chi2 sees
    is draw-dominated and DOES shrink on pooling.  Measured on the two nbr b=0.0 seeds:
    corr(raw d(m)) = +0.99 while corr(fit residuals) = -0.71, the d_inf bar shrinks by only
    0.966 on pooling (independence would give 0.707) while the residual rms shrinks by 0.513.
    So "the error barely moved on pooling" and "the chi2 moved a lot" are consistent, not
    contradictory, and neither implies the other about seed independence.

    HARTLAP.  GLS needs `C^-1`, but `C` is estimated, and the inverse of a noisy covariance
    is biased high even when the covariance itself is unbiased -- which shrinks the errors
    and flatters the chi2, precisely the number being quoted as evidence that the 1/M law
    holds.  The inverse is scaled by `(N-p-2)/(N-1)`, where `p` is the DIMENSION OF THE
    DATA VECTOR (`mask.sum()`, the number of rungs retained) and NOT the number of fitted
    parameters -- the two readings differ and only the former is Hartlap.  At N=200 blocks
    and p=5 rungs the factor is 193/199 = 0.9698, so an uncorrected chi2 would be 3.1%
    larger.  N is taken as the block count, which is GENEROUS: jackknife replicates are not
    independent samples, so the corrected bars should still be read as slightly optimistic
    rather than exact.

    Returns `(fits, note)` where `fits` is a list of
    `(name, beta, cov, chi2_per_dof, mask, order)`.
    """
    D = np.atleast_2d(np.asarray(D, dtype=float))
    reps = np.asarray(reps, dtype=float)
    S, K = D.shape
    # THE JACKKNIFE ALREADY CONTAINS THE DRAW NOISE.  An earlier version of this function
    # added `C_seed / S` on top of `C_jk`, on the reasoning that galaxy noise is common
    # across seeds while draw noise is not.  That double counts, and the argument is wrong
    # in a specific way worth recording: `d_M` is a mean over galaxies, each carrying its
    # OWN draw realisation, so per-galaxy draw noise is part of the block-to-block scatter
    # and the block jackknife estimates it along with everything else.  Worse, `reps` here
    # is the SEED-AVERAGED replicate array, so the jackknife of it already reflects the 1/S
    # shrinkage that averaging seeds produces.  Adding `C_seed / S` therefore inflates a
    # covariance that was already complete -- and an inflated covariance shows up as a
    # flatteringly LOW chi2, i.e. as a fit that looks passed when it has merely gone
    # untested.  `C_jk` alone is the total sampling covariance; the seed scatter is used as
    # an independent CROSS-CHECK on it instead (see `seed_scatter_check`).
    C = jk_cov(reps)
    note = ""
    if S == 1:
        note = ("ONE draw seed, so there is no independent check on the covariance; the "
                "jackknife is taken on trust.")
    yv = D.mean(axis=0)
    # `x` lets the caller fit against 1/ESS instead of 1/M.  NOTE WHAT THIS CAN AND CANNOT
    # DO: `C` above is the EMPIRICAL jackknife covariance, so M enters the chi2 only through
    # this abscissa.  Substituting ESS "in the covariance only" therefore cannot move the
    # chi2 by even a little -- there is no modelled 1/M in the covariance to replace.  See
    # the note printed beside the ESS table in `main`.
    xv = np.asarray(x, dtype=float) if x is not None \
        else np.array([1.0 / m for m in ladder], dtype=float)
    nblk = len(reps)

    def gls(mask, order):
        A = np.stack([xv[mask] ** k for k in range(order + 1)], axis=1)
        p = int(mask.sum())
        Ci = np.linalg.pinv(C[np.ix_(mask, mask)])
        if nblk - p - 2 > 0:                      # Hartlap
            Ci = Ci * (nblk - p - 2) / (nblk - 1)
        cov = np.linalg.pinv(A.T @ Ci @ A)
        bh = cov @ (A.T @ Ci @ yv[mask])
        r = yv[mask] - A @ bh
        dof = max(p - (order + 1), 1)
        return bh, cov, float(r @ Ci @ r) / dof

    allm = np.ones(K, dtype=bool)
    variants = [(f"{xname}, all rungs", allm, 1)]
    if ladder[0] == 1 and K >= 3:
        # M=1 is not a mixture at all -- it is the pinned estimator with the truth swapped
        # for a random row -- so it has no claim on the asymptotic line, and as the leftmost
        # point it carries the most leverage on the intercept.
        nom1 = allm.copy()
        nom1[0] = False
        variants.append((f"{xname}, drop M=1", nom1, 1))
    if K >= 4:
        # With only three rungs a pure-1/M fit absorbs any curvature into `a` and shifts the
        # intercept without complaining.  The quadratic term TESTS the law over this range
        # instead of assuming it.
        variants.append((f"{xname} + sq, all", allm, 2))
    out = []
    for name, mask, order in variants:
        bh, cov, c2 = gls(mask, order)
        out.append((name, bh, cov, c2, mask, order))
    return out, note



def seed_scatter_check(D, reps):
    """Does the jackknife covariance PREDICT the scatter actually seen across draw seeds?

    This is a ONE-SIDED test for an UNDERSTATED covariance.  It cannot detect an inflated
    one: the seed spread measures only the draw component, so it sits far below the total
    whenever galaxy sampling dominates, which is the normal healthy case.  A low chi2 with
    small ratios therefore remains ambiguous -- that ambiguity is what the chi2 inflation
    and the pre-registered thresholds are for, not this check.

    For S seeds the observed spread of `d_M` across seeds estimates the DRAW component
    alone.  The jackknife bar on the seed-averaged `d_M` is the TOTAL (galaxy + draw/S), so
    the observed seed spread must not EXCEED it: draw noise is a part of the total, not an
    addition to it.  Returns per-rung `(seed_spread, jackknife_bar, ratio)`.
    """
    S = len(D)
    if S < 2:
        return None
    spread = D.std(axis=0, ddof=1) / np.sqrt(S)     # error on the seed mean
    bar = np.sqrt(np.diag(jk_cov(reps)))
    return list(zip(spread, bar, spread / np.maximum(bar, 1e-12)))


def intercept_spread(fits):
    """Max intercept shift across fit variants, and whether it exceeds 2 sigma.

    A decision rule the analyst cannot quietly choose between afterwards: if the
    extrapolation moves by more than the baseline's own 2 sigma, the SPREAD is the
    systematic and no single fit may be quoted.
    """
    b0, cov0 = fits[0][1], fits[0][2]
    mv = max((abs(f[1][0] - b0[0]) for f in fits[1:]), default=0.0)
    return mv, mv >= 2 * np.sqrt(cov0[0, 0])




def _rho_from(a, b):
    va, vb, vd = a.var(ddof=1), b.var(ddof=1), (b - a).var(ddof=1)
    den = 2.0 * np.sqrt(va * vb)
    return (va + vb - vd) / den if den > 0 else np.nan


def rho_with_error(a_reps, b_reps):
    """rho and a DELETE-ONE-BLOCK jackknife bar on it.

    The leave-one-out replicates deviate from their mean in proportion to the block's own
    contribution, so the correlation OF the replicates is the correlation of the per-block
    fluctuations, and deleting replicate rows is a legitimate jackknife of that
    correlation.  Doing it this way rather than quoting the Fisher form `(1-rho^2)/sqrt(N-3)`
    matters because the replicates are not independent draws, which the Fisher expression
    assumes -- so this bar is the wider and more honest of the two.

    Without a bar, "rho agrees with the prediction to 0.08" is a tolerance chosen after
    seeing the numbers rather than a test that could have failed.
    """
    n = len(a_reps)
    full = _rho_from(a_reps, b_reps)
    keep = np.arange(n)
    vals = np.array([_rho_from(a_reps[keep != i], b_reps[keep != i]) for i in range(n)])
    sig = float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2)))
    return float(full), sig


def chi2_sf(x, dof):
    """Survival function, for turning chi2 into a p-value without assuming scipy."""
    try:
        from scipy.stats import chi2 as _c
        return float(_c.sf(x, dof))
    except Exception:
        import math
        if dof % 2 == 0:                       # exact for even dof
            k, t, s = dof // 2, 1.0, 1.0
            for i in range(1, k):
                t *= (x / 2) / i
                s += t
            return float(math.exp(-x / 2) * s)
        return float(math.erfc(math.sqrt(max(x, 0.0) / 2)))   # dof=1 exact, else indicative


def rho_floor(M, top):
    """The SMALLEST correlation nested rungs can have.  Parameter-free.

    Write `d_M = g + (1/M) sum_{j<=M} eps_j`, with `g` the pin-arm block noise (COMMON to
    every rung, because every rung is differenced against the same pin) and `eps_j` the
    per-draw terms (NESTED, because rung M averages the first M of them).  Then

        Var(d_M)      = a + s/M
        Cov(d_M, d_N) = a + s/N          for M < N   -- only the shared prefix contributes
        rho           = sqrt((a + s/N) / (a + s/M))

    which is monotone in the noise ratio `a/s`: it runs from `sqrt(M/N)` when the per-draw
    noise dominates up to 1 when the pin-arm noise does.  So `sqrt(M/N)` is a FLOOR holding
    for ANY noise mix, with no free parameter to tune.

    Worth more than comparing against a synthetic rho, because it can be pre-registered
    without knowing the noise budget: rho BELOW the floor is not "lower than expected", it
    is impossible under nesting, and indicts the accumulator, the ladder snapshot or the
    seed handling rather than the physics.
    """
    return float(np.sqrt(M / top))


def implied_noise_ratio(rho_val, M, top):
    """Invert rho for `a/s`, the pin-arm / per-draw noise ratio.

    A SECOND, independent route to a number the run already measures directly (the block
    jackknife bar against the seed scatter).  Two routes agreeing is a real consistency
    check; disagreeing means the noise model behind the GLS covariance is wrong, whatever
    the fit's chi2 says.
    """
    r2 = float(rho_val) ** 2
    if not (0.0 < r2 < 1.0):
        return float("nan")
    return (r2 / M - 1.0 / top) / (1.0 - r2)


def grid_pairs(runs):
    """Is `d(m)` BANK-INDEPENDENT?  The check the whole ladder result rests on.

    `d(m)` is a difference of two estimators evaluated on the same node bank, and it is
    tempting to assume the bank's quadrature error cancels in it.  It does not: the
    marginalised likelihood is a mixture over `M` draws and so is SMOOTHER in `e` than the
    pinned one, a fixed bank resolves a broad integrand better than a peaked one, and the
    two arms therefore carry DIFFERENT bank errors.  The difference of those errors sits
    directly on `d(m)`.

    So run the same rung at two `grid_n` and difference the differences, block by block:

        D(M) = [d(m) at grid B] - [d(m) at grid A]

    `D` small means the bank error cancels out of `d(m)` after all and the ladder result
    stands; `D` of order the pinned arm's own bank error (~0.7% at grid_n=61, cont.185/186)
    means the ladder result is measuring the bank and not the draws.

    TWO GUARDS, because a broken pairing and a genuine null are the same output.  The arms
    must have seen identical mock data -- checked bitwise on the `xhat` digest, not assumed
    from having asked for the same GPU -- and the pairing gain must be far above 1x, which
    is the shared galaxy noise actually cancelling.
    """
    from collections import defaultdict
    byk = defaultdict(dict)
    for r in runs:
        k = (",".join(r["features"]), float(r["cfg"]["beta"]), int(r["cfg"]["max_rows"]),
             tuple(r["ladder"]), int(r["cfg"]["draw_seed"]))
        byk[k][int(r["cfg"]["grid_n"])] = r
    any_pair = False
    for k, arms in sorted(byk.items()):
        if len(arms) < 2:
            continue
        any_pair = True
        feat, beta, rows, ladder, dseed = k
        gs = sorted(arms)
        print(f"\n{'='*94}\nGRID CONVERGENCE OF d(m):  {feat}  beta={beta}  "
              f"rows={rows:,}  draw seed {dseed}")
        g1 = float(arms[gs[0]]["cfg"]["closure_g"])
        g2 = float(arms[gs[0]]["cfg"]["closure_g2"])
        gvec = np.array([g1, g2]); gn = float(np.hypot(*gvec)); gdir = gvec / gn
        for a, b in zip(gs, gs[1:]):
            ra, rb = arms[a], arms[b]
            da, db = ra["digest"], rb["digest"]
            if da and db:
                verdict = ("IDENTICAL: the arms are genuinely paired" if da == db else
                           "*** DIFFERENT: the arms saw different mock data, so the "
                           "difference below is NOT paired and its bar is fiction ***")
                print(f"  xhat digest  grid_n={a}: {' '.join(da)}\n"
                      f"               grid_n={b}: {' '.join(db)}\n  -> {verdict}")
            else:
                print(f"  xhat digest unavailable in one arm (run predates the guard); "
                      f"pairing UNVERIFIED -- treat a null below as unproven")
            sa, sb = stats(ra, gdir, gn), stats(rb, gdir, gn)
            print(f"\n  {'M':>5} {'d(m)@'+str(a):>11} {'d(m)@'+str(b):>11} "
                  f"{'D = B - A':>11} {'+/-':>9} {'nsig':>6} {'pair':>8}")
            for m in ladder:
                dA = sa[m][0] - sa["pin"][0]
                dB = sb[m][0] - sb["pin"][0]
                rA = sa[m][1] - sa["pin"][1]
                rB = sb[m][1] - sb["pin"][1]
                sd = float(jackknife_sigma((rB - rA)[:, None])[0])
                single = float(jackknife_sigma(rB[:, None])[0])
                D = dB - dA
                print(f"  {m:>5} {dA:>+11.3%} {dB:>+11.3%} {D:>+11.3%} {sd:>9.3%} "
                      f"{abs(D)/max(sd,1e-12):>6.1f} {single/max(sd,1e-12):>7.1f}x")
            print(f"\n  A pairing gain near 1x means the two arms share no noise and the "
                  f"bars above are\n  meaningless regardless of what D says.  cont.185/186 "
                  f"saw 48x-273x on comparisons\n  that genuinely paired.")
    if not any_pair:
        print("\n(no two runs differ only in grid_n, so no bank-convergence check to do)")


def plot(panels, path):
    """d(m) against 1/M, one line per tier -- the convergence at a glance.

    Plotted against 1/M rather than M so the predicted law is a STRAIGHT LINE through the
    origin: curvature is then visible as curvature, and the intercept the eye reads off at
    1/M = 0 is the extrapolated closure residual, which is the number that has to be zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for lab, pts, fit in panels:
        x = np.array([1.0 / m for m, _, _ in pts])
        y = np.array([v for _, v, _ in pts]) * 100
        e = np.array([b for _, _, b in pts]) * 100
        line = ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=2, lw=1.2, label=lab)
        if fit is not None:
            xx = np.linspace(0, x.max() * 1.05, 50)
            ax.plot(xx, (fit[0] + fit[1] * xx) * 100, ls="--", lw=0.9,
                    color=line[0].get_color(), alpha=0.7)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("1 / M   (M = catalogue draws per galaxy;  1/M = 0 is the exact marginal)")
    ax.set_ylabel(r"$d(m)$ vs pinned conditioning  [%]")
    ax.set_title("Catalogue-drawn conditioning prior: closure vs draw count")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nfigure -> {path}")


def shards_block(runs):
    """Print the k-way pool-shard family for every complete set among `runs`.

    A family is k runs identical in everything -- features, beta, rows, grid, ladder, DRAW
    seed, split seed and k -- except which disjoint shard of the catalogue is drawable.  The
    draw seed must match across the family or the arms stop being paired.

    PREFERRED OVER halves (k = 2).  k arms carry k-1 degrees of freedom for k jobs where
    halves carry 1 for 2, and the finite-catalogue term is k times larger per arm so the
    jackknife floor is a smaller fraction of it.  Measured at equal cost: 8 jobs as k=8
    reach 1.84 sigma where 8 jobs as four half-splits reach 1.13 (cont.189 §6c).
    """
    fam = {}
    for r in runs:
        c = r["cfg"]
        sh = int(c.get("pool_shard", -1))
        if sh < 0:
            continue
        key = (",".join(r["features"]), float(c["beta"]), int(c["max_rows"]),
               int(c["grid_n"]), tuple(r["ladder"]), int(c["draw_seed"]),
               int(c.get("pool_split_seed", -1)), int(c.get("pool_nshards", 2)))
        fam.setdefault(key, {})[sh] = r
    if not fam:
        return
    print(f"\n{'='*94}\nPOOL SHARDS: the irreducible finite-catalogue term from a k-way "
          f"split of the draw pool")
    for key in sorted(fam):
        arms, k = fam[key], key[7]
        have = sorted(arms)
        print(f"\n  {key[0]}  beta={key[1]}  rows={key[2]:,}  draw seed={key[5]}  "
              f"split seed={key[6]}  k={k}")
        # INCOMPLETE FAMILIES ARE REFUSED, not silently averaged over what arrived: the
        # arms are disjoint by construction, so a missing one is missing catalogue, and the
        # across-arm scatter would be measured on a pool that is not the one it claims.
        if len(have) < k:
            print(f"    INCOMPLETE: {len(have)} of {k} shards present ({have}).  "
                  f"Refusing to estimate -- the missing arms are missing catalogue,\n"
                  f"    not missing precision.  Submit the rest before reading this.")
            continue
        ladder = list(arms[have[0]]["ladder"])
        g = np.array([float(arms[have[0]]["cfg"]["closure_g"]),
                      float(arms[have[0]]["cfg"]["closure_g2"])])
        gn = float(np.hypot(*g))
        st = [stats(arms[i], g / gn, gn) for i in have]
        # THE HARD GATE: pin never reads the pool, so it must be identical across all arms.
        pins = np.array([x["pin"][0] for x in st])
        gap = float(pins.max() - pins.min())
        if gap > 0:
            print(f"    *** PIN GATE FAILED: pin spans {gap*100:+.6f}% across the arms, but "
                  f"pin never reads the pool.\n        NOTHING BELOW IS INTERPRETABLE.")
            continue
        print("    pin gate: identical across all arms, as it must be")
        ceil = np.sqrt((k - 1) / 2.0)
        # THE "INDEPENDENT OF `a`" CLAIM IS TRUE ONLY AS q -> 0, and this line used to state
        # it flatly.  cont.189 §6a's algebra is significance = (1-q)*sqrt((n-1)/2) with
        # q = F/V and V = k*a + F, so `a` cancels from the RATIO at fixed V and re-enters
        # through V.  Its numerical check (1.22 sigma at both a = 0.1% and a = 1.0%) was run
        # where the floor was negligible, i.e. exactly the limit where the claim holds.  The
        # production run sits outside it, which is how the wording got caught.
        print(f"    significance CEILING at k={k}: {ceil:.2f} sigma, reached only as q -> 0; "
              f"at the MEASURED q it is\n    (1-q) times that, per column below.  `a` cancels "
              f"from the ratio at fixed V and returns through\n    V = k*a + F, so a small "
              f"true `a` lowers the ceiling however many arms are run (cont.189 §6a).")
        # q = F/V IS PRINTED, because the design's whole significance argument runs through
        # it: cont.189 §6a gives significance = (1-q)*sqrt((n-1)/2), and the pre-registered
        # 1.77 sigma at k=8 was q = 0.053 carried over from a simulation with an injected
        # a_full of 0.150% rms.  q is not a property of the design -- q = F/(k*a + F) -- so a
        # smaller true `a` drives it toward 1 and the ceiling toward zero.  Printing V and F
        # while leaving the reader to form the ratio hid exactly the number that decides
        # whether the run could ever have detected anything.
        # THE CEILING COLUMN IS HEADED BY ITS FORMULA, not by a name.  It is a deterministic
        # function of `q` beside it, and two algebraically dependent columns printed side by
        # side read as two facts agreeing when they cannot disagree -- the layout of
        # corroboration with the substance unavailable.  Naming it after its input is the
        # cheapest way to stop that reading.
        print(f"    {'M':>4} {'sd arms [%]':>12} {'floor [%]':>10} {'q = F/V':>9} "
              f"{'(1-q)*%.2f' % ceil:>11} {'a signed [%]':>13} {'a quoted [%]':>16} {'dof':>5}")
        for m in ladder:
            af = np.array([x[m][0] - x["pin"][0] for x in st])
            ajk = np.array([x[m][1] - x["pin"][1] for x in st])
            a, sd, dof = shard_family_term(af, ajk)
            kind, amp = a_verdict(a, sd)
            signed = np.sign(a) * np.sqrt(abs(a)) * 100
            V = float(np.var(af, ddof=1))
            floor = max(0.0, V - a * k)
            tag = f"{amp*100:.3f} ({'<' if kind == 'upper limit' else '='})"
            qv = (V - a * k) / V if V > 0 else float("nan")
            print(f"    {m:>4} {np.sqrt(V)*100:>12.3f} {np.sqrt(floor)*100:>10.3f} "
                  f"{qv:>9.2f} {max(0.0, (1 - qv) * ceil):>9.2f} "
                  f"{signed:>13.3f} {tag:>16} {dof:>5}")
        print("    `a` is scaled to the FULL pool and assumes a ~ 1/N_pool; a SECOND k value "
              "is what tests that.\n    Read at the TOP rung: below it the draw term has not "
              "died and contaminates the arms.")
        print("    `ceiling` is the significance this design could reach AT THE MEASURED q,\n"
              "    not the q the plan assumed.  q >= 1 means the jackknife floor is at or\n"
              "    above the across-arm scatter: no detection was available at any effect\n"
              "    size, and the run yields a LIMIT.  That is what a small `a` looks like\n"
              "    from inside the estimator, not a fault in the arms.")


def halves_block(runs):
    """Print the A-vs-B pool-halves comparison for every matched pair among `runs`.

    A pair is two runs identical in everything -- features, beta, rows, grid, ladder, DRAW
    seed and SPLIT seed -- except that one drew from half A and the other from half B.  The
    draw seed must match or the difference stops being paired and the gain is meaningless.
    """
    idx = {}
    for r in runs:
        c = r["cfg"]
        half = str(c.get("pool_half", "none"))
        if half == "none":
            continue
        key = (",".join(r["features"]), float(c["beta"]), int(c["max_rows"]),
               int(c["grid_n"]), tuple(r["ladder"]), int(c["draw_seed"]),
               int(c.get("pool_split_seed", -1)))
        idx.setdefault(key, {})[half] = r
    pairs = {k: v for k, v in idx.items() if "A" in v and "B" in v}
    if not pairs:
        return
    print(f"\n{'='*94}\nPOOL HALVES: does the answer depend on WHICH half of the catalogue "
          f"supplies the prior?")
    print("  Grade on the MONOTONE RISE of the gain with M, not on the point values "
          "(doc/WORKLOG.md cont.189 §2).")
    acc = {}
    for key in sorted(pairs):
        a, b = pairs[key]["A"], pairs[key]["B"]
        ladder = list(a["ladder"])
        g = np.array([float(a["cfg"]["closure_g"]), float(a["cfg"]["closure_g2"])])
        gn = float(np.hypot(*g))
        sa, sb = stats(a, g / gn, gn), stats(b, g / gn, gn)
        rows, pin_gap = pool_halves_report(sa, sb, ladder)
        print(f"\n  {key[0]}  beta={key[1]}  rows={key[2]:,}  draw seed={key[5]}  "
              f"split seed={key[6]}")
        # THE HARD GATE FIRST.  pin never reads the pool, so if it moved, the split reached
        # something it should not have and no number below this line means anything.
        if pin_gap > 0:
            print(f"    *** PIN GATE FAILED: pin differs between the halves by "
                  f"{pin_gap*100:+.6f}%, but pin never reads the pool.\n"
                  f"        The split perturbed something it should not have.  "
                  f"NOTHING BELOW IS INTERPRETABLE.")
        else:
            print("    pin gate: identical across the halves, as it must be "
                  "(pin never reads the pool)")
        print(f"    {'M':>4} {'d_A [%]':>10} {'d_B [%]':>10} {'A-B [%]':>10} "
              f"{'sd_pair':>9} {'nsig':>6} {'gain':>7}")
        for w in rows:
            print(f"    {w['M']:>4} {w['d_a']*100:>10.3f} {w['d_b']*100:>10.3f} "
                  f"{w['diff']*100:>10.3f} {w['sd_paired']*100:>9.3f} "
                  f"{w['nsig']:>6.1f} {w['gain']:>7.2f}")
        gains = [w["gain"] for w in rows]
        rising = all(y > x for x, y in zip(gains, gains[1:]))
        print(f"    gain {'RISES monotonically' if rising else 'is NOT monotone'} "
              f"({gains[0]:.2f} -> {gains[-1]:.2f})"
              + ("" if rising else "  <- pre-registered failure mode (cont.189 §2 (i))"))
        acc.setdefault(key[:6], []).append(rows)
    halves_aggregate(acc)


def jackknife_blindness_check(sd_paired, n_blocks=200):
    """`(scatter, expected, verdict)` on whether the block jackknife is blind to the pool.

    `irreducible_term` subtracts `sd_paired**2` as a noise floor, and that is only right if
    the jackknife cannot see the pool at all.  The blindness argument is exact ONLY for the
    part of a pool change that is a COHERENT offset -- the same displacement for every
    galaxy, which enters every leave-one-block replicate identically and so has exactly zero
    jackknife variance.  Any galaxy-DEPENDENT part is a different matter: galaxies weight
    the pool rows differently, so that part IS visible to a block jackknife over galaxies.
    Where it exists, `sd_paired**2` contains a slice of the very thing being subtracted away
    from, the subtraction OVER-subtracts, and `a` comes out BIASED LOW -- the direction that
    over-claims precision, which is why it must not ride on an assumption.

    The diagnostic is free once several splits exist.  If the jackknife is genuinely blind,
    `sd_paired` should be near-identical from split to split: same galaxies, same blocks,
    same draw machinery, only pool membership changing.  Scatter beyond its own sampling
    error (a jackknife sd from B blocks carries about `1/sqrt(2(B-1))` fractional error) is
    direct evidence that the jackknife is seeing pool structure and the subtraction is
    eating real signal.
    """
    sd = np.asarray(sd_paired, dtype=float)
    if len(sd) < 2 or not np.all(sd > 0):
        return None, None, "too few splits"
    scatter = float(np.std(sd, ddof=1) / np.mean(sd))
    expected = float(1.0 / np.sqrt(2.0 * (int(n_blocks) - 1)))
    v = ("BLIND (consistent with sampling error): the floor subtraction is safe"
         if scatter <= 2 * expected else
         "NOT BLIND: sd_paired varies with the pool, so subtracting it BIASES a LOW")
    return scatter, expected, v


def halves_aggregate(acc, n_shards=2, n_blocks=200):
    """Across-split summary: A-B as a distribution, and `a` from its scatter."""
    for key, splits in sorted(acc.items()):
        n = len(splits)
        ladder = [w["M"] for w in splits[0]]
        print(f"\n  ACROSS SPLITS  {key[0]}  beta={key[1]}  rows={key[2]:,}  "
              f"draw seed={key[5]}:  {n} split(s)")
        if n < 2:
            print("    ONE SPLIT IS n = 1.  Nothing is quotable from it: a single A-B is "
                  "one draw from a\n    distribution whose width is exactly what needs "
                  "measuring (cont.189 §4).  Run more\n    --pool-split-seed values.")
            continue
        # Knowable from n alone, so it goes ABOVE the numbers: a reader who sees "no
        # detection" must be able to tell an absent effect from an unaffordable test.
        n_sp = len(splits)
        ceil = np.sqrt((n_sp - 1) / 2.0)
        print(f"    significance CEILING at {n_sp} splits: {ceil:.2f} sigma (times 1-q).  It "
              "does NOT depend on how large `a` is:\n    the across-split variance is itself "
              "estimated from n samples and the excess inherits its\n    sqrt(2/(n-1)) error "
              f"whole (cont.189 §6a).  {'A DETECTION IS UNREACHABLE at this n' if ceil < 2 else 'A 2 sigma detection is reachable'}"
              "; 2 sigma needs 9 splits, 3 needs 19.")
        print(f"    {'M':>4} {'mean A-B [%]':>13} {'sd splits [%]':>14} "
              f"{'jk floor [%]':>13} {'a signed [%]':>13} {'a quoted [%]':>16}")
        for j, m in enumerate(ladder):
            d = np.array([sp[j]["diff"] for sp in splits])
            sdp = np.array([sp[j]["sd_paired"] for sp in splits])
            a_prod, a_sd, _ = irreducible_term(d, sdp, n_shards=n_shards)
            kind, amp = a_verdict(a_prod, a_sd)
            # signed amplitude: a negative `a` is a null result and must stay visible
            signed = np.sign(a_prod) * np.sqrt(abs(a_prod)) * 100
            tag = f"{amp*100:.3f} ({'<' if kind == 'upper limit' else '='})"
            print(f"    {m:>4} {d.mean()*100:>13.3f} {d.std(ddof=1)*100:>14.3f} "
                  f"{np.sqrt((sdp**2).mean())*100:>13.3f} {signed:>13.3f} {tag:>16}")
        top = np.array([sp[-1]["sd_paired"] for sp in splits])
        sc, ex, verdict = jackknife_blindness_check(top, n_blocks=n_blocks)
        if sc is not None:
            print(f"    blindness check at the top rung: sd_paired scatter {sc:.3f} vs "
                  f"sampling error {ex:.3f}\n      -> {verdict}")
        print("    `a signed` is NOT clipped at zero: a negative draw is an ordinary null "
              "result at small n,\n    and clipping would bias a symmetric estimator upward "
              "exactly where signal is weakest.\n    `a quoted` applies the PRE-REGISTERED "
              "rule (cont.189 §6): '=' is a >2 sigma detection,\n    '<' is a one-sided 95% "
              "upper limit.  Which one is reported is fixed before the data.")
        print("    `a` is the IRREDUCIBLE finite-catalogue error, as an rms, scaled to "
              "the FULL pool.\n    Read it at the TOP rung only: below that the draw term "
              "has not died and contaminates it.\n    It assumes a ~ 1/N_pool, which halves "
              "alone cannot test -- add a second shard size.")


def main(paths, plot_path=None, min_m=1):
    panels = []
    ess_slopes = []   # AMENDMENT 5: the across-configuration test
    runs = [load(p) for p in paths]
    if not runs:
        raise SystemExit("no .npz given")
    groups = {}
    for r in runs:
        key, _ = run_identity(r)
        groups.setdefault(key, []).append(r)

    grid_pairs(runs)
    halves_block(runs)
    shards_block(runs)

    for key in sorted(groups):
        rs = groups[key]
        assert_group_is_replicates(rs)
        c0 = rs[0]["cfg"]
        feat = ",".join(rs[0]["features"])
        beta, rows = float(c0["beta"]), int(c0["max_rows"])
        grid_n, ladder = int(c0["grid_n"]), tuple(rs[0]["ladder"])
        half = str(c0.get("pool_half", "none"))
        split = int(c0.get("pool_split_seed", -1))
        g1 = float(rs[0]["cfg"]["closure_g"])
        g2 = float(rs[0]["cfg"]["closure_g2"])
        gvec = np.array([g1, g2])
        gn = float(np.hypot(*gvec))
        gdir = gvec / gn
        seeds = [int(r["cfg"]["draw_seed"]) for r in rs]
        print(f"\n{'='*94}\nfeatures = {feat}\n  beta = {beta}  rows = {rows:,}  "
              f"grid_n = {grid_n}  ladder = {list(ladder)}  |g| = {gn:.3f}"
              + (f"  POOL HALF {half} (split seed {split})" if half != "none" else ""))
        print(f"  {len(rs)} run(s), draw seeds {seeds}   ESS at M={max(ladder)}: "
              + "  ".join(f"{np.mean(r['ess']):.2f}" for r in rs)
              + f"   (efficiency {np.mean([np.mean(r['ess']) for r in rs])/max(ladder):.1%})")

        # Mean ESS per rung, averaged over rows then over runs.  Computed HERE, before the
        # correlation block, because both the parameter-free rho prediction and the
        # alternative regressor need it and they must use the same numbers.
        er_runs = [r for r in rs if r.get("ess_rung") is not None]
        er = [r["ess_rung"] for r in er_runs]
        ess_by_rung = ess_sd_by_rung = inv_ess_by_rung = ess_cv_by_rung = None
        ess_shift_by_rung = None
        if er:
            # The abscissa is weighted by each row's information when the run carries it
            # (`row_weights`); the flat mean is the fallback and its error is then UNBOUNDED,
            # not small.  All-or-nothing across the group: mixing a weighted run with a flat
            # one would average two different statistics.
            def _w(src):
                w = [row_weights(r, ladder, gdir, source=src) for r in er_runs]
                return w if all(x is not None for x in w) else None
            wr = _w("pin")
            inv_ess_by_rung, ess_by_rung, ess_cv_by_rung, ess_sd_by_rung, ess_shift_by_rung = \
                ess_abscissa(er, ladder, wr=wr)
            if wr is None:
                print("\n  [abscissa] per-row information ABSENT from these runs, so <1/ESS> is the "
                      "FLAT row mean.\n             The exact abscissa weights rows by their "
                      "information, and the gap is\n             Cov(I_i, 1/ESS_i)/<I>, expected "
                      "POSITIVE (informative rows are hard to match),\n             i.e. the flat "
                      "mean UNDERSTATES it, same sign as the Jensen gap.  UNBOUNDED HERE.\n"
                      "             Not proxyable from the block sums: blocks are interleaved "
                      "(pair % jk_blocks),\n             so a between-block covariance reads ~0 "
                      "regardless of the row-level truth.\n             Runs written from "
                      "2026-08-19 carry `<rung>_i_row` and this becomes a number.")
            else:
                worst = max(abs(v) for v in ess_shift_by_rung.values())
                print("\n  [abscissa] rows weighted by their own information along g.  Shift of "
                      "<1/ESS> vs the\n             flat row mean, per rung -- this IS "
                      "Cov(I_i, 1/ESS_i)/<I>, exactly, not to leading order:")
                print("             " + "  ".join(f"M={m}: {100*ess_shift_by_rung[m]:+.2f}%"
                                                  for m in ladder))
                # The confound is MEASURED, not assumed away.  Each rung's own I is
                # finite-draw biased by the very effect the ladder measures, so an unfrozen
                # weight drifts along the ladder for that reason alone and would forge a
                # rung-dependence.  The pin arm has no marginalisation and no draw bias.
                wpr = _w("per_rung")
                if wpr is not None:
                    _, _, _, _, sh_pr = ess_abscissa(er, ladder, wr=wpr)
                    drift = max(abs(sh_pr[m] - ess_shift_by_rung[m]) for m in ladder)
                    print("             weight FROZEN on the pin arm.  Recomputing it from each "
                          "rung's own (draw-biased) I\n             would move the shift by at most "
                          f"{100*drift:+.2f} pts: "
                          + "  ".join(f"M={m}: {100*(sh_pr[m]-ess_shift_by_rung[m]):+.2f}"
                                      for m in ladder)
                          + ("\n             *** that is comparable to the 2-pt rung-dependence "
                             "threshold below, so the frozen\n                 weight is doing real "
                             "work and the per-rung version would have forged a trend. ***"
                             if drift > 0.01 else
                             "\n             (small: the freeze is a correctness argument here, not "
                             "a numerical one)"))
                    # AND IT IS A MEASUREMENT, NOT ONLY A NUISANCE.  The unfrozen weight drifts
                    # BECAUSE per-row I is itself draw-biased, worse at low M.  So the size of
                    # the drift per rung is an independent read on the bias in the INFORMATION
                    # -- the denominator of ghat -- from the same runs, touching nothing the fit
                    # uses.  The ladder measures the numerator's bias; this measures the
                    # denominator's.  Printed at every rung, not only when it trips the warning.
                    sl, sl_sd, _ = drift_slope(er, wr, wpr, ladder)
                    if np.isfinite(sl):
                        # The resolution goes NEXT TO the threshold, not in a footnote: a
                        # quiet alarm has to be readable as "consistent with 1/M" or merely
                        # "unresolved", and only the error bar separates those.
                        res = (f" +/- {sl_sd:.2f} (draws)" if np.isfinite(sl_sd)
                               else " (bootstrap failed)")
                        print(f"             the drift itself falls as M^{sl:+.2f}{res} "
                              "-- this is the DENOMINATOR's draw\n             bias, measured "
                              "independently of the ladder, which measures the numerator's.")
                        if not np.isfinite(sl_sd):
                            print("             resolution unavailable, so NEITHER outcome is "
                                  "interpretable here.")
                        else:
                            flo = slope_floor(len(ladder))
                            thr = max(2 * sl_sd, flo)
                            dom = ("the injection systematic" if flo > 2 * sl_sd
                                   else "draw noise")
                            print(f"             detectable departure ~{thr:.2f} in the exponent, "
                                  f"set by {dom}\n             (bootstrap {2*sl_sd:.2f} vs "
                                  f"{flo:.2f} systematic at K={len(ladder)}).  The bootstrap sees "
                                  "draw noise "
                                  "ONLY:\n             injected 1/M recovers as M^-0.95 at "
                                  "+/-0.01, so the bar alone would call\n             correct data "
                                  "4 sigma discrepant.")
                            if sl + 1.0 > thr:
                                print("             *** SHALLOWER THAN 1/M by "
                                      f"{(sl + 1.0)/thr:.1f}x the detectable departure: the "
                                      "information bias\n                 outlives the shear bias, "
                                      "and a denominator bias surviving where the\n"
                                      "                 numerator's has died is invisible to the "
                                      "1/M -> 0 extrapolation. ***")
                            else:
                                print("             consistent with 1/M TO THAT RESOLUTION.  READ "
                                      "AS AN UPPER BOUND, not a\n             confirmation: a "
                                      "departure smaller than the figure above is undetectable "
                                      "here,\n             so quiet means 'not resolved', not "
                                      "'no effect'.")
                print(f"             largest {100*worst:+.2f}%.  A shift that VARIES across rungs "
                      "distorts the fit SHAPE,\n             which the GLS feels far more than a "
                      "common rescaling of the slope."
                      + ("\n             *** and it varies by more than 2% across rungs -- read the "
                         "1/M and 1/ESS fits\n                 as genuinely different models, not "
                         "as a reparameterisation. ***"
                         if max(ess_shift_by_rung.values()) - min(ess_shift_by_rung.values()) > 0.02
                         else ""))

        st = [stats(r, gdir, gn) for r in rs]
        # the pinned column knows nothing about the draws, so replicates must agree exactly
        if len(rs) > 1:
            spread = max(abs(s["pin"][0] - st[0]["pin"][0]) for s in st)
            note = ("identical, as it must be" if spread < 1e-12 else
                    f"*** DIFFER by {spread:.3%} -- the runs do NOT share their data, so "
                    f"the scatter below is NOT pure quadrature error ***")
            print(f"  pin across runs: {note}")

        print(f"\n  {'M':>5} {'d(m) per draw seed':>44}  {'mean':>9} {'scatter':>9} "
              f"{'paired':>9} {'nsig':>6}")
        summary = []
        for m in ladder:
            d = [s[m][0] - s["pin"][0] for s in st]
            # paired jackknife inside each run, then averaged: the galaxy-noise part
            pj = [float(jackknife_sigma(((s[m][1] - s["pin"][1]))[:, None])[0]) for s in st]
            mean = float(np.mean(d))
            # with >=2 draw seeds the seed scatter is the honest bar on the quadrature
            # error and SUPERSEDES the paired one, which cannot see draw-to-draw variation
            sc = (float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan)
            bar = sc if len(d) > 1 else float(np.mean(pj))
            cells = "  ".join(f"{v:+.3%}" for v in d)
            print(f"  {m:>5} {cells:>44}  {mean:>+9.3%} "
                  f"{(f'{sc:.3%}' if len(d)>1 else '  --'):>9} "
                  f"{np.mean(pj):>9.3%} {abs(mean)/max(bar,1e-12):>6.1f}")
            summary.append((m, mean, bar))

        # --- rho, and the NESTED-rung comparison, recomputed from the stored blocks ---
        # Both work for runs produced before the driver reported them, because the npz
        # keeps the per-block score and info sums -- the full sufficient statistic -- not
        # just the finished table.
        def rho(a_reps, b_reps):
            """Correlation of two arms from the three jackknife bars they imply.

            The pairing gain alone is a soft diagnostic: its "good" value depends on how
            correlated two arms have any right to be.  Independent equal-width arms
            already give gain 1/sqrt(2) = 0.71x (rho = 0); gain 1.0x is rho = 0.5.  The
            hard alarm is rho < 0 -- arms ANTI-correlating on identical data, which has
            no benign explanation.
            """
            sa = float(jackknife_sigma(a_reps[:, None])[0])
            sb = float(jackknife_sigma(b_reps[:, None])[0])
            sd = float(jackknife_sigma((b_reps - a_reps)[:, None])[0])
            den = 2.0 * sa * sb
            return ((sa ** 2 + sb ** 2 - sd ** 2) / den if den > 0 else np.nan,
                    sb / max(sd, 1e-12))

        print(f"\n  arm correlation vs pin (rho<0 is the alarm; 0.71x = independent):")
        print(f"  {'M':>5} {'rho':>7} {'gain':>8}")
        for m in ladder:
            r, gn_ = np.mean([rho(s["pin"][1], s[m][1]) for s in st], axis=0)
            print(f"  {m:>5} {r:>7.2f} {gn_:>7.2f}x")

        if len(ladder) >= 2:
            # THE TIGHT OBSERVABLE.  Rungs are nested to the BIT -- marginal_score_pass
            # keeps one running accumulator and snapshots it at each rung, so M=2 contains
            # M=1's accumulated term identically.  Same estimator family + shared atoms
            # means these DO pair, unlike the pin comparison.  Under d(M) = d_inf + a/M
            # the rung-vs-deepest differences are a*(1/M - 1/Mmax): they carry no d_inf at
            # all and measure the SLOPE alone.  Consistent `a` across rungs confirms the
            # 1/M law as a prediction; inconsistent `a` means the fit below is being
            # applied to something that is not a 1/M convergence, and its intercept is
            # meaningless however good the chi2 looks.
            top = ladder[-1]
            print(f"\n  nested-ladder slope, each rung vs the deepest (M={top}) -- these "
                  f"bars are the tight\n  ones, and here a gain near 1x IS the failure "
                  f"signature (broken nesting), not the\n  expected value.")
            print(f"  {'M':>5} {'d(m) vs Mmax':>14} {'+/-':>9} {'gain':>7} "
                  f"{'rho':>13} {'floor':>6} {'noise a/s':>13} {'implied a':>10}")
            broke, ratios = [], []
            # WHICH VECTOR rho IS COMPUTED ON, because the answer changes the prediction
            # and three different vectors are in play here:
            #   m_M            the rung's own m.  Contains the ENTIRE pin arm, so any two
            #                  rungs share almost all their noise and rho sits near 1
            #                  whatever the draws do -- the floor test would pass
            #                  VACUOUSLY.  Never test the floor on this.
            #   d_M = m_M-m_pin  what the fit is actually parametrised on, and what the
            #                  floor sqrt(M/Mmax) and the a/s inversion were derived for.
            #                  THIS is the vector rho is taken on below.
            #   D_M = d_M-d_Mmax  the DISPLAYED difference.  Here the common noise cancels
            #                  exactly and the correlation becomes parameter-FREE -- an
            #                  exact prediction rather than a floor.  Checked separately.
            dreps = {k: [s[k][1] - s["pin"][1] for s in st] for k in ladder}
            for m in ladder[:-1]:
                dv = float(np.mean([s[m][0] - s[top][0] for s in st]))
                sd = float(np.mean([jackknife_sigma((s[m][1] - s[top][1])[:, None])[0]
                                    for s in st]))
                rr = [rho_with_error(a, b) for a, b in zip(dreps[top], dreps[m])]
                r = float(np.mean([v[0] for v in rr]))
                rsig = float(np.mean([v[1] for v in rr]))
                gn_ = float(np.mean([rho(a, b)[1] for a, b in zip(dreps[top], dreps[m])]))
                fl = rho_floor(m, top)
                a_imp = dv / (1.0 / m - 1.0 / top)
                # propagate rho's bar into a/s numerically -- the map is nonlinear, so a
                # one-sided difference at +1 sigma is the honest local scale
                ns_v = implied_noise_ratio(r, m, top)
                ns_s = abs(implied_noise_ratio(min(r + rsig, 0.999), m, top) - ns_v)
                ratios.append((ns_v, ns_s))
                flag = "" if r >= fl - 2 * rsig else "  <-- BELOW FLOOR"
                if flag:
                    broke.append(m)
                print(f"  {m:>5} {dv:>14.3%} {sd:>9.3%} {gn_:>6.2f}x "
                      f"{r:>6.2f}+-{rsig:<5.2f} {fl:>6.2f} {ns_v:>7.2f}+-{ns_s:<5.2f} "
                      f"{a_imp:>10.3%}{flag}")
            print("    PRE-REGISTERED READING OF rho, fixed before the data were seen.  The "
                  "floor sqrt(M/Mmax)\n    holds for ANY mix of pin-arm and per-draw noise, "
                  "so there is nothing in it to tune.\n    rho AT the floor means per-draw "
                  "noise dominates; rho near 1 means the shared pin arm\n    does; rho "
                  "BELOW the floor is impossible under nesting and indicts the accumulator,\n"
                  "    the ladder snapshot or the seed handling -- NOT the physics, and not "
                  "the fit.")
            print("    'noise a/s' inverts rho for the pin-arm/per-draw noise ratio -- a "
                  "SECOND route to a\n    quantity the run measures directly (jackknife bar "
                  "vs seed scatter).  The two agreeing\n    is the consistency check; "
                  "disagreement invalidates the GLS covariance whatever the\n    chi2 says.")
            # a/s IS ONE NUMBER, so recovering it rung by rung is a TWO-SIDED test of the
            # noise model -- unlike the floor, which is one-sided and says nothing when it
            # passes.  With a bar on each rung the constancy becomes a chi2 rather than an
            # eyeball judgement about a 15% spread.
            if len(ratios) >= 2:
                v = np.array([a for a, _ in ratios])
                e = np.array([max(b, 1e-9) for _, b in ratios])
                w = 1.0 / e ** 2
                mu = float((w * v).sum() / w.sum())
                c2 = float((w * (v - mu) ** 2).sum())
                dofc = len(v) - 1
                pv = chi2_sf(c2, dofc)
                verdict = ("CONSISTENT with one value" if pv > 0.01 else
                           "*** NOT CONSTANT -- the noise model behind the GLS covariance "
                           "is wrong ***")
                print(f"    a/s constancy: weighted mean {mu:.3f}, chi2 = {c2:.2f}/{dofc} "
                      f"(p = {pv:.3f}) -- {verdict}")

            # EXACT check on the DISPLAYED differences.  Subtracting the deepest rung
            # cancels the common pin noise outright, so `a` disappears and the correlation
            # is fixed by the rung values alone:
            #   Var(D_M)      = s(1/M - 1/Mmax)
            #   Cov(D_M, D_N) = s(1/N - 1/Mmax)
            #   rho(D_M, D_N) = sqrt((1/N - 1/Mmax) / (1/M - 1/Mmax))
            # Nothing to tune, no noise budget to know -- the run either reproduces these
            # numbers or it does not.  Stronger than the floor, and it fires in cases the
            # floor cannot see.
            if len(ladder) >= 4:
                print(f"\n  EXACT (parameter-free) correlations among the DISPLAYED "
                      f"differences D_M = d_M - d_M{top}.\n  The common pin noise cancels "
                      f"here, so these are predictions, not bounds.")
                # AMENDMENT 4(a): the SAME algebra with ESS in place of M.  This is the
                # covariance half of the substitution, and this table is its ONLY test --
                # `C` in the fit is the empirical jackknife matrix, so swapping the error
                # model cannot move the chi2 there.  Printed side by side, never merged.
                ess_ok = inv_ess_by_rung is not None and len(ladder) >= 4
                print(f"  {'pair':>10} {'rho meas':>13} {'rho pred':>9} {'diff':>8} "
                      f"{'nsig':>6}" + (f" | {'pred(ESS)':>9} {'nsig':>6}" if ess_ok else ""))
                Dr = {k: [dreps[k][i] - dreps[top][i] for i in range(len(st))]
                      for k in ladder[:-1]}
                for i in range(len(ladder) - 2):
                    M, N = ladder[i], ladder[i + 1]
                    rr = [rho_with_error(a, b) for a, b in zip(Dr[M], Dr[N])]
                    rm = float(np.mean([q[0] for q in rr]))
                    # NOT `rs`: that is this group's list of runs, bound by the enclosing
                    # `for ..., rs in sorted(groups.items())` and read again further down.
                    # Shadowing it was latent until something below actually used it.
                    r_sd = float(np.mean([q[1] for q in rr]))
                    rp = float(np.sqrt((1.0 / N - 1.0 / top) / (1.0 / M - 1.0 / top)))
                    nsig_r = abs(rm - rp) / max(r_sd, 1e-9)
                    extra = ""
                    if ess_ok:
                        # <1/ESS>, not 1/<ESS> -- the same Jensen correction as the fit
                        # abscissa; the per-row bias law is what both are standing in for.
                        eM, eN, eT = (inv_ess_by_rung[M], inv_ess_by_rung[N],
                                      inv_ess_by_rung[top])
                        num, den = (eN - eT), (eM - eT)
                        rpe = float(np.sqrt(num / den)) if den > 0 and num > 0 else np.nan
                        nse = abs(rm - rpe) / max(r_sd, 1e-9)
                        extra = f" | {rpe:>9.3f} {nse:>6.1f}"
                    bad = "  <-- OFF" if nsig_r > 3 else ""
                    print(f"  {f'{M} vs {N}':>10} {rm:>7.3f}+-{r_sd:<5.3f} {rp:>9.3f} "
                          f"{rm - rp:>+8.3f} {nsig_r:>6.1f}{extra}{bad}")

            if broke:
                print(f"    *** rho is below the parameter-free floor at M={broke} -- the "
                      f"nesting is BROKEN.\n    Every d(m) and every fit below is "
                      f"untrustworthy; fix the accumulator first. ***")

        fit_ladder_rungs = [m for m in ladder if m >= min_m]
        if min_m > 1:
            print(f"\n  FIT RESTRICTED to M >= {min_m}: rungs "
                  f"{[m for m in ladder if m < min_m]} are excluded.  At beta>0 the M=1 "
                  f"snapshot is\n    (ll+lw)-lw = ll -- the weight cancels identically, so "
                  f"the low rungs estimate the PROPOSAL,\n    not the target, and are a "
                  f"different estimator sharing the axis.  This cut is declared on that\n"
                  f"    mechanism, not chosen from the chi2.")
        if len(fit_ladder_rungs) >= 3:
            # Deliberately NOT `ladder = fit_ladder_rungs`.  `ladder` is the group-key loop
            # target from `for (feat, beta, rows, grid_n, ladder), rs in ...`, and the whole
            # table above it means the FULL ladder.  Rebinding it here was harmless only
            # because nothing below happened to want the full one -- exactly the latent
            # shape of the `pre` shadowing that killed three GPU jobs.  ruff PLW2901 flags it.
            D = np.array([[s_[m][0] - s_["pin"][0] for m in fit_ladder_rungs] for s_ in st])
            reps = np.mean([np.stack([s_[m][1] - s_["pin"][1] for m in fit_ladder_rungs],
                                     axis=1) for s_ in st], axis=0)
            fits, note = fit_ladder(D, reps, fit_ladder_rungs)
            if note:
                print(f"\n  NOTE: {note}")
            sc = seed_scatter_check(D, reps)
            if sc:
                print(f"\n  covariance cross-check -- seed scatter vs the jackknife bar it "
                      f"must fit inside:")
                print(f"  {'M':>5} {'seed spread':>12} {'jk bar':>9} {'ratio':>7}")
                for M, (sp, br, rt) in zip(fit_ladder_rungs, sc):
                    # ONE-SIDED ON PURPOSE.  A ratio near or above 1 is an alarm: the draw
                    # scatter cannot exceed a total that contains it, so the covariance is
                    # understated.  A SMALL ratio is not the mirror-image alarm -- it just
                    # says draw noise is a minor part of the galaxy-sampling noise, which
                    # is what a converged ladder is supposed to look like.  An earlier
                    # version flagged rt < 0.35 as "covariance likely CONSERVATIVE"; that
                    # was wrong, and it would have called every healthy run suspect.
                    warn = ("  <-- EXCEEDS the bar: covariance UNDERSTATED"
                            if rt > 1.3 else "")
                    print(f"  {M:>5} {sp:>12.3%} {br:>9.3%} {rt:>7.2f}{warn}")
                S = len(D)
                print(f"    the spread column carries {S - 1} degree(s) of freedom, so its "
                      f"own fractional error is\n    ~{1/np.sqrt(2*(S-1)):.0%}; read the "
                      f"column as a whole and only trust a ratio above 1 if several rungs "
                      f"show it.\n    Small ratios are EXPECTED (draw noise is a minor part "
                      f"of galaxy-sampling noise) and are\n    not evidence that the "
                      f"covariance is inflated.")
            if ess_by_rung is not None:
                em = np.array([ess_by_rung[m] for m in fit_ladder_rungs], dtype=float)
                inv = 1.0 / np.maximum(em, 1e-12)
                # THE INTERCEPT IS AT 1/ESS = 0 AND NO RUNG REACHES IT.  Quoting d_inf
                # without the lever hides how far the extrapolation runs: a short span with
                # a bad chi2 gives an intercept the bar flatters.  Same discipline as
                # quoting a minimum detectable offset.
                print(f"\n  extrapolation lever: 1/ESS spans {inv.min():.3f} - {inv.max():.3f} "
                      f"(factor {inv.max()/max(inv.min(),1e-12):.1f}); the nearest rung sits "
                      f"{inv.min():.3f}\n    from the 1/ESS = 0 intercept, so d_inf below is an "
                      f"extrapolation to a regime no rung reaches.")
                # IS 1/ESS JUST A REPARAMETRISED 1/M?  If ESS = c*M^p to within the noise,
                # then fitting against 1/ESS is fitting against M^-p, which hands the fit a
                # free exponent on the abscissa.  Any monotone reparametrisation of that
                # kind flatters a marginal chi2 without adding information -- verified by
                # driving the 4(b) path with a SYNTHETIC ESS = M^0.9, which moved chi2/dof
                # 2.73 -> 1.83 on data that had gained nothing.  So this residual is the
                # number that says how much the 4(b) comparison below is worth.
                print(f"\n  {'M':>5} {'<ESS>':>9} {'CV(ESS)':>9} {'1/<ESS>':>9} "
                      f"{'<1/ESS>':>9} {'ratio':>7}")
                for m in fit_ladder_rungs:
                    a_, b_ = 1.0 / ess_by_rung[m], inv_ess_by_rung[m]
                    print(f"  {m:>5} {ess_by_rung[m]:>9.3f} {ess_cv_by_rung[m]:>9.3f} "
                          f"{a_:>9.4f} {b_:>9.4f} {b_/max(a_,1e-12):>7.3f}")
                jmax = max(inv_ess_by_rung[m] * ess_by_rung[m] for m in fit_ladder_rungs)
                print(f"    THE FIT USES <1/ESS> (the last column).  The bias law is per row,"
                      f" so the abscissa is a\n    MEAN OF RECIPROCALS; 1/<ESS> is a "
                      f"different number whenever ESS varies across rows, by\n    "
                      f"1 + Var(ESS)/<ESS>^2 to leading order.  Largest gap here: "
                      f"{jmax:.3f}x ({jmax - 1:+.1%}).")
                if jmax > 1.05:
                    print(f"    THIS IS A BIAS, NOT A VARIANCE -- more rows measure it "
                          f"better, they do not shrink it,\n    and it is NOT covered by "
                          f"AMENDMENT 7's inflate-only asymmetry because it can move the\n"
                          f"    chi2 either way.  Any 4(b) number computed against 1/<ESS> "
                          f"would not be interpretable.")
                lm = np.log(np.array(fit_ladder_rungs, dtype=float))
                le = np.log(np.maximum(em, 1e-12))
                if len(lm) >= 3:
                    pslope, pint = np.polyfit(lm, le, 1)
                    resid = le - (pslope * lm + pint)
                    rms = float(np.sqrt(np.mean(resid ** 2)))
                    print(f"    ESS = {np.exp(pint):.3f} * M^{pslope:.4f}; log-residual rms "
                          f"= {rms:.4f} ({np.expm1(rms):+.2%} in ESS).")
                    if rms < 0.02:
                        # A REPARAMETRISATION IS NOT A FREE PASS.  The worry that it might
                        # be was pre-registered (AMENDMENT 5(ii)) and then measured and
                        # withdrawn (AMENDMENT 6): on synthetic ladders whose truth is
                        # exactly 1/M, refitting against M^-0.9 gives chi2/dof of 465, 43
                        # and 5.9 across a 10x noise range -- rejected everywhere.  The GLS
                        # sees the SHAPE of the abscissa because the rungs are strongly
                        # correlated, so a wrong reparametrisation is visible, and the 4(b)
                        # chi2 below is a real test.  Test:
                        # test_the_wrong_abscissa_is_STRONGLY_rejected_not_flattered.
                        print(f"    ESS is a power law in M here to within "
                              f"{np.expm1(rms):.1%}, so 1/ESS is M^-{pslope:.2f} to that "
                              f"accuracy.\n    That does NOT make the 4(b) chi2 below "
                              f"vacuous: the exponent comes from the MEASURED ESS, not "
                              f"from\n    fitting d(m), and a wrong abscissa is strongly "
                              f"rejected rather than absorbed (AMENDMENT 6).\n    The "
                              f"across-configuration test -- is the ESS-unit slope `a` "
                              f"COMMON to configurations whose\n    1/M slopes differ by an "
                              f"order of magnitude -- remains the stronger of the two.")
            else:
                print(f"\n  extrapolation lever: per-rung ESS not stored in these runs "
                      f"(pre-dates ess_rung), so the\n    distance from the nearest rung to the "
                      f"intercept is UNQUANTIFIED here.")
            print(f"\n  {'fit':>20} {'d_inf':>10} {'+/-':>8} {'a':>10} {'+/-':>8} "
                  f"{'b':>10} {'chi2/dof':>9} {'p':>7} {'infl +/-':>9}")
            for name, bh, cov, c2, mask, order in fits:
                bstr = (f"{bh[2]:>+10.3%}" if order >= 2 else f"{'--':>10}")
                doff = max(int(mask.sum()) - (order + 1), 1)
                pv = chi2_sf(c2 * doff, doff)
                # A GLS bar is computed UNDER the fitted model.  When chi2/dof sits above 1
                # the data are rejecting that model, so the bar is not the honest
                # uncertainty on the intercept -- either the functional form is wrong or the
                # covariance is still understated, and both make it too small.  Inflating by
                # sqrt(chi2/dof) is the standard minimum response.  It is a FLOOR on the
                # uncertainty, not a fix: it assumes the misfit is unmodelled scatter rather
                # than a wrong law, which is exactly what is in doubt when chi2 is high.
                infl = max(1.0, np.sqrt(c2))
                print(f"  {name:>20} {bh[0]:>+10.3%} {np.sqrt(cov[0,0]):>8.3%} "
                      f"{bh[1]:>+10.3%} {np.sqrt(cov[1,1]):>8.3%} {bstr} {c2:>9.2f} "
                      f"{pv:>7.3f} {np.sqrt(cov[0,0])*infl:>9.3%}")
            bh, cov, chi2_dof = fits[0][1], fits[0][2], fits[0][3]
            print("    QUOTE THE INFLATED BAR when p is small: the plain GLS bar assumes "
                  "the fitted law,\n    and a law the data reject cannot certify its own "
                  "uncertainty.")
            print("    d_inf is the closure residual: BOTH likelihoods are exact for these "
                  "data, so it\n    must be zero.  a is the price of the quadrature: "
                  f"|bias| < 0.1% needs M ~ {abs(bh[1])/0.001:.0f} draws here.")
            print("    chi2 uses a Hartlap-corrected inverse; jackknife replicates are not "
                  "independent,\n    so N is a generous count and these bars stay slightly "
                  "optimistic.")
            if len(fits) > 1:
                mv, moved = intercept_spread(fits)
                msg = (("*** MOVES by %.3f%% -- the extrapolation is MODEL-DEPENDENT; "
                        "quote the SPREAD\n    as the systematic, not any single fit ***")
                       % (100 * mv) if moved else f"stable (max shift {mv:.3%})")
                print(f"    intercept across fit variants: {msg}")
            chi2, dof = chi2_dof, 1
            panels.append((f"{feat.count(',')+1}D: {feat[:34]}"
                           + (f"  beta={beta}" if beta else "  uniform"),
                           summary, (bh[0], bh[1])))
            if chi2 / dof > 3:
                print(f"    chi2/dof = {chi2/dof:.1f} -- the 1/M law does NOT describe this "
                      f"ladder; the leading\n    log-of-an-unbiased-estimate term is not the "
                      f"whole story and the extrapolation is not usable.")

            # ---- AMENDMENT 4(b): the bias law refitted against 1/ESS -----------------
            # DECLARED SEPARATELY AND PRINTED SEPARATELY, never merged with the table
            # above, because the two halves of the substitution are different kinds of
            # claim.  Variance ~ 1/ESS is what ESS MEANS.  Bias ~ 1/ESS is a CONJECTURE:
            # the finite-M bias of a log-mixture is O(1/M) for unweighted i.i.d. draws, and
            # that it tracks ESS instead is plausible (both are driven by the spread of the
            # contributions) but is not derived anywhere here.  A pass below is evidence
            # FOR that conjecture, not a fix to be quietly adopted.
            #
            # CORRECTION TO THE PRE-REGISTRATION, recorded rather than papered over.
            # Amendment 4 declared 4(a) -- "ESS in the covariance only" -- as a variant that
            # could restore p > 0.05 on its own.  It cannot.  `C` in `fit_ladder` is the
            # EMPIRICAL jackknife covariance; there is no modelled 1/M anywhere in it, so M
            # reaches the chi2 only through the regressor, which is 4(b).  4(a)'s only
            # observable is the parameter-free rho prediction, printed above with its ESS
            # column beside the 1/M one.  Reading 4(a) as "did the chi2 improve" would be
            # reading a number that is arithmetically incapable of moving.
            if inv_ess_by_rung is not None and len(fit_ladder_rungs) >= 3:
                xe = np.array([inv_ess_by_rung[m] for m in fit_ladder_rungs], dtype=float)
                fits_e, _ = fit_ladder(D, reps, fit_ladder_rungs, x=xe, xname="1/ESS")
                print(f"\n  AMENDMENT 4(b) -- SAME data, SAME covariance, regressor 1/ESS "
                      f"instead of 1/M.\n  A CONJECTURE under test, not a correction; "
                      f"compare chi2 against the table above.")
                print(f"  {'fit':>20} {'d_inf':>10} {'+/-':>8} {'a':>10} {'+/-':>8} "
                      f"{'b':>10} {'chi2/dof':>9} {'p':>7} {'infl +/-':>9}")
                for name, bhe, cove, c2e, maske, ordere in fits_e:
                    bstr = (f"{bhe[2]:>+10.3%}" if ordere >= 2 else f"{'--':>10}")
                    doff = max(int(maske.sum()) - (ordere + 1), 1)
                    infl = max(1.0, np.sqrt(c2e))
                    print(f"  {name:>20} {bhe[0]:>+10.3%} {np.sqrt(cove[0,0]):>8.3%} "
                          f"{bhe[1]:>+10.3%} {np.sqrt(cove[1,1]):>8.3%} {bstr} {c2e:>9.2f} "
                          f"{chi2_sf(c2e * doff, doff):>7.3f} "
                          f"{np.sqrt(cove[0,0])*infl:>9.3%}")
                d1 = max(int(fits[0][4].sum()) - 2, 1)
                p1 = chi2_sf(fits[0][3] * d1, d1)
                de = max(int(fits_e[0][4].sum()) - 2, 1)
                pe = chi2_sf(fits_e[0][3] * de, de)
                # The two fits are NOT nested and share the data, so this is a comparison of
                # two descriptions, not a significance test.  Say which it is.
                verdict = ("1/ESS describes the ladder better" if pe > p1 else
                           "1/M describes the ladder at least as well")
                print(f"    p(1/M) = {p1:.3f} vs p(1/ESS) = {pe:.3f}: {verdict}.  These are "
                      f"NOT nested\n    models and they share the same data, so this is a "
                      f"comparison of two descriptions,\n    not a significance test -- no "
                      f"delta-chi2 interpretation is available.")
                print(f"    the 4(b) intercept sits at 1/ESS = 0, {xe.min():.3f} beyond the "
                      f"nearest rung\n    (span {xe.min():.3f} - {xe.max():.3f}, factor "
                      f"{xe.max()/max(xe.min(),1e-12):.1f}).")
                ess_slopes.append((f"{feat[:22]} b={beta}", float(fits_e[0][1][1]),
                                   float(np.sqrt(fits_e[0][2][1, 1])),
                                   float(fits[0][1][1])))

                # ERRORS IN VARIABLES ON THE ABSCISSA -- the peer session's point, and the
                # decisive caveat on everything above.  The chi2 responds to the SHAPE of
                # the abscissa with enormous gain (1.20 -> 465 for a 10% exponent change),
                # and 1/ESS is measured, not known.  A test that sensitive cannot be read
                # while its x-axis is treated as exact.  Enumerating every sign pattern of a
                # +/-1 sigma shift is exhaustive at these rung counts and needs no
                # linearisation, so it bounds the chi2 rather than estimating it.
                sde = np.array([ess_sd_by_rung[m] for m in fit_ladder_rungs], dtype=float)
                rel = sde / np.maximum(xe, 1e-12)
                K = len(fit_ladder_rungs)
                c2s = []
                if K <= 12:
                    for signs in itertools.product((-1.0, 1.0), repeat=K):
                        x_p = xe + np.array(signs) * sde
                        if np.any(x_p <= 0):
                            continue
                        f_p, _ = fit_ladder(D, reps, fit_ladder_rungs, x=x_p)
                        c2s.append(f_p[0][3])
                if c2s:
                    lo, hi = float(min(c2s)), float(max(c2s))
                    dd = max(int(fits_e[0][4].sum()) - 2, 1)
                    plo, phi = chi2_sf(hi * dd, dd), chi2_sf(lo * dd, dd)
                    print(f"    per-rung ESS is measured to {rel.min():.2%}-{rel.max():.2%} "
                          f"(DRAW noise only -- the pool is fixed,\n    so a pool-level shift "
                          f"is invisible to this bar); shifting every rung by +/-1 sigma\n"
                          f"    (all {len(c2s)} sign patterns) puts chi2/dof in "
                          f"[{lo:.2f}, {hi:.2f}], p in [{plo:.3f}, {phi:.3f}].")
                    if plo <= 0.05 <= phi:
                        print(f"    *** THE ABSCISSA ERROR ALONE SPANS p = 0.05, so 4(b) "
                              f"CANNOT DECIDE the question at\n    this ESS precision.  Do "
                              f"not report whichever side the central value landed on.\n"
                              f"    Pre-registered in AMENDMENT 7; the across-configuration "
                              f"test below is unaffected. ***")
                    else:
                        print(f"    the p = 0.05 threshold is outside that range, so the "
                              f"verdict is not an artefact of\n    the abscissa "
                              f"uncertainty.")
            elif inv_ess_by_rung is None:
                print(f"\n  AMENDMENT 4(b) not evaluated: these runs pre-date `ess_rung`, "
                      f"so there is no\n    per-rung ESS to regress against.  Re-run to test "
                      f"it; do not infer it from the M=Mmax ESS alone.")


    # ---- AMENDMENT 5: is the ESS-unit slope COMMON across configurations? --------------
    # THE STRONGER OF THE TWO ESS TESTS, and the reason is that it cannot be bought with a
    # reparametrisation.  Within one ladder, swapping 1/M for 1/ESS changes the abscissa and
    # the chi2 responds; that is a real test (AMENDMENT 6) but it is one configuration's
    # worth of evidence.  ACROSS configurations the claim has teeth: if the finite-draw bias
    # really tracks ESS, then the slope expressed per unit 1/ESS must be the SAME number for
    # configurations whose slopes per unit 1/M differ by an order of magnitude.  One exponent
    # cannot reconcile three such configurations unless the conjecture is true.
    if len(ess_slopes) >= 2:
        print(f"\n{'='*94}\nAMENDMENT 5 -- IS THE ESS-UNIT SLOPE COMMON ACROSS "
              f"CONFIGURATIONS?\n  If bias ~ 1/ESS, `a` in ESS units is one number; the 1/M "
              f"column shows what it has to explain.")
        print(f"  {'configuration':>26} {'a (1/ESS units)':>17} {'+/-':>9} "
              f"{'a (1/M units)':>15} {'ratio':>8}")
        for lab, a_e, sd_e, a_m in ess_slopes:
            ratio = a_m / a_e if abs(a_e) > 1e-12 else np.nan
            print(f"  {lab:>26} {a_e:>+17.3%} {sd_e:>9.3%} {a_m:>+15.3%} {ratio:>8.2f}")
        av = np.array([q[1] for q in ess_slopes])
        sv = np.array([max(q[2], 1e-12) for q in ess_slopes])
        w = 1.0 / sv ** 2
        abar = float((w * av).sum() / w.sum())
        chi2 = float((w * (av - abar) ** 2).sum())
        dof = len(av) - 1
        pv = chi2_sf(chi2, dof)
        spread = float(np.max(av) - np.min(av))
        mspread = float(np.max(np.abs([q[3] for q in ess_slopes]))
                        / max(np.min(np.abs([q[3] for q in ess_slopes])), 1e-12))
        print(f"  common value {abar:+.3%},  chi2 = {chi2:.2f}/{dof} (p = {pv:.4f}); the "
              f"1/M slopes span a factor {mspread:.0f}.")
        if pv > 0.05:
            print(f"    CONSISTENT with a single ESS-unit slope.  This is the evidence FOR "
                  f"bias ~ 1/ESS, and it\n    is the claim to quote -- a reparametrisation "
                  f"cannot produce agreement across\n    configurations that disagree by a "
                  f"factor {mspread:.0f} in 1/M units.")
        else:
            print(f"    NOT consistent: spread {spread:.3%} across configurations at "
                  f"p = {pv:.4f}.  The bias does\n    NOT simply track ESS, whatever the "
                  f"within-ladder chi2 says.  Report 4(b) as a better\n    DESCRIPTION of "
                  f"each ladder and not as a mechanism.")
        print(f"    Two or three configurations give {dof} degree(s) of freedom, so this "
              f"test is not powerful;\n    a pass is 'not excluded', not 'established'.")
    elif ess_slopes:
        print(f"\n  AMENDMENT 5 across-configuration test needs at least 2 configurations "
              f"with per-rung\n    ESS; only {len(ess_slopes)} present here.")

    if plot_path and panels:
        plot(panels, plot_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--plot", default=None, help="write a d(m) vs 1/M figure here")
    ap.add_argument("--min-m", type=int, default=1,
                    help="drop ladder rungs below this M before FITTING (the table still "
                         "shows them).  Use it for beta>0 runs: at M=1 the self-normalised "
                         "weight cancels identically (acc-lw_run = (ll+lw)-lw = ll), so the "
                         "low rungs run the PROPOSAL estimator, not the target one, and do "
                         "not lie on the same 1/M law.  State the cut when quoting a fit.")
    a = ap.parse_args()
    main(sorted({p for q in a.paths for p in (glob.glob(q) or [q])}), a.plot, a.min_m)
