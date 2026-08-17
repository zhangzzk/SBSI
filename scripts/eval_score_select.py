#!/usr/bin/env python -B
"""§5B end to end WITH selection: the full (5.3), on the real flow, against a known shear.

WHAT THIS CLOSES.  `WORKLOG.md` cont.164 lists three defects in the §5B implementation.  This
script exercises the fix for the second and most consequential one: the estimator had no
`I_sel`, so it centred the score's numerator and left the denominator alone.  `INFERENCE.md`
A.7 prices that at `m = -64%` for a cut on the median, and `tests/test_population_terms.py`
now reproduces that number from the production code.  Here it runs on the real flow.

THE CUT.  It must be a function of the flow's OUTPUT, or `P_pass` does not exist (§4.7).  On a
V1-shaped model, whose only output is the measured shape, that left exactly one expressible
cut, `|xhat| < c`.  On the V3 flow the output is `(g1, g2, mag_auto, log flux_radius)`, so a
measured MAGNITUDE or SIZE cut is available too -- the cut an analysis would actually apply,
and the reason this script was moved onto V3.  `--cut-bound NAME:LO:HI` names any output
column; `OutputCut` refuses a name the flow does not predict rather than let §4.7 be violated
quietly.

All three are ISOTROPIC in the shape angle, so §5B.2's prediction applies to each: for a
spin-2 shear an isotropic cut kills the numerator term by orientation averaging and leaves
`I_sel` as the only surviving selection correction.

    <s>_sel  ~ 0     (both components, by isotropy)
    I_sel    ~ isotropic, and NOT small
    ghat with both corrections -> the injected g;  without I_sel -> biased low.

They are NOT the same test, though, because `Pi(e)` reaches the shape grid by a different
route in each: through the measurement noise of the shape itself for `|xhat|`, and through
whatever correlation the flow carries between true shape and measured flux/size for the other
two.  A cut that is weakly shape-dependent has a small `I_sel` and is a weak test; the keep
FRACTION is matched across the three (about 70% each, measured on the V3 domain) so that the
comparison is between cut channels and not between sample sizes.

CLOSURE, SO THERE IS A TRUTH.  Data are drawn FROM the flow at a known shear, exactly as
`eval_score_response.py --mode closure` does, so the likelihood is exact and any departure of
`ghat` from `g` is the estimator's own.  The cut is then applied to those draws.

Pi_k, CHEAPLY.  The population block needs `Pi_k = P(xhat in S | true shape = e_k)` averaged
over the population of the OTHER true properties (`population_log_pi`'s docstring explains why
that average is needed at all -- this bank is per-galaxy in its conditioning).  Estimating it
costs one flow sample per (subsample row, node), not a nested integral: substitute the node's
shape into a subsample of real rows, sample, and count what passes.  With a few hundred rows
and a few samples each that is a few million draws.

Usage (Slurm, GPU):
    python scripts/eval_score_select.py --closure-g 0.05 --cut-abs-ehat 0.6
    python scripts/eval_score_select.py --closure-g 0.05 --cut-abs-ehat none \
        --cut-bound measured_mag_auto::24.5
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbsi.measurement_model import load_measurement_model  # noqa: E402
from sbsi.models import get_model  # noqa: E402
from sbsi.posterior_shape import (  # noqa: E402
    _E_DERIVED_FEATURES, PosteriorShapeEstimator, make_e_grid,
)
from sbsi.precision import MODES, precision_region  # noqa: E402
from sbsi.preprocessing import DEFAULT_SELECTION_CUTS, rescale  # noqa: E402
from sbsi.score_inference import (  # noqa: E402
    OutputCut, ScoreCacheMismatch, ShapeScoreNodes, blocked_sums, jackknife_blocks,
    jackknife_sigma, merge_block_sum_caches, population_terms,
)
from sbsi.shear_map import apply_shear_to_ellipticity  # noqa: E402

from eval_score_response import (  # noqa: E402
    G0_CAT, LL_DTYPE, PRIOR_CACHE, build_prior, load_g0, score_pass,
)

# One seed of the V3 ensemble.  A closure test needs ONE flow, not the ensemble mean: the
# data are drawn from this flow and inverted with this flow, so the model is exact by
# construction and the residual is the estimator's own.  Averaging 16 seeds would generate
# from a different density than it estimates with and quietly stop being a closure test.
MODEL = str(get_model("V3").flow_checkpoints[0])


def primary_domain_cuts(metadata):
    """`DEFAULT_SELECTION_CUTS` narrowed to the primary domain THIS checkpoint was trained on.

    Read from metadata, never inferred from the filename (AGENTS.md).  The V3 flows carry
    `primary_mag_max=25.8` and `primary_re_min=0.5`; a checkpoint that carries neither gets
    the default domain back unchanged, which is what the V1 runs used.
    """
    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    mag, re_min = metadata.get("primary_mag_max"), metadata.get("primary_re_min")
    if mag is not None:
        cuts[1][1] = float(mag)
    if re_min is not None:
        cuts[3][0] = float(re_min)
    if cuts == [list(c) for c in DEFAULT_SELECTION_CUTS]:
        return None, (mag, re_min), ""          # nothing narrowed: the default domain
    tag = "dom" + "".join(f"{v:g}".replace(".", "") for v in (cuts[1][1], cuts[3][0]))
    return cuts, (mag, re_min), tag


def azimuthally_average(grid, pi, n_bins):
    """Replace `Pi_k` by the mean over all nodes at the same `|e|`, killing angular structure.

    THE TEST THIS EXISTS FOR.  §5B.2 predicts `<s>_sel = 0` for an isotropic cut, and the
    measured value is 138-169 sigma from zero (cont.176).  `scripts/check_flow_isotropy.py`
    then showed the flow is not isotropic in shape: `Pi` on rings of constant `|e|` varies by
    65-79x its own noise at large `|e|`.  Those two facts are consistent but not causally
    linked by anything yet -- deciding needs the harmonic content of the prior score `u`,
    which is NOT purely spin-2 (the shear map is nonlinear in `e`, so `u` carries m=0, m=2 and
    m=4, and `Pi`'s m=4 term can couple to it too).

    Rather than do that algebra, do the experiment: force `Pi` to depend on `|e|` alone and
    see whether `<s>_sel` survives.  If it collapses toward zero, the anisotropy IS the cause
    and the fix is in the model.  If it survives, the anisotropy is a red herring for this
    term and the fault is in `ShapeScoreNodes` / `population_terms` / the grid quadrature.

    WHAT THIS IS NOT.  It is a diagnostic, never a correction: it discards real structure the
    model predicts, so a `d(m)` computed from an averaged `Pi` is not a calibration result and
    must not be quoted as one.  Compare it only against the raw run off the SAME score cache,
    which makes the pair differ in `Pi` and nothing else.

    Radial structure is preserved by interpolating the binned means back to each node's own
    `|e|`, so the only thing removed is variation AT FIXED radius.
    """
    r = np.hypot(np.asarray(grid, float)[:, 0], np.asarray(grid, float)[:, 1])
    edges = np.linspace(r.min(), r.max() + 1e-12, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    out = np.empty_like(pi)
    for j in range(pi.shape[-1]):
        col = pi[..., j]
        tot = np.bincount(idx, weights=col, minlength=n_bins)
        cnt = np.bincount(idx, minlength=n_bins).astype(float)
        ok = cnt > 0
        centres = 0.5 * (edges[:-1] + edges[1:])
        out[..., j] = np.interp(r, centres[ok], (tot[ok] / cnt[ok]))
    return out


def pass_fraction_by_node(bundle, df, grid, rk, cut, n_samples, batch_size, seeds,
                          ladder, rng):
    """`Pi_k = P(xhat in S | true shape = e_k)`, averaged over the population.

    One flow sample per (row, node, draw).  The node's shape is substituted into real rows,
    so every other conditioning variable keeps its catalogue value and the average over rows
    IS the marginalisation over the rest of the scene prior.

    COMMON RANDOM NUMBERS ACROSS NODES, which is not a refinement but the difference between
    a usable `Pi` and an unusable one.  `<s>_sel = E_Pi[u]` is a ratio of two integrals that
    very nearly cancel -- for an isotropic cut it should vanish outright -- so INDEPENDENT
    Bernoulli noise at each node does not average away, it lands directly on the answer.
    Measured on a 16-draw pilot it produced a spurious `<s>_sel = 0.049` that removed two
    thirds of the estimator's numerator.  Re-seeding identically for every node makes `Pi_k`
    a smooth function of `k`: the latents are shared, so neighbouring nodes differ by the
    signal rather than by noise.

    ERROR BARS COME FROM INDEPENDENT REALISATIONS, not from splitting the rows.  Sharing
    latents across nodes is what makes `Pi` smooth, but it also means one realisation
    imprints a fixed pattern on the whole node bank -- and a split-half over ROWS cannot
    see it, because both halves carry the same latents.  Measured, that fake error bar
    reported a 34-sigma detection of a quantity that is zero by symmetry.  This is exactly
    the trap §5B.4 records for nested ladders, met again one level down, and the cure is
    the same: independent replicates.  `seeds` therefore drives `n_rep` full sweeps, and
    the scatter ACROSS them is the honest uncertainty.

    RANDOM ROWS, AND A FRESH DRAW PER REPLICATE.  The population rows used to be the
    catalogue's first `M` -- a PREFIX -- and the rungs of the convergence ladder were then
    nested inside one another.  That is the nested-ladder trap of §5B.4 for a third time:
    five apparently independent points all inherit whichever way the one prefix happened to
    be unrepresentative, so the ladder shows a smooth monotone "convergence" that is really
    a single correlated draw relaxing toward the truth.  Measured, the prefix ran 1.3-1.7
    sigma faint in `measured_mag_auto`; fainter galaxies have noisier measured shapes and so
    a lower pass fraction, and `<Pi>` duly came out biased LOW at every rung (-3.6%, -2.1%,
    and -3.0/-1.4/-0.5% in the previous run) with a sign that never flipped.  Rows are now
    drawn at random WITHOUT replacement, and each replicate draws its own set, so the
    scatter across replicates measures the row-sampling error as well as the latent noise.
    The rungs stay nested WITHIN a replicate (which is what keeps the ladder free) but are
    independent ACROSS replicates, so each rung's error bar is honest even though the rungs'
    central values remain correlated with one another.

    THE LADDER IS FREE.  `Pi` at a smaller population sample is a prefix of the rows of that
    replicate's random permutation, so `ladder=[256, 1024, ...]` returns every rung from the
    one sweep.  That matters because the expensive part of this script is the per-object
    pass, not `Pi`, while `Pi`'s convergence is the leading systematic.

    THE RESIDUAL FLOW CANNOT SEE THE NODE, SO IT IS DRAWN ONCE.  This loop used to rebuild
    the whole conditioning frame per node -- `sub.copy()`, `rescale()`, and the
    preprocessor's `transform_frame` inside `bundle.sample` -- 2765 times per replicate on
    tens of thousands of rows, to change two columns.  Worse, it re-ran the flow each time.
    But the model is a location family in the shape:

        ConditionalMeanFlow.sample(c) = flow.sample(flow_ctx(c)) + mu(c),

    and `flow_ctx` index-selects away exactly `flow_drop_indices = [0, 1, 8, 9]` -- `e1`,
    `e2` and their missing indicators.  The 10-layer coupling stack is therefore BLIND to
    which node we are at, and the common random numbers this docstring already insists on
    mean its draws at node `k` are not merely similar to its draws at node `k'`, they are
    the same draws.  So the whole sweep is one flow pass plus `G` evaluations of the
    16->128->2 mean head, about 590x less arithmetic, with the node entering only through
    `mu`.  This is the same structure `PosteriorShapeEstimator` asserts for the scoring
    side; it was simply never exploited on the sampling side.

    It also makes the CRN exact rather than seed-dependent: the latents are now literally
    one array reused, instead of two reseedings that happen to coincide.

    `cut` is the `OutputCut`, applied to RAW `(m, n_samples, D)` samples as a torch tensor
    and returning a `(m, n_samples)` mask, so the per-node reduction never leaves the GPU;
    only the `(n_ladder,)` rung values come back.  It is the SAME object the score pass
    uses on its NumPy draws, which is what keeps `Pi` describing the sample being scored.
    Returns `Pi (n_ladder, G, n_rep)`.
    """
    g = np.asarray(grid, float)
    ladder = list(ladder)
    m_max = max(ladder)
    if m_max > len(df):
        raise ValueError(f"ladder rung {m_max} exceeds the {len(df)} rows available")

    model, pre, tstd = bundle.model, bundle.condition_preprocessor, bundle.target_transform
    dev = bundle.device
    i1, i2 = pre.feature_names.index("e1_input_p"), pre.feature_names.index("e2_input_p")
    n_feat = len(pre.feature_names)
    # The reuse is exact only if the residual flow really is blind to the shape columns.
    # Assert it here rather than inherit it from `PosteriorShapeEstimator`, because this
    # function does not construct one and a future checkpoint could keep `e` in the flow.
    e_dims = {i1, i2}
    if pre.add_missing_indicators:
        e_dims |= {n_feat + i1, n_feat + i2}
    seen = e_dims & set(int(v) for v in model.keep_indices.tolist())
    if seen:
        raise ValueError(f"residual flow sees shape context dims {sorted(seen)}; its draws "
                         "would depend on the node and cannot be reused across the bank")
    bad = _E_DERIVED_FEATURES & set(pre.feature_names)
    if bad:
        raise ValueError(f"feature set contains e-derived features {sorted(bad)}; setting "
                         "e1/e2_input_p alone would leave the context inconsistent")

    gs = torch.as_tensor(
        np.stack([(g[:, 0] - pre.means[i1]) / pre.scales[i1],
                  (g[:, 1] - pre.means[i2]) / pre.scales[i2]], axis=1),
        dtype=torch.float32, device=dev)
    sc = torch.as_tensor(np.asarray(tstd.scales, dtype=np.float32), device=dev)
    mn = torch.as_tensor(np.asarray(tstd.means, dtype=np.float32), device=dev)
    rungs = torch.as_tensor([mm - 1 for mm in ladder], dtype=torch.long, device=dev)
    norm = torch.as_tensor([float(mm) for mm in ladder], dtype=torch.float32, device=dev)

    subs = [df.iloc[rng.choice(len(df), m_max, replace=False)].reset_index(drop=True)
            for _ in seeds]
    out = np.empty((len(ladder), len(g), len(seeds)), dtype=np.float64)
    for j, (sd, sub) in enumerate(zip(seeds, subs)):
        ctx0 = torch.as_tensor(pre.transform_frame(rescale(sub.copy(), **rk)),
                               dtype=torch.float32, device=dev)
        torch.manual_seed(sd)                      # SAME latents at every node -> CRN
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sd)
        with torch.no_grad():
            # batched in the SAME order and size `bundle.sample` used, so the latent stream
            # -- and hence every number this function returns -- is unchanged by the rewrite
            resid = torch.cat([model.flow.sample(model._flow_ctx(ctx0[s:s + batch_size]),
                                                 n_samples=n_samples)
                               for s in range(0, len(ctx0), batch_size)], dim=0)
            ctx = ctx0.clone()                     # scratch: only the shape columns move
            if pre.add_missing_indicators:
                ctx[:, n_feat + i1] = ctx[:, n_feat + i2] = 0.0   # grid e is never missing
            for k in range(len(g)):
                ctx[:, i1], ctx[:, i2] = gs[k, 0], gs[k, 1]
                xh = (resid + model._mu(ctx)[:, None, :]) * sc + mn        # (m, ns, D) raw
                cs = torch.cumsum(cut(xh).to(torch.float32).mean(dim=1), dim=0)
                out[:, k, j] = (cs[rungs] / norm).double().cpu().numpy()
                if k % 500 == 0:
                    print(f"    Pi rep {j+1}/{len(seeds)} node {k:>6,}/{len(g):,}  "
                          f"<Pi>={out[-1, :k+1, j].mean():.4f}", flush=True)
    return out


def _opt_float(text):
    """argparse type: a float, or `none`/`off` for "this cut component is absent"."""
    return None if str(text).strip().lower() in ("none", "off", "") else float(text)


def score_catalogue(args, nodes, bundle, grid, legs, cut, n_legs):
    """The expensive half: `(s_i, I_i)` for every object, reduced to per-block sums.

    Returns `(blk_kept, blk_uncut_or_None, n_keep, n_tot)`, each `blk` being the
    `(cnt, ns, ni)` triple of `blocked_sums`.  Nothing here depends on the population
    block, which is exactly why it can be cached.
    """
    est = PosteriorShapeEstimator(bundle, grid, device=args.device)
    acc = {"kept": [[], [], []], "uncut": [[], [], []]}
    n_keep = n_tot = 0
    for li, (frl, ehl, prl) in enumerate(legs()):
        blk = prl % args.jk_blocks          # ring partners and repeats share a block
        kl = cut(ehl)
        n_keep += int(kl.sum())
        n_tot += len(kl)
        # SCORE EACH LEG ONCE.  `(s_i, I_i)` are per-object -- they depend on that row's
        # own conditioning and its own `xhat`, and on nothing about the cut, which enters
        # only by choosing WHICH rows join the sums.  So the cut estimate is a subset of
        # the uncut one, and scoring the kept rows a second time was pure waste (43% of
        # the run at this cut).  The one thing that could couple a row to its position in
        # the frame is the armed mu-correction, which `log_likelihood` indexes by
        # `row_offset`; assert it is absent rather than assume it, and fall back to two
        # independent passes if it is ever armed.
        if args.uncut_control:
            assert getattr(est, "_mu_corr", None) is None, \
                "mu-correction is armed: rows are position-dependent, score legs separately"
            su, iu = score_pass(est, nodes, frl, ehl, args.chunk, f"leg {li+1}/{n_legs}",
                                grad_chunk=args.grad_chunk, grad_delta=args.grad_delta,
                                slab_mult=args.slab_mult, ll_dtype=LL_DTYPE[args.ll_dtype])
            sk, ik = su[kl], iu[kl]
            for dst, v in zip(acc["uncut"], (su, iu, blk)):
                dst.append(v)
        else:
            sk, ik = score_pass(
                est, nodes, frl.iloc[np.flatnonzero(kl)].reset_index(drop=True),
                ehl[kl], args.chunk, f"kept L{li+1}/{n_legs}",
                grad_chunk=args.grad_chunk, grad_delta=args.grad_delta,
                slab_mult=args.slab_mult, ll_dtype=LL_DTYPE[args.ll_dtype])
        for dst, v in zip(acc["kept"], (sk, ik, blk[kl])):
            dst.append(v)
        del frl, ehl
    del est

    def reduce_(which):
        s, info, block = (np.concatenate(v) for v in acc[which])
        return blocked_sums(s, info, block, args.jk_blocks)

    return (reduce_("kept"), reduce_("uncut") if args.uncut_control else None,
            n_keep, n_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default=MODEL)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--load-oversample", type=float, default=1.6,
                    help="raw rows read per selected row wanted.  A flow trained on a "
                         "narrowed primary domain (V3: mag<25.8, Re>0.5) keeps ~18%% of "
                         "raw rows, so it needs ~7 here; load_g0 prints the value to use "
                         "when it comes up short")
    ap.add_argument("--primary-domain", choices=["checkpoint", "default"],
                    default="checkpoint",
                    help="checkpoint: restrict the rows to the primary true-property "
                         "domain THIS flow was trained on, read from its metadata.  "
                         "default: the wide DEFAULT_SELECTION_CUTS domain, which for a V3 "
                         "flow means scoring rows it never saw")
    ap.add_argument("--closure-g", type=float, default=0.02)
    ap.add_argument("--closure-g2", type=float, default=0.0,
                    help="second component of the injected shear (default 0, i.e. shear "
                         "along +g1).  +g1 is a SYMMETRY AXIS of the square node lattice, "
                         "so the quadrature's leading m=4 angular error is invisible to a "
                         "run along it -- it enters as cos(4*phi), which is stationary "
                         "there.  Run g1=g2=g/sqrt(2) (45 degrees, the lattice diagonal, "
                         "where cos(4*phi)=-1) to expose it; the two should agree, and the "
                         "gap is the lattice error on m.  Everything is reported projected "
                         "onto the injected direction, so the two runs are comparable")
    ap.add_argument("--cut-abs-ehat", type=_opt_float, default=0.6,
                    help="keep |xhat| < this, or 'none' for no shape cut.  Isotropic on "
                         "purpose (see module docstring)")
    ap.add_argument("--cut-bound", action="append", default=[], metavar="NAME:LO:HI",
                    help="keep LO <= NAME < HI, with NAME one of the flow's OUTPUT columns "
                         "and an empty LO or HI meaning unbounded.  Repeatable, ANDed with "
                         "--cut-abs-ehat.  e.g. measured_mag_auto::24.5")
    ap.add_argument("--pi-rows", default="4096",
                    help="subsample for Pi_k.  A COMMA LIST sweeps several sizes off the "
                         "one (expensive) per-object pass, which is how the convergence "
                         "of the population block gets measured for free")
    ap.add_argument("--pi-samples", type=int, default=8)
    ap.add_argument("--uncut-control", action="store_true",
                    help="also run the estimator with NO cut, where the population terms "
                         "vanish identically.  Not optional for interpretation: the cut "
                         "result is only meaningful relative to what the SAME estimator "
                         "returns on the same rows without one.")
    ap.add_argument("--pi-reps", type=int, default=4,
                    help="independent Pi realisations; their scatter is the error bar")
    ap.add_argument("--ring", choices=["none", "rot90"], default="rot90",
                    help="rot90: pair every drawn true shape with its 90-degree rotation "
                         "(e -> -e) on the SAME catalogue row.  Cancels intrinsic shape "
                         "noise, which is the entire error budget of the unpaired run")
    ap.add_argument("--share-latents", action="store_true",
                    help="reuse the flow's latents between ring partners, so the "
                         "MEASUREMENT noise cancels too wherever the flow is equivariant")
    ap.add_argument("--shape-reps", type=int, default=1,
                    help="independent true-shape realisations over the SAME catalogue "
                         "rows.  The catalogue caps at ~929k selected rows; re-drawing "
                         "shapes gives fresh xhat from the exact same model and shrinks "
                         "the dominant shape-noise term past that cap")
    ap.add_argument("--jk-blocks", type=int, default=200,
                    help="delete-one-block jackknife blocks; pairs are kept together")
    ap.add_argument("--save-scores", default=None,
                    help="cache the catalogue half of (5.3) -- the per-block partial sums "
                         "-- to this .npz.  The score pass is hours and depends on nothing "
                         "about the population block")
    ap.add_argument("--load-scores", default=None,
                    help="reuse a --save-scores cache instead of re-scoring.  Refuses to "
                         "load one built with different settings; a mismatch would pair "
                         "one run's galaxies with another run's Pi")
    ap.add_argument("--row-shard", type=int, default=0,
                    help="which disjoint slice of the catalogue to score (0-based)")
    ap.add_argument("--row-shards", type=int, default=1,
                    help="how many slices to cut the catalogue into.  Shards scored in "
                         "parallel jobs merge by passing every cache to --load-scores as a "
                         "comma list; the per-block sums are additive, so the merged "
                         "jackknife is exact rather than approximate")
    ap.add_argument("--pi-grid-n", type=int, default=141,
                    help="node count for the POPULATION block only (<s>_sel, I_sel, Pi); 0 "
                         "means reuse --grid-n.  The Pi-weighted integral converges much "
                         "more slowly than the per-galaxy one -- see check_quadrature.py -- "
                         "and refining it does NOT invalidate a score cache.  The default is "
                         "141 because reusing --grid-n=61 is a KNOWN 0.7%% bias on m for the "
                         "|xhat|<0.6 cut (cont.177); 141 was converged for THAT cut, so "
                         "re-run the ladder when the cut changes")
    ap.add_argument("--pi-azimuthal-average", action="store_true",
                    help="DIAGNOSTIC: replace Pi by its mean over each ring of constant |e| "
                         "before forming <s>_sel.  Decides whether the measured flow "
                         "anisotropy causes the <s>_sel anomaly.  Not a correction -- see "
                         "azimuthally_average()")
    ap.add_argument("--pi-azimuthal-bins", type=int, default=80)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--grad-chunk", type=int, default=256)
    ap.add_argument("--grad-delta", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--slab-mult", type=int, default=64)
    ap.add_argument("--ll-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="fp32", choices=list(MODES),
                    help="arithmetic for the flow's forward passes.  The score pass is "
                         "~60 PFLOP of 256x256 matmul per leg and fp32 cannot touch the "
                         "tensor cores; see sbsi/precision.py and validate any change "
                         "with scripts/bench_score_speed.py before trusting a result")
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    # THE INJECTED SHEAR IS A VECTOR, AND EVERY READ-OUT IS ITS PROJECTION.  With the
    # default `--closure-g2 0` this is the old scalar run exactly (`gdir = (1,0)`, so the
    # projection is `ghat_1`); with both components set it is the same estimator read
    # along the direction actually injected, which is what makes a 45-degree run
    # comparable to an on-axis one.
    gvec = np.array([float(args.closure_g), float(args.closure_g2)])
    gmag = float(np.hypot(*gvec))
    gn = gmag or 1.0                    # g=0 is the null test: quote c = ghat, not m
    is_null = gmag == 0.0
    gdir = np.array([1.0, 0.0]) if is_null else gvec / gmag
    lab = "c = ghat (truth 0)" if is_null else "m = ghat/g - 1"
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={args.device}  torch={torch.__version__}", flush=True)

    # The prior's domain must be decided BEFORE the prior is built, which is why the model
    # is loaded first: the checkpoint is what says which primary domain this run is about.
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    tnames = list(bundle.target_transform.target_names)
    print(f"\nmodel: {os.path.basename(args.measurement_model)}\n"
          f"  outputs   {tnames}\n"
          f"  conditions {list(bundle.condition_preprocessor.feature_names)}", flush=True)

    # THE CUT, BUILT AGAINST THIS FLOW'S OUTPUT NAMES.  `OutputCut` rejects a --cut-bound
    # naming anything the flow does not predict, which is §4.7 enforced rather than assumed.
    try:
        cut = OutputCut.from_specs(tnames, abs_shape=args.cut_abs_ehat,
                                   specs=args.cut_bound)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"--cut-bound/--cut-abs-ehat: {e}")
    print(f"  cut       keep {cut.describe()}", flush=True)

    cuts, (dmag, dre), domain_tag = primary_domain_cuts(bundle.metadata)
    if args.primary_domain == "default":
        cuts, domain_tag = None, ""
        print("  domain    DEFAULT (wide).  This flow's own training domain is NOT applied, "
              "so some rows are outside it", flush=True)
    elif cuts is None:
        print("  domain    DEFAULT (wide) -- this checkpoint records no narrowed primary "
              "domain, so there is nothing to match", flush=True)
    else:
        print(f"  domain    from checkpoint metadata: primary true mag < {dmag}, "
              f"Re > {dre}  (prior cache tag {domain_tag})", flush=True)
    prior = build_prior(args, cuts=cuts, tag=domain_tag,
                        oversample=args.load_oversample)
    grid, cell = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"grid: G={len(grid)} nodes, cell {cell:.3e}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}", flush=True)

    # A SEPARATE, FINER NODE BANK FOR THE POPULATION BLOCK.  The per-galaxy scores and the
    # population terms are DIFFERENT integrals that merely happen to have been evaluated on
    # the same grid, and they do not converge at the same rate.  Measured flow-free by
    # `scripts/check_quadrature.py`: at the production `grid_n=61` the prior-only Bartlett
    # identities are already good to 1.2e-3 (0.04% on `m`), but the `Pi`-WEIGHTED integral
    # that gives `I_sel` is off by 2.4%, which is 0.90% on `m` -- larger than the whole
    # measured residual and 3x the 0.3% deliverable.  `Pi` adds angular structure (the m=2
    # and m=4 terms) on top of the smooth prior, and the grid has to resolve that too.
    #
    # Refining ONLY the population block is both correct and nearly free.  Correct, because
    # the per-galaxy quadrature error largely cancels in the cut-minus-uncut difference
    # (both sides use it) while `I_sel` enters the cut estimate ALONE and so lands undiluted
    # -- which is exactly the uncancelled part.  Nearly free, because since the `Pi` fast
    # path the population block is minutes: a full ladder at G=2765 is ~3 min, so G=7693 is
    # ~8, against ~16 h to redo the score pass at the finer grid.
    #
    # It also leaves the score caches VALID.  `pi_grid_n` is deliberately absent from
    # `cache_key` below: the cached per-block sums depend on the score-pass grid and not on
    # this one, so refining here must not invalidate 5.7 h of scoring.
    if args.pi_grid_n and args.pi_grid_n != args.grid_n:
        pi_grid, pi_cell = make_e_grid(n=args.pi_grid_n, emax=args.grid_emax,
                                       rmax=args.grid_rmax)
        pi_nodes = ShapeScoreNodes(pi_grid, prior, delta=args.fd_delta,
                                   info_delta=args.info_delta)
        print(f"population block on a FINER bank: G={len(pi_grid)} nodes, "
              f"cell {pi_cell:.3e} (score pass stays at G={len(grid)})", flush=True)
    else:
        pi_grid, pi_nodes = grid, nodes

    df = load_g0(args.g0_catalogue, args.max_rows,
                 shard=args.row_shard, n_shards=args.row_shards, cuts=cuts,
                 oversample=args.load_oversample)
    print(f"rows: {len(df):,} from {os.path.basename(args.g0_catalogue)}", flush=True)

    # ---- data: draw true shapes from the prior, shear, push through the flow ----------
    # RING PAIRS.  The estimator's error bar is set almost entirely by the intrinsic
    # shape noise of the drawn population -- +/-4.3% at 400k rows, a Cramer-Rao bound
    # that brute force only beats as 1/sqrt(N).  But this is a CLOSURE test: the true
    # shapes are ours to choose, so we can choose them in 90-degree-rotated pairs on the
    # same catalogue row.  Rotating a galaxy by 90 degrees sends `e -> -e` in the spin-2
    # basis, so the pair's intrinsic shapes cancel exactly, while the sheared shapes
    # `S_g(e)` and `S_g(-e)` do NOT cancel -- their sum is second order in `e` and first
    # order in `g`, i.e. it is pure signal.  Each member is still marginally a correct
    # draw (the prior is isotropic), so nothing about the estimator's expectation moves;
    # only the variance does.  This is the standard ring test, used here as the
    # variance-reduction device it is.
    rng = np.random.default_rng(args.seed)
    g = float(args.closure_g)

    def shear_and_sample(e1, e2, seed):
        f = df.iloc[:len(e1)].copy()
        f["e1_input_rot0_p"], f["e2_input_rot0_p"] = apply_shear_to_ellipticity(
            e1, e2, gvec[0] * np.ones(len(e1)), gvec[1] * np.ones(len(e1)))
        f = rescale(f, **rk)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return f, bundle.sample(f, n_samples=1, batch_size=args.batch_size)[:, 0, :]

    # SHAPE REPEATS.  The catalogue yields only ~929k selected rows, which caps the ring at
    # ~1.9M objects -- not enough for the sub-percent error bar this needs.  But the
    # catalogue row supplies only the OTHER conditioning properties (magnitude, size,
    # neighbours); the true shape is drawn by us.  Re-drawing shapes on the same rows
    # therefore yields genuinely fresh `xhat` from the exact same model, and shrinks the
    # dominant (shape-noise) term.  What it does NOT shrink is the population sampling of
    # those other properties, which stays fixed at 929k rows -- so this buys precision on
    # the estimator's bias FOR THIS POPULATION, not a wider population average.  All
    # repeats of a row share a jackknife block, so the reuse is handled honestly.

    def legs():
        """`(frame, ehat, pair_id)` per shape realisation; ring members are separate legs."""
        for r in range(args.shape_reps):
            e1i, e2i = prior.sample(len(df), rng)
            base = args.flow_seed + 10 * r
            yield shear_and_sample(e1i, e2i, base) + (np.arange(len(df)),)
            if args.ring == "rot90":
                # Shared latents need the same batch layout, which two frames of equal
                # length and the same seed already have (`model.sample` draws one `randn`
                # block per batch, in row order).  Measured, sharing them makes the error
                # bar WORSE (x1.2 vs x1.6 reduction), so it is off by default.
                yield shear_and_sample(-e1i, -e2i,
                                       base if args.share_latents else base + 1) \
                    + (np.arange(len(df)),)

    n_legs = args.shape_reps * (2 if args.ring == "rot90" else 1)
    print(f"\n{n_legs} leg(s): {args.shape_reps} shape realisation(s) x "
          f"{'ring pair (rot90)' if args.ring == 'rot90' else 'single'} on {len(df):,} rows "
          f"-> {n_legs * len(df):,} objects, {args.jk_blocks} jackknife blocks", flush=True)

    # CACHE THE CATALOGUE HALF.  `(s_i, I_i)` know nothing about the population block, and
    # everything (5.3) needs from them is a per-block partial sum.  Scoring is hours;
    # `<s>_sel` and `I_sel` are minutes and are the term still converging.  Caching lets
    # the population block be re-estimated against a FIXED catalogue, which also makes
    # those re-runs paired -- a change in `Pi` is then not confounded with a change in the
    # galaxies.  The key covers everything that moves the sums; loading across a mismatch
    # would silently pair one run's galaxies with another run's `Pi`.
    # `cut` is the bare `|xhat|` radius when that is the whole selection, so the caches
    # written before --cut-bound existed still load; `cut_bounds` appears only when there
    # ARE bounds, because merge_block_sum_caches treats a key missing from a cache as a
    # mismatch and adding one unconditionally would strand every earlier shard.
    cache_key = dict(cut=cut.abs_shape, closure_g=g, rows=len(df), ring=args.ring,
                     shape_reps=args.shape_reps, share_latents=int(args.share_latents),
                     jk_blocks=args.jk_blocks, grid_n=args.grid_n,
                     grid_emax=args.grid_emax, grid_rmax=args.grid_rmax,
                     fd_delta=args.fd_delta, info_delta=args.info_delta,
                     grad_delta=args.grad_delta, uncut=int(args.uncut_control),
                     flow_seed=args.flow_seed, seed=args.seed,
                     model=os.path.basename(args.measurement_model),
                     catalogue=os.path.basename(args.g0_catalogue),
                     precision=args.precision, ll_dtype=args.ll_dtype,
                     row_shard=args.row_shard, row_shards=args.row_shards)
    if cut.bounds:
        cache_key["cut_bounds"] = cut.key()
    if gvec[1]:
        # only when non-zero, for the same reason as `cut_bounds`: a key absent from a
        # banked cache reads as a mismatch, so adding it unconditionally would strand
        # every shard scored before --closure-g2 existed (all of which had g2 = 0)
        cache_key["closure_g2"] = float(gvec[1])
    if cuts is not None and cuts != [list(c) for c in DEFAULT_SELECTION_CUTS]:
        cache_key["primary_domain"] = json.dumps(cuts)

    if args.load_scores:
        # MERGING SHARDS.  A comma list concatenates several caches' per-block sums; the
        # exactness argument and the two guards live in `merge_block_sum_caches`, which is
        # in the library precisely so `tests/test_score_cache_merge.py` can reach it.  This
        # bookkeeping has produced two silent failures (a key regression that made jobs
        # "COMPLETE" in 53 s with no results, and the double-count guard) while every
        # identity around it was tested, so it is the last place to leave untested.
        paths = [p for p in str(args.load_scores).split(",") if p]
        try:
            blk_k, blk_u, n_keep, n_tot, seen = merge_block_sum_caches(
                paths, cache_key, want_uncut=args.uncut_control)
        except ScoreCacheMismatch as e:                 # a clean job log beats a traceback
            raise SystemExit(f"--load-scores {e}")
        print(f"cached scores loaded from {len(paths)} shard(s) "
              f"(row shards {seen}), {len(blk_k[0])} jackknife blocks, "
              f"{n_tot:,} objects; no score pass this run", flush=True)
    else:
        with precision_region(args.precision, args.device):
            blk_k, blk_u, n_keep, n_tot = score_catalogue(args, nodes, bundle, grid,
                                                          legs, cut, n_legs)
        if args.save_scores:
            d = os.path.dirname(os.path.abspath(args.save_scores))
            os.makedirs(d, exist_ok=True)
            np.savez(args.save_scores, key=json.dumps(cache_key, sort_keys=True),
                     n_keep=n_keep, n_tot=n_tot,
                     cnt_k=blk_k[0], ns_k=blk_k[1], ni_k=blk_k[2],
                     **({} if blk_u is None else
                        dict(cnt_u=blk_u[0], ns_u=blk_u[1], ni_u=blk_u[2])))
            print(f"score sums cached -> {args.save_scores}", flush=True)

    # `Pi`'s cheap internal consistency check compares the prior-weighted <Pi> against this.
    # It used to be computed from the kept mask directly; caching the score pass replaced the
    # rows with block sums and dropped it, so every run that got as far as the `Pi` report
    # died on a NameError there -- which is to say every run since that refactor.
    keep_frac = n_keep / max(n_tot, 1)
    print(f"cut [{cut.describe()}]: keeps {n_keep:,}/{n_tot:,} = {keep_frac:.2%}",
          flush=True)

    # ---- uncut control: the same estimator, same rows, no selection at all --------
    reps_u = None
    if args.uncut_control:
        gh_u, sig_u, reps_u = jackknife_blocks(*blk_u)
        gp_u = float(gh_u @ gdir)
        sp_u = float(jackknife_sigma((reps_u @ gdir)[:, None])[0])
        # Cramer-Rao ALONG THE INJECTED DIRECTION: sqrt(d^T (sum I)^-1 d).  For the
        # on-axis default this is the old 1/sqrt(sum I_11) up to the off-diagonal, which
        # is 0.6% of the diagonal here -- i.e. the printed bar moves in the 5th digit,
        # and only because the general formula is the right one to compare a 45-degree
        # run against.
        fisher_u = float(np.sqrt(gdir @ np.linalg.solve(blk_u[2].sum(axis=0), gdir)))
        print(f"\nUNCUT CONTROL (Pi == 1, so both population terms vanish by construction):")
        print(f"  ghat = [{gh_u[0]:+.6f} +/- {sig_u[0]:.6f}, "
              f"{gh_u[1]:+.6f} +/- {sig_u[1]:.6f}]   "
              f"{lab} = {gp_u/gn - (0 if is_null else 1):+.3%} +/- {sp_u/abs(gn):.3%}")
        print(f"  error bar is a {args.jk_blocks}-block JACKKNIFE.  The Cramer-Rao/Fisher "
              f"bar would say {fisher_u/abs(gn):.3%};")
        print(f"  ring pairing beats it by x{fisher_u/max(sp_u, 1e-30):.1f} "
              f"(x1 means no variance reduction, which is correct for --ring none).")
        print(f"  Any bias here is the BASELINE estimator's, not the cut's.  The cut rows")
        print(f"  below must be read against this number, not against zero.")

    # ---- population block, swept over the Pi sample size ------------------------------
    ladder = sorted(int(v) for v in str(args.pi_rows).split(","))
    seeds = [args.seed + 1000 * j for j in range(args.pi_reps)]
    print(f"\nPi_k from {max(ladder)} RANDOM rows x {len(grid):,} nodes x "
          f"{args.pi_samples} draws x {args.pi_reps} reps (fresh rows each) = "
          f"{max(ladder)*len(grid)*args.pi_samples*args.pi_reps:,} draws"
          f"{'  (ladder ' + ','.join(map(str, ladder)) + ')' if len(ladder) > 1 else ''}",
          flush=True)
    with precision_region(args.precision, args.device):
        pi_ladder = pass_fraction_by_node(
            bundle, df, pi_grid, rk,
            cut=cut,
            n_samples=args.pi_samples, batch_size=args.batch_size, seeds=seeds,
            ladder=ladder, rng=np.random.default_rng(args.seed + 77))

    if args.pi_azimuthal_average:
        pi_ladder = np.stack([azimuthally_average(pi_grid, pi_ladder[t],
                                                  args.pi_azimuthal_bins)
                              for t in range(pi_ladder.shape[0])])
        print(f"\n*** DIAGNOSTIC: Pi azimuthally averaged in {args.pi_azimuthal_bins} radial "
              f"bins ***\n    Pi now depends on |e| alone, so any angular structure the flow "
              f"has is gone.\n    <s>_sel collapsing toward zero => the flow anisotropy "
              f"causes it; surviving => it\n    does not, and the fault is in the score / "
              f"quadrature path.  This is NOT a\n    correction: d(m) from an averaged Pi is "
              f"not a calibration result.  Compare only\n    against the raw run off the SAME "
              f"--load-scores cache.", flush=True)

    def terms(p):
        return population_terms(pi_nodes, np.log(np.maximum(p, 1e-12)))

    n = int(blk_k[0].sum())
    mean_i = blk_k[2].sum(axis=0) / n
    w = pi_nodes.prior_weights()
    print(f"\nper-object: N={n:,}  <I> = [[{mean_i[0,0]:+.5f}, {mean_i[0,1]:+.5f}], "
          f"[{mean_i[1,0]:+.5f}, {mean_i[1,1]:+.5f}]]")
    print(f"  {len(seeds)} independent Pi realisations; scatter across them is the error")

    summary = []
    for t, m_pi in enumerate(ladder):
        pi_reps = pi_ladder[t]                                # (G, n_rep)
        per_rep = [terms(pi_reps[:, j]) for j in range(pi_reps.shape[1])]
        s_all = np.stack([q[0] for q in per_rep])             # (n_rep, 2)
        i_all = np.stack([q[1] for q in per_rep])             # (n_rep, 2, 2)
        pi = pi_reps.mean(axis=1)
        s_sel, i_sel = terms(pi)
        r = max(len(seeds), 2)
        es = s_all.std(axis=0, ddof=1) / np.sqrt(r)           # error on the mean
        ei = i_all.std(axis=0, ddof=1) / np.sqrt(r)

        print(f"\n=== Pi population sample M = {m_pi:,} "
              f"{'='*max(0, 46 - len(str(m_pi)))}")
        print(f"Pi_k: min={pi.min():.4f} max={pi.max():.4f} "
              f"prior-weighted mean={float(w @ pi):.4f}  "
              f"(actual keep {keep_frac:.4f}, mismatch "
              f"{float(w @ pi)/keep_frac - 1:+.2%})")
        print(f"  ^ the cheap internal consistency check on Pi.  <Pi> is integrated over "
              f"the g=0 prior\n    and the keep fraction is measured at |g|={gmag:.3f}; that "
              f"difference is <s>_sel*g/<Pi> ~ "
              f"{abs(float(s_sel @ gdir))*gmag/max(float(w@pi),1e-9):.2%}, "
              f"far below the mismatches\n    that mattered, so a mismatch above ~0.1% is "
              f"Pi's own sampling error and nothing else.")
        print(f"<s>_sel = [{s_sel[0]:+.6f} +/- {es[0]:.6f}, "
              f"{s_sel[1]:+.6f} +/- {es[1]:.6f}]   |s|/sigma = "
              f"{abs(s_sel[0])/max(es[0],1e-12):.1f}, "
              f"{abs(s_sel[1])/max(es[1],1e-12):.1f}")
        print(f"I_sel   = [[{i_sel[0,0]:+.5f} +/- {ei[0,0]:.5f}, {i_sel[0,1]:+.5f}], "
              f"[{i_sel[1,0]:+.5f}, {i_sel[1,1]:+.5f} +/- {ei[1,1]:.5f}]]")
        off = 0.5 * (abs(i_sel[0, 1]) + abs(i_sel[1, 0]))
        dia = 0.5 * (i_sel[0, 0] + i_sel[1, 1])
        # both projected onto the injected direction, so this is the fraction of the
        # information the cut removes ALONG the parameter actually being estimated
        iota_frac = float(gdir @ i_sel @ gdir) / float(gdir @ mean_i @ gdir)
        print(f"  I_sel/<I> = {iota_frac:+.4f} (A.7's 2/pi analogue);  "
              f"off-diag/diag = {off/max(abs(dia),1e-12):.3f} (0 if the cut is isotropic)")

        # Pi's OWN UNCERTAINTY MUST REACH THE ANSWER.  The jackknife bars galaxies; it says
        # nothing about how well `Pi` -- hence `<s>_sel` and `I_sel` -- is known.  Those are
        # measured from a finite population sample, they enter only the CUT estimate, and so
        # they land undiluted on the cut-minus-uncut difference.  Left out, a 0.63% error on
        # `I_sel` (which is +/-0.24% on m at this cut) silently vanished and turned a 2.2
        # sigma residual into an apparent 2.9 sigma one.  Re-solve the estimator once per Pi
        # replicate and add the scatter of the mean in quadrature.  Cheap: the catalogue sums
        # are already formed, so each replicate is one 2x2 solve.
        sum_s, sum_i, n_keep_rows = blk_k[1].sum(axis=0), blk_k[2].sum(axis=0), n

        def ghat_with(ss_, ii_):
            zs = np.zeros(2) if ss_ is None else np.asarray(ss_, float)
            zi = np.zeros((2, 2)) if ii_ is None else np.asarray(ii_, float)
            return np.linalg.solve(sum_i - n_keep_rows * zi, sum_s - n_keep_rows * zs)

        rows = (("none      (sum s / sum I)", None, None),
                ("numerator only", s_sel, None),
                ("FULL (5.3)", s_sel, i_sel))
        print(f"  {'correction':<24} {'ghat_1':>10} {'ghat_2':>10} {'m = ghat/g - 1':>14}"
              + (f" {'d(m) vs uncut':>16} {'sig_gal':>8} {'sig_Pi':>8} {'sigma':>8} {'nsig':>6}"
                 if reps_u is not None else ""))
        for name, ss, ii in rows:
            gh, sig, reps = jackknife_blocks(*blk_k, ss, ii)
            if ss is None and ii is None:
                sig_pi = 0.0                       # no population term, nothing to propagate
            else:
                gj = [float(ghat_with(q[0] if ss is not None else None,
                                      q[1] if ii is not None else None) @ gdir)
                      for q in per_rep]
                sig_pi = float(np.std(gj, ddof=1) / np.sqrt(len(gj)))
            gp = float(gh @ gdir)
            sp = float(jackknife_sigma((reps @ gdir)[:, None])[0])
            line = (f"  {name:<24} {gh[0]:>10.6f} {gh[1]:>10.6f} "
                    f"{gp/gn - (0 if is_null else 1):>18.3%} +/- {sp/abs(gn):.3%}")
            if reps_u is not None:
                # THE PAIRED COMPARISON IS THE ACTUAL QUESTION: does the correction put
                # the cut sample back where the uncut one is?  Both run on the same rows
                # and share most of their shape noise, so the DIFFERENCE is much better
                # determined than either -- but only when jackknifed AS a difference,
                # block by block.  Adding the two bars in quadrature throws that away.
                dm = (gp - gp_u) / gn
                d = (reps @ gdir - reps_u @ gdir) / gn
                sd_gal = float(jackknife_sigma(d[:, None])[0])
                sd_pi = sig_pi / abs(gn)
                sd = float(np.hypot(sd_gal, sd_pi))
                line += (f" {dm:>15.3%} {sd_gal:>7.3%} {sd_pi:>7.3%} {sd:>7.3%} "
                         f"{abs(dm)/max(sd,1e-12):>6.1f}")
                if name.startswith("FULL"):
                    summary.append((m_pi, float(w @ pi), iota_frac,
                                    gp / gn - (0 if is_null else 1), dm, sd))
            print(line)

    print(f"\n  truth g = [{gvec[0]:+.6f}, {gvec[1]:+.6f}], |g| = {gmag:.6f}, read out "
          f"projected on that direction   (sigma = galaxy {args.jk_blocks}-block jackknife\n  and Pi replicate scatter, in quadrature -- Pi enters only the CUT estimate, so\n  its error lands undiluted on the difference)")
    print("  §5B.2 predicts the numerator correction does nothing for an isotropic cut")
    print("  and that I_sel carries the whole effect.  Compare rows 1-2 (should agree)")
    print("  against row 3 (should move, and toward the truth).")
    print("  A selection correction that works drives d(m) vs uncut to zero.")

    if len(summary) > 1:
        print(f"\n  Pi CONVERGENCE of the FULL (5.3) row -- read this before the numbers "
              f"above:\n  {'M':>8} {'<Pi>':>8} {'I_sel/<I>':>10} {'m':>9} "
              f"{'d(m) vs uncut':>15} {'sigma':>9}")
        for m_pi, pib, ratio, mm, dm, sd in summary:
            print(f"  {m_pi:>8,} {pib:>8.4f} {ratio:>10.4f} {mm:>8.3%} "
                  f"{dm:>14.3%} {sd:>8.3%}")
        d = [abs(summary[i + 1][4] - summary[i][4]) for i in range(len(summary) - 1)]
        print(f"  rung-to-rung moves in d(m): "
              f"{', '.join('%.3f%%' % (100*x) for x in d)}")
        print("  Flat within the error bars => converged, and the last rung is the answer.")
        print("  Still moving monotonically => NOT converged; the last rung is then an")
        print("  upper bound on the accuracy, not the accuracy.  (Rungs are nested within")
        print("  a replicate, so their central values are correlated; the error bars are")
        print("  not, being built from independent row draws.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
