"""constgold selection table at NEAR-DOMAIN cuts, on REAL measured mag/size -- no proxy anywhere.

WHAT CHANGED AND WHY IT MATTERS. constgold previously stored only ONE per-leg measured quantity
(S/N), so the model had to cut on a mag+size PROXY for S/N while the sim cut on the real thing, and
thresholds had to be matched by KEEP-FRACTION rather than by value. That proxy carried its own error
(-0.16 to -0.26 pts at mild cuts). The catalogue now carries `measured_mag_auto_{plus,minus}` and
`measured_flux_radius_{plus,minus}` (job 15366166, verified to reproduce all 40 pre-existing columns
exactly), so for magnitude and size cuts BOTH SIDES CUT ON THE SAME REAL QUANTITY AT THE SAME
ABSOLUTE THRESHOLD. No proxy, no quantile matching.

Rows marked (*) are the exception: real-S/N cuts, which the flow still cannot represent because it
does not output FLUX_AUTO/FLUXERR_AUTO. Those keep the proxy on the model side and are flagged.

COLUMNS (owner's spec: the old (2) sheared-intrinsic column dropped, residual bias added)
  (1) pure sel  R_unsheared(cut) / R_sheared(no cut). The unsheared column projects the SAME raw
                shape in both legs, so its shape response is identically 0 and its no-cut R is 0 --
                whatever survives is the pure MOVING-BOUNDARY term. Normalised by the sheared no-cut
                R because its own would be 0/0.
  (1b) /R_meas  the SAME numerator over R_meas(no cut) instead -- a DIAGNOSTIC, not a replacement.
                (1)'s denominator is the sheared-intrinsic R (~1.00) while (3)/(4) use the measured R
                (~0.86), which invites renormalising (1) onto (3)'s scale. That is only half the
                conversion: (1) averages INTRINSIC shapes and (3) averages MEASURED ones, and
                measured shapes are diluted relative to intrinsic, so the scale change needs that
                dilution factor too. The two corrections oppose and largely cancel. The factor is
                unmeasured here, so neither normalisation is asserted correct; (1) stays as defined
                and (1b) is printed beside it so the size of the ambiguity is visible.
  (3) measured  R_meas(cut)/R_meas(no cut) - 1, the shift the SIM has in measured shapes.
  (4) MODEL m   R_model(cut)/R_model(no cut) - 1, the flow's counterpart of (3).
  m             R_meas(cut)/R_model(cut) - 1, the residual bias AT THE CUT.
  dm            m(cut) - m(no cut), the SELECTION-INDUCED excess.

WHAT COLUMN (3) ACTUALLY CONTAINS -- TWO TERMS, NOT ONE. It compares the mean response of the
surviving sample with the mean over everything, so it mixes:
  (a) POPULATION RE-WEIGHTING -- even a fixed, shear-INDEPENDENT cut changes which galaxies are
      averaged, and R varies strongly with brightness and size. Not a bias: the selected sample
      really does have a different mean response, and the model has to reproduce it.
  (b) MOVING-BOUNDARY SELECTION -- the cut is on MEASURED quantities, which depend on shear, so the
      boundary moves and objects cross it in a shear-correlated way. This is selection bias proper.
Column (1) isolates (b): it projects the SAME intrinsic shape in both legs, so a cut that kept
identical objects in both legs gives exactly 0 and any shear-independent population change cancels
by construction. (3) minus (1) is therefore roughly the (a) part. The split matters for reading the
table: `mag<25` shifts +16.75% of which the boundary term is -0.08% (essentially all re-weighting --
bright galaxies have high R), while `R>0.70"` shifts +11.93% with a +5.44% boundary term. Both terms
must be right for the model to pass, since (4) is built the same way on the model side; (a) is the
easy half (correct R vs galaxy properties, which is figure 2) and (b) is the new thing.

SIGN CONVENTION. `m` is sim/model - 1 everywhere -- the project convention (AGENTS.md, fig3,
`m = R_sim/(R_flow+R_blend) - 1`). An earlier version of this table inverted it for the cut column
only (model/sim - 1), which put the no-cut reference line and the cut rows on opposite sides of zero:
the `R>0.30"` row keeps 99.99975% of the population and so IS the no-cut case, yet printed +0.247%
against a no-cut line at -0.245%. Same number, mirrored. Do not reintroduce the inversion.

ERRORS ARE FORMED PER SEED, NOT FROM ENSEMBLE MEANS. R_model(cut) and R_model(no cut) move together
from seed to seed, so their RATIO is far stabler than either term. Propagating the numerator's
scatter alone (sem(R_model(cut))/R_model(no cut)) ignores that cancellation and returns the same
+-0.43% on every row no matter how aggressive the cut -- which is what the earlier version did. Each
ratio is now built inside each seed and the spread taken across seeds, which is also the cancellation
AGENTS.md invokes to justify 4 seeds for selection work.

Note which quantity that argument actually covers: (4) and `dm` are model-vs-model or
difference-of-m, so the common-mode seed offset cancels and 4 seeds is genuinely enough. `m` at a cut
is model-vs-SIM and the sim side has no seed dependence, so it inherits the full no-cut seed error.
`dm` is therefore the column to read at 4 seeds; absolute `m` is a 16-seed (shape) quantity.

THE FIDUCIAL MODEL IS FLOW + EMULATOR BLEND. Earlier runs of this script used the flow alone, whose R
is self-response only, while sim column (3) contains the neighbour response too -- so m_flow carried a
constant -15.32% offset that was the MISSING BLEND TERM, not a model error. The model column now adds
the per-object R_blend from the tuned in-domain emulator (`--blend-lookup`), weighted by the SAME
per-draw pass mask as the shapes, so the blend term is averaged over exactly the objects the model
selects. m_flow is now a real residual bias and should be read at face value.

COVERAGE IS ENFORCED, NOT ASSUMED. An emulator applies its stored training cuts at inference; rows
outside them return nothing and would silently fall back to R_blend=0. On the WIDE population that
collapsed <R_blend> from 0.159 to 0.059 and produced a spurious +28.9% m (job 15366950). This script
refuses to run below `--min-blend-match` and restricts BOTH sim and model to rows carrying an
R_blend, so the two sides always share one population.

FIREWALL: constgold is EVALUATION only. Nothing is trained, fitted or selected here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS, rescale, source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbs_shear.paths import CROWD_LOOKUP as CROWD

NEED = ["case", "input_index", "neighbored", "distance", "polarization_angle",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "S/N_plus", "S/N_minus",
        "measured_mag_auto_plus", "measured_mag_auto_minus",
        "measured_flux_radius_plus", "measured_flux_radius_minus"]

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
PX = 0.2


def leg_avg(ap, am, g):
    return (ap - am) / (2.0 * g)


@torch.no_grad()
def model_selected(bundle, df, g, gh1, gh2, intr, cuts, groups, n_samples, batch_size,
                   seed, device, sign=1.0, chunk=200_000, rblend=None, fixed_masks=None,
                   sel_weight=None):
    """Score the flow once; accumulate selected means for every (group, cut) pair.

    Cuts are given as ABSOLUTE thresholds on the flow's own sampled measured magnitude and size --
    the same physical quantities and the same numbers the sim uses. Only the (*) S/N rows fall back
    to the mag+size proxy, which the flow can form from its outputs.

    LEG SPLIT (returned in the 4th element, `outx`). The fiducial outputs collapse the two legs
    before they leave this function: the blend term is returned as 0.5*(bp+bm) and the keep fraction
    as 0.5*(keep+ + keep-). That collapse is CORRECT for the response itself -- the blend term enters
    the measured shape as +R_blend*g in the plus leg and -R_blend*g in the minus leg, so the leg
    difference that defines a response turns the two selected means into their SUM, not their
    difference -- but it throws away the leg DIFFERENCE, which is the model's own moving boundary.
    `outx` keeps every per-leg quantity so the difference is recoverable:
      keep_plus/keep_minus  fraction of draws passing in each leg -> the COUNT boundary term
      b_plus/b_minus        mean emulator R_blend over the draws selected in each leg
      sel_plus/sel_minus    mean of `sel_weight` over the draws selected in each leg
    `sel_weight` must be a per-object, SHEAR-INDEPENDENT label; passing the unsheared intrinsic shape
    projection makes `leg_avg(sel_plus, sel_minus)` the exact model counterpart of the sim's column
    (1) `pure_sel`, because both then average the SAME shear-independent quantity over a
    leg-dependent selected set. If a cut selected identical draws in both legs the term is exactly 0,
    which is what the `[T]` true-property rows verify.
    Nothing here feeds R_model; the fiducial path is untouched.
    """
    names = bundle.target_transform.target_names
    i1, i2 = names.index("measured_ngmix_g1"), names.index("measured_ngmix_g2")
    imag, ilr = names.index("measured_mag_auto"), names.index("measured_log_flux_radius")
    ln10 = np.log(10.0)
    n = len(df)
    keys = ["__nocut__"] + [c["name"] for c in cuts]
    acc = {gname: {k: [[0.0, 0], [0.0, 0]] for k in keys} for gname in groups}
    # Parallel accumulator for the emulator's per-object R_blend, weighted by the SAME per-draw pass
    # mask as the shapes, so the blend term is averaged over exactly the objects the model selects.
    accb = {gname: {k: [0.0, 0.0] for k in keys} for gname in groups}
    # Same shape as accb, but weighted by the shear-INDEPENDENT per-object label `sel_weight`.
    # Its leg difference is the model's pure moving-boundary term (see the docstring).
    accs = {gname: {k: [0.0, 0.0] for k in keys} for gname in groups}

    for li, s in ((0, +g), (1, -g)):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            fr = df.iloc[lo:hi].reset_index(drop=True).copy()
            e1s, e2s = apply_shear_to_ellipticity(intr[0][lo:hi], intr[1][lo:hi],
                                                  s * gh1[lo:hi], s * gh2[lo:hi])
            fr["e1_input_rot0_p"], fr["e2_input_rot0_p"] = e1s, e2s
            fr = rescale(fr, **RK)
            torch.manual_seed(seed)                      # CRN: identical latents in both legs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)
            proj = sign * (d[:, :, i1] * gh1[lo:hi, None] + d[:, :, i2] * gh2[lo:hi, None])
            rbc = rblend[lo:hi][:, None] if rblend is not None else None
            swc = sel_weight[lo:hi][:, None] if sel_weight is not None else None
            mag = d[:, :, imag]
            # Compare size in LOG space. Exponentiating first overflowed to +inf on extreme flow
            # draws, and `inf > thr` is True, so those draws were KEPT by every size cut -- inflating
            # the model's selected set on the size rows only (the sim side was never affected).
            # `log_size > log(thr)` is exactly equivalent for thr > 0 and cannot overflow.
            logsz = d[:, :, ilr]                         # flow emits NATURAL log of radius in px
            fin = np.isfinite(proj)
            for gname, gmask in groups.items():
                gm = gmask[lo:hi][:, None]
                base = fin & gm
                acc[gname]["__nocut__"][li][0] += float(np.where(base, proj, 0.0).sum())
                acc[gname]["__nocut__"][li][1] += int(base.sum())
                if rbc is not None:
                    accb[gname]["__nocut__"][li] += float(np.where(base, rbc, 0.0).sum())
                if swc is not None:
                    accs[gname]["__nocut__"][li] += float(np.where(base, swc, 0.0).sum())
                for c in cuts:
                    pm = base
                    for sc in c["conds"]:
                        if sc["var"] == "fixed":
                            # TRUE-property cut: the model does NOT sample this quantity, it is a
                            # conditioning input, so both sides select the IDENTICAL objects and the
                            # selection boundary is exactly as sharp on the model side as on the sim
                            # side. That removes the measured-quantity-distribution channel entirely,
                            # leaving `dm` as a pure response error. Broadcast over draws.
                            pm = pm & fixed_masks[sc["key"]][lo:hi][:, None]
                            continue
                        if sc["var"] == "mag":
                            xv = mag
                        elif sc["var"] == "size":
                            xv = logsz                   # threshold already stored as log(px)
                        else:                            # proxy S/N, (*) rows only
                            xv = sc["a"] * mag + sc["b"] * (logsz / ln10)
                        pm = pm & ((xv > sc["thr"]) if sc["keep_high"] else (xv < sc["thr"]))
                    acc[gname][c["name"]][li][0] += float(np.where(pm, proj, 0.0).sum())
                    acc[gname][c["name"]][li][1] += int(pm.sum())
                    if rbc is not None:
                        accb[gname][c["name"]][li] += float(np.where(pm, rbc, 0.0).sum())
                    if swc is not None:
                        accs[gname][c["name"]][li] += float(np.where(pm, swc, 0.0).sum())

    out, outb, outk, outx = {}, {}, {}, {}
    for gname in groups:
        out[gname], outb[gname], outk[gname], outx[gname] = {}, {}, {}, {}
        for k in keys:
            a = acc[gname][k]
            mp = a[0][0] / a[0][1] if a[0][1] else np.nan
            mm = a[1][0] / a[1][1] if a[1][1] else np.nan
            out[gname][k] = leg_avg(mp, mm, g)          # R_flow (self-response) alone
            bp = bm = np.nan
            if rblend is not None:
                # R_blend has no leg dependence, but the SELECTION does, so average the two legs'
                # selected means -- the same convention leg_avg uses for the shapes.
                bp = accb[gname][k][0] / a[0][1] if a[0][1] else np.nan
                bm = accb[gname][k][1] / a[1][1] if a[1][1] else np.nan
                outb[gname][k] = 0.5 * (bp + bm)
            else:
                outb[gname][k] = 0.0
            # MODEL-side keep fraction: what fraction of flow DRAWS pass the same threshold. The
            # counts were already accumulated and were previously discarded. It is the diagnostic
            # that separates the two things `dm` mixes: if the model keeps a different fraction than
            # the sim, the two sides are not cutting the same population and part of `dm` is an error
            # in the flow's MEASURED-quantity distribution, not in its shape response.
            b0 = acc[gname]["__nocut__"]
            kp = a[0][1] / b0[0][1] if b0[0][1] else np.nan
            km = a[1][1] / b0[1][1] if b0[1][1] else np.nan
            outk[gname][k] = 0.5 * (kp + km)
            sp = sm = np.nan
            if sel_weight is not None:
                sp = accs[gname][k][0] / a[0][1] if a[0][1] else np.nan
                sm = accs[gname][k][1] / a[1][1] if a[1][1] else np.nan
            # LEG SPLIT -- diagnostic only, never folded into out/outb/outk.
            outx[gname][k] = dict(
                m_plus=float(mp), m_minus=float(mm),
                b_plus=float(bp), b_minus=float(bm), b_resp=float(leg_avg(bp, bm, g)),
                keep_plus=float(kp), keep_minus=float(km), keep_resp=float(leg_avg(kp, km, g)),
                sel_plus=float(sp), sel_minus=float(sm), sel_resp=float(leg_avg(sp, sm, g)),
                n_plus=int(a[0][1]), n_minus=int(a[1][1]))
    return out, outb, outk, outx


def fingerprint(df, g, rblend, groups):
    """Cheap identity of the prepared population, so array tasks cannot be merged across a drift.

    Every array task rebuilds the population independently. That is only safe if the build is
    deterministic -- it is (no RNG unless --max-rows subsamples, and that is seeded) -- but "is
    deterministic" is an assumption about code that changes. This turns it into a checked fact:
    a merge across tasks built from different catalogues, cuts or row orders will refuse rather
    than silently average incompatible accumulators.
    """
    ci = df["case"].to_numpy(np.int64)
    ii = df["input_index"].to_numpy(np.int64)
    return dict(n=int(len(df)), case_sum=int(ci.sum()), idx_sum=int(ii.sum()),
                g=float(g), n_all=int(groups["ALL"].sum()),
                rb_sum=float(np.nansum(rblend)) if rblend is not None else 0.0)


def _fp_equal(a, b, tol=1e-9):
    if set(a) != set(b):
        return False
    for k in a:
        if isinstance(a[k], float):
            if not (abs(a[k] - b[k]) <= tol * max(1.0, abs(a[k]))):
                return False
        elif a[k] != b[k]:
            return False
    return True


def report(sim, per, group_rows, cuts, save_npz, dom_mag_max, dom_re_min):
    """Aggregate per-seed accumulators into the printed table and the npz.

    Shared verbatim by the single-process path and the array-merge path, so the two cannot drift.
    `per` is a list of (rf, rb), (rf, rb, rk) or (rf, rb, rk, rx) dicts -- one entry per seed, in any
    order (they are exchangeable). The short forms are accepted so dumps written before the
    model-keep (`rk`) and leg-split (`rx`) diagnostics existed still merge; those simply print no
    `mkeep` column and no leg-split section.
    `group_rows` is {group: n_rows}; `cuts` needs only `name` and `proxy` per entry.
    """
    groups = list(group_rows)
    keys = ["__nocut__"] + [c["name"] for c in cuts]
    has_k = all(len(p) > 2 and p[2] is not None for p in per)
    has_x = all(len(p) > 3 and p[3] is not None for p in per)
    # FIDUCIAL MODEL = flow + emulator blend. `flowonly` is kept and printed alongside so the
    # effect of adding the emulator is visible rather than asserted.
    tot = [{gn: {k: p[0][gn][k] + p[1][gn][k] for k in keys} for gn in groups} for p in per]
    flowonly = {gn: {k: float(np.mean([p[0][gn][k] for p in per])) for k in keys} for gn in groups}
    blendonly = {gn: {k: float(np.mean([p[1][gn][k] for p in per])) for k in keys} for gn in groups}

    for gn in groups:
        R0 = sim[gn]["__nocut__"]
        M0 = float(np.mean([t[gn]["__nocut__"] for t in tot]))
        print("\n" + "=" * 104)
        print(f"CONSTGOLD SELECTION, NEAR-DOMAIN CUTS -- {gn}")
        print("=" * 104)
        print(f"  sim R(no cut): unsheared={R0['unsheared']:+.5f} sheared={R0['sheared']:+.5f} "
              f"measured={R0['measured']:+.5f}")
        # PER-SEED no-cut m, then averaged -- matches how the per-object dumps (fig3) build the
        # ensemble m, so the two paths are comparable term by term.
        mnc_i = np.array([R0["measured"] / t[gn]["__nocut__"] - 1.0 for t in tot]) * 100.0
        m_nc = float(np.mean(mnc_i))
        e_nc = (float(np.std(mnc_i, ddof=1) / np.sqrt(len(mnc_i)))
                if len(mnc_i) > 1 else np.nan)
        print(f"  model R(no cut) = R_flow {flowonly[gn]['__nocut__']:+.5f} + R_blend "
              f"{blendonly[gn]['__nocut__']:+.5f} = {M0:+.5f}")
        print(f"  m at no cut = {m_nc:+.3f}% +- {e_nc:.3f}%   ({len(per)} seeds)")
        if len(per) < 16:
            print("    ^ UNDER-POWERED: `m` is an e-RESPONSE quantity and the e-response standard "
                  "is 16\n      seeds (AGENTS.md). Cuts here are on flux/size, but the number "
                  "reported is a\n      SHAPE bias, so the 16-seed standard binds.")
        print(f"\n  {'cut':>22} {'keep':>6} {'mkeep':>6} {'d%':>7} | {'(1) pure':>9} "
              f"{'(1b)/Rm':>9} {'(3) meas':>9} "
              f"{'(4) MODEL':>17} | {'m':>17} {'dm = m-m(nocut)':>19}")
        rows = []
        mkeeps = []
        for c in cuts:
            k = c["name"]
            s = sim[gn][k]
            # MODEL keep fraction, and its RELATIVE offset from the sim's. This is the control that
            # says whether `dm` can be read as a response error at all: the two sides only cut the
            # same population if they keep the same fraction.
            mk = float(np.mean([p[2][gn][k] for p in per])) if has_k else np.nan
            dk = (mk / s["keep"] - 1.0) * 100.0 if (has_k and s["keep"] > 0) else np.nan
            mkeeps.append((mk, dk))
            c1 = s["unsheared"] / R0["sheared"] * 100.0
            c1b = s["unsheared"] / R0["measured"] * 100.0
            c3 = (s["measured"] / R0["measured"] - 1.0) * 100.0
            # Ratios formed INSIDE each seed, then averaged -- see the module docstring. Doing it on
            # ensemble means and propagating the numerator alone throws away the seed cancellation.
            c4_i = np.array([t[gn][k] / t[gn]["__nocut__"] - 1.0 for t in tot]) * 100.0
            m_i = np.array([s["measured"] / t[gn][k] - 1.0 for t in tot]) * 100.0
            dm_i = m_i - mnc_i
            sd = lambda a: (float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan)
            c4, e4 = float(np.mean(c4_i)), sd(c4_i)
            mm_, em = float(np.mean(m_i)), sd(m_i)
            dm, edm = float(np.mean(dm_i)), sd(dm_i)
            mk, dk = mkeeps[-1]
            mks = f"{mk:>6.3f} {dk:>+6.2f}%" if has_k else f"{'--':>6} {'--':>7}"
            print(f"  {k:>22} {s['keep']:>6.3f} {mks} | {c1:>+8.3f}% {c1b:>+8.3f}% {c3:>+8.3f}% "
                  f"{c4:>+11.3f} +-{e4:<5.3f} | {mm_:>+11.3f} +-{em:<5.3f} "
                  f"{dm:>+12.3f} +-{edm:<5.3f}")
            rows.append((k, s["keep"], c1, c1b, c3, c4, e4, mm_, em, dm, edm,
                         bool(c.get("proxy")), mk, dk))

        # ---- LEG-SPLIT BOUNDARY DIAGNOSTIC -------------------------------------------------------
        # Column (1) is the SIM's moving-boundary term. Until now the model had no counterpart at
        # all: the two legs were averaged inside model_selected, which is right for the response but
        # destroys the leg DIFFERENCE that IS the boundary. Three parallel measures are printed, each
        # the same construction on both sides -- average a shear-INDEPENDENT per-object weight over
        # the leg-dependent selected set, then leg_avg. They differ only in the weight:
        #   shape   weight = unsheared intrinsic shape projection. Model counterpart of column (1),
        #           normalised the same way (over R_sheared(no cut)), so sim and model are directly
        #           comparable numbers on one scale.
        #   blend   weight = the emulator's per-object R_blend. Available on BOTH sides because
        #           R_blend is a per-object scalar and the sim's own leg masks can be applied to it.
        #   count   weight = 1. dK/dg: how much of the population the boundary sweeps per unit shear.
        # A model whose selection boundary moved like the sim's would match all three.
        xrows = []
        if has_x:
            sx = all(("keep_plus" in sim[gn][c["name"]]) for c in cuts)
            print(f"\n  LEG-SPLIT BOUNDARY (shear-independent weight averaged over each leg's own "
                  f"selected set)")
            print(f"  {'cut':>22} | {'shape: sim(1)':>13} {'model':>16} {'mdl-sim':>9} | "
                  f"{'blend: sim':>10} {'model':>14} | {'count: sim':>10} {'model':>14}")
            for c in cuts:
                k = c["name"]
                s = sim[gn][k]
                sd = lambda a: (float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan)
                bs_i = np.array([p[3][gn][k]["sel_resp"] for p in per]) / R0["sheared"] * 100.0
                bb_i = np.array([p[3][gn][k]["b_resp"] for p in per])
                bk_i = np.array([p[3][gn][k]["keep_resp"] for p in per])
                bs, ebs = float(np.mean(bs_i)), sd(bs_i)
                bb, ebb = float(np.mean(bb_i)), sd(bb_i)
                bk, ebk = float(np.mean(bk_i)), sd(bk_i)
                s_shape = s["unsheared"] / R0["sheared"] * 100.0
                s_blend = s.get("b_resp", np.nan) if sx else np.nan
                s_keep = s.get("keep_resp", np.nan) if sx else np.nan
                print(f"  {k:>22} | {s_shape:>+12.4f}% {bs:>+9.4f}+-{ebs:<5.4f} "
                      f"{bs - s_shape:>+8.4f}% | {s_blend:>+10.5f} {bb:>+8.5f}+-{ebb:<5.5f} | "
                      f"{s_keep:>+10.4f} {bk:>+8.4f}+-{ebk:<5.4f}")
                xrows.append((s_shape, bs, ebs, s_blend, bb, ebb, s_keep, bk, ebk,
                              s.get("keep_plus", np.nan), s.get("keep_minus", np.nan),
                              float(np.mean([p[3][gn][k]["keep_plus"] for p in per])),
                              float(np.mean([p[3][gn][k]["keep_minus"] for p in per]))))
            print("    shape/blend/count = the SAME boundary construction with three weights;")
            print("    shape is column (1)'s definition, so sim(1) and model are on one scale.")
            print("    [T] rows cut on a conditioning INPUT: both sides select identical objects, so")
            print("    every entry must be exactly 0 -- that is the null control for this block.")

        # Persist the table so figures read DATA, never a parsed log or a transcribed number.
        if save_npz and gn == "ALL":
            kw = {}
            if xrows:
                kw = dict(bnd_shape_sim=np.array([r[0] for r in xrows], float),
                          bnd_shape_model=np.array([r[1] for r in xrows], float),
                          bnd_shape_model_err=np.array([r[2] for r in xrows], float),
                          bnd_blend_sim=np.array([r[3] for r in xrows], float),
                          bnd_blend_model=np.array([r[4] for r in xrows], float),
                          bnd_blend_model_err=np.array([r[5] for r in xrows], float),
                          bnd_keep_sim=np.array([r[6] for r in xrows], float),
                          bnd_keep_model=np.array([r[7] for r in xrows], float),
                          bnd_keep_model_err=np.array([r[8] for r in xrows], float),
                          sim_keep_plus=np.array([r[9] for r in xrows], float),
                          sim_keep_minus=np.array([r[10] for r in xrows], float),
                          model_keep_plus=np.array([r[11] for r in xrows], float),
                          model_keep_minus=np.array([r[12] for r in xrows], float))
            np.savez(save_npz, **kw,
                     name=np.array([r[0] for r in rows]),
                     keep=np.array([r[1] for r in rows], float),
                     pure_sel=np.array([r[2] for r in rows], float),
                     pure_sel_meas=np.array([r[3] for r in rows], float),
                     measured=np.array([r[4] for r in rows], float),
                     model_m=np.array([r[5] for r in rows], float),
                     model_sem=np.array([r[6] for r in rows], float),
                     m_cut=np.array([r[7] for r in rows], float),
                     m_cut_err=np.array([r[8] for r in rows], float),
                     dm=np.array([r[9] for r in rows], float),
                     dm_err=np.array([r[10] for r in rows], float),
                     is_proxy=np.array([r[11] for r in rows], bool),
                     model_keep=np.array([r[12] for r in rows], float),
                     keep_offset=np.array([r[13] for r in rows], float),
                     R_sim_meas=R0["measured"], R_model=M0,
                     R_flow=flowonly[gn]["__nocut__"], R_blend=blendonly[gn]["__nocut__"],
                     m_nocut=m_nc, m_nocut_err=e_nc, n_seeds=len(per),
                     n_rows=int(group_rows[gn]),
                     dom_mag_max=dom_mag_max, dom_re_min=dom_re_min)
            print(f"\n  saved table -> {save_npz}")
    print_footer()


def print_footer():
    print("\n  (1) pure sel = R_unsheared(cut)/R_sheared(no cut): same raw shape both legs, so the")
    print("      shape response is 0 and this is the pure MOVING-BOUNDARY term. A shear-independent")
    print("      population change cancels here by construction.")
    print("  (1b) = same numerator over R_meas(no cut). DIAGNOSTIC ONLY -- (1) averages intrinsic")
    print("      shapes and (3) measured ones, so a denominator swap alone does not put them on one")
    print("      scale; the missing dilution factor opposes it. Read (1) qualitatively vs (3).")
    print("  (3) measured = R_meas(cut)/R_meas(no cut) - 1     (4) MODEL m = same ratio for the flow")
    print("      (3) mixes POPULATION RE-WEIGHTING (which galaxies survive; not a bias) with the")
    print("      moving-boundary term (1). (3) minus (1) is roughly the re-weighting part.")
    print("  m  = R_meas(cut)/R_model(cut) - 1   -- residual bias AT the cut, sim/model-1 like fig3.")
    print("  dm = m(cut) - m(no cut)             -- the SELECTION-INDUCED excess.")
    print("  All errors are the spread of the PER-SEED ratio. dm and (4) cancel the common-mode seed")
    print("  offset; absolute m does not (the sim side has no seed dependence), so absolute m needs")
    print("  the full 16-seed ensemble.")
    print("  Rows WITHOUT (*) cut sim and model on the SAME real measured quantity at the SAME")
    print("  absolute threshold: no proxy, no quantile matching. (*) rows are real S/N, which the")
    print("  flow cannot form (no FLUX_AUTO/FLUXERR_AUTO output), so those keep the mag+size proxy.")
    print("CG_NEARDOMAIN_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = keep the whole in-domain population (the default). A\n                          subsample is NOT free: per-object response scatter is std ~5.1\n                          against a mean ~0.86, so 4M of 11.67M moved R_sim by 0.24%% and\n                          every absolute m with it -- larger than the 16-seed error. It\n                          hits EVERY row, not just no cut (measured 2026-07-31g: +0.03 to\n                          +0.40 pts across cuts, +0.25 at no cut). Sharing the draw between\n                          R_sim and R_model does not cancel it, because the draw moves the\n                          noisy R_sim and barely moves the smooth R_model. Only dm and\n                          column (4), both model-vs-model, actually cancel.")
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[26.0, 25.5, 25.0])
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[0.30, 0.40, 0.60, 0.70],
                    help="MEASURED flux_radius cuts, arcsec. NOTE: measured size is floored\n                          by the PSF (R50=0.527\"), so cuts below ~0.5\" keep ~100%% and are\n                          no-ops by construction -- kept because that IS the finding. 0.70\n                          is retained from the previous list: it is the one row that showed\n                          a real failure (-4.12%%), and dropping it would hide a known problem.")
    ap.add_argument("--sn-cuts", type=float, nargs="+", default=[8.0, 9.0, 10.0],
                    help="cuts on the real MEASURED S/N (absolute, not keep-fractions)")
    ap.add_argument("--true-cuts", action="store_true",
                    help="add cuts on TRUE mag/Re (rows tagged [T]). These are conditioning\n"
                         "                          inputs, not flow outputs, so both sides select\n"
                         "                          IDENTICAL objects -- dm is then a pure response\n"
                         "                          error with no selection-modelling channel. The\n"
                         "                          decisive control against the measured rows.")
    ap.add_argument("--combo-cuts", nargs="+", default=["26.0:0.30", "25.0:0.40"],
                    metavar="MAG:SIZE",
                    help="joint magnitude+size cuts, as 'mag:size_arcsec' pairs. Default reproduces\n"
                         "                          the two historical rows. NOTE both defaults pair a\n"
                         "                          real magnitude cut with a size threshold that is a\n"
                         "                          NO-OP (R>0.30\"/0.40\" keep ~100%% because the PSF\n"
                         "                          floors measured flux_radius at R50=0.527\"), so\n"
                         "                          neither is a genuine two-axis cut and neither can\n"
                         "                          test whether the magnitude and size boundary terms\n"
                         "                          (opposite in sign) cancel. Pass e.g. 25.0:0.60 for\n"
                         "                          a pair where BOTH thresholds bite.")
    ap.add_argument("--true-combo-cuts", nargs="*", default=[], metavar="MAG:RE",
                    help="joint TRUE-property cuts, as 'true_mag:true_Re' pairs (needs\n"
                         "                          --true-cuts). Empty by default. Both sides select\n"
                         "                          identical objects, so `m` on these rows is the\n"
                         "                          pure shape-response bias with the moving-boundary\n"
                         "                          term identically zero.")
    ap.add_argument("--newcomer-cuts", type=float, nargs="*", default=[], metavar="MAG",
                    help="split the population at a magnitude threshold into the three disjoint\n"
                         "                          pieces both / newcomer / dropout, by whether the\n"
                         "                          MEASURED and TRUE magnitude each pass it. Empty by\n"
                         "                          default. All three are frozen masks, so the\n"
                         "                          boundary term is identically zero and `m` is the\n"
                         "                          pure shape-response bias of that subpopulation.\n"
                         "                          Explains why the measured and true magnitude cuts\n"
                         "                          disagree in sign at the same threshold.")
    ap.add_argument("--true-mag-cuts", type=float, nargs="+", default=[25.5, 25.0],
                    help="TRUE-magnitude thresholds for --true-cuts (domain is already mag<26)")
    ap.add_argument("--true-re-cuts", type=float, nargs="+", default=[0.4, 0.5],
                    help="TRUE-Re thresholds in arcsec for --true-cuts (domain is already Re>0.3)")
    ap.add_argument("--complements", action="store_true",
                    help="also emit the COMPLEMENT of each mag/size cut (rows tagged [c]). Every\n"
                         "                          default cut keeps the bright/large corner, so\n"
                         "                          their dm CANNOT disagree in sign -- the\n"
                         "                          complements make that constraint measurable\n"
                         "                          rather than letting 'positive at every cut' read\n"
                         "                          as independent evidence.")
    ap.add_argument("--sn-a", type=float, default=-0.3676)
    ap.add_argument("--sn-b", type=float, default=-0.7736)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--blend-lookup",
                    default="results/blend_lookup_indomtuned_c40-139.feather",
                    help="per-object emulator R_blend; '' disables (flow-only model)")
    ap.add_argument("--min-blend-match", type=float, default=0.99,
                    help="refuse if fewer than this fraction of rows get an R_blend")
    ap.add_argument("--save-npz", default="results/constgold_neardomain_table.npz",
                    help="persist the ALL table so figures read data, not a log")
    ap.add_argument("--dom-mag-max", type=float, default=26.0)
    ap.add_argument("--dom-re-min", type=float, default=0.3)
    ap.add_argument("--dump-per-seed", default=None,
                    help="ARRAY MODE: score exactly ONE --ckpt and write its accumulators (plus the "
                         "sim side and a population fingerprint) to this JSON, instead of building "
                         "the table. Merge with scripts/merge_neardomain_seeds.py. Seeds are "
                         "independent given the population, so this turns a ~2h20 sequential 16-seed "
                         "run into ~16 min of wall clock at the cost of rebuilding the population "
                         "in every task.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    df = pf.read_table(args.cat, columns=NEED, memory_map=True).to_pandas()
    n_raw = len(df)
    df = df[df["case"] >= args.min_case].reset_index(drop=True)
    print(f"constgold(+meas): {n_raw:,} -> {len(df):,} with case>={args.min_case} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if len(df) == 0:
        raise SystemExit("REFUSING: no rows after the case cut.")
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"], df["e2_input_rot0_p"] = e1i, e2i
    df["gamma1_input_p"], df["gamma2_input_p"] = 0.0, 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    # DOMAIN CUT is mandatory: the dom6x6 flow was trained with primary_mag_max=26.0 /
    # primary_re_min=0.3, and scoring it outside that box put R_model(no cut) at +0.173 vs ~0.29,
    # inflating every model entry ~4x (run 15365425). Sim and model must share one population.
    dom = (df["r_input_p"].to_numpy(float) < args.dom_mag_max) & \
          (df["Re_input_p"].to_numpy(float) > args.dom_re_min)
    df = df[dom].reset_index(drop=True)
    print(f"  after selection + DOMAIN cut: {len(df):,}", flush=True)
    if args.max_rows and len(df) > args.max_rows:
        sel = np.random.default_rng(0).choice(len(df), size=args.max_rows, replace=False)
        sel.sort()
        df = df.iloc[sel].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} (seed 0)", flush=True)

    # ---- emulator R_blend: the fiducial model is flow + blend, not flow alone ------------------
    rblend = None
    if args.blend_lookup:
        lk = pf.read_table(args.blend_lookup, memory_map=True).to_pandas()
        j = df[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left")
        rblend = j["R_blend"].to_numpy(float)
        matched = np.isfinite(rblend)
        print(f"blend lookup {os.path.basename(args.blend_lookup)}: matched "
              f"{100*matched.mean():.2f}%  <R_blend>={np.nanmean(rblend):+.5f}", flush=True)
        if matched.mean() < args.min_blend_match:
            raise SystemExit(
                f"REFUSING: only {100*matched.mean():.1f}% of rows have an emulator R_blend. "
                "Unmatched rows would silently fall back to R_blend=0 and collapse the blend term "
                "-- exactly the coverage artifact that produced a spurious +28.9% m on the wide "
                "population (job 15366950). Use a lookup that covers this population.")

    cf = pf.read_table(args.crowd).to_pandas()
    fcols = [c for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max") if c in cf.columns]
    mm = df[["case", "input_index"]].merge(cf[["case", "input_index", *fcols]],
                                           on=["case", "input_index"], how="left")
    for c in fcols:
        df[c] = mm[c].to_numpy(float)

    g = float(np.median(df.shear_magnitude.to_numpy(float)))
    sa = df.shear_angle.to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    me1p, me2p = df.measured_e1_plus.to_numpy(float), df.measured_e2_plus.to_numpy(float)
    me1m, me2m = df.measured_e1_minus.to_numpy(float), df.measured_e2_minus.to_numpy(float)
    myp, st = me1p * c2 + me2p * s2, df.et_plus.to_numpy(float)
    okp = np.isfinite(myp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((myp * st)[okp]) > 0 else -1.0
    proj = lambda a, b: sign * (a * c2 + b * s2)

    i1, i2 = df["e1_input_rot0_p"].to_numpy(float), df["e2_input_rot0_p"].to_numpy(float)
    gh1, gh2 = c2, s2
    e1p, e2p = apply_shear_to_ellipticity(i1, i2, g * gh1, g * gh2)
    e1m, e2m = apply_shear_to_ellipticity(i1, i2, -g * gh1, -g * gh2)
    kinds = [("unsheared", proj(i1, i2), proj(i1, i2)),
             ("sheared", proj(e1p, e2p), proj(e1m, e2m)),
             ("measured", proj(me1p, me2p), proj(me1m, me2m))]
    # The shear-INDEPENDENT per-object weight that defines column (1). Handed to the model side too,
    # so its boundary term is built from the identical quantity (see model_selected's docstring).
    selw = proj(i1, i2)

    magp = df["measured_mag_auto_plus"].to_numpy(float)
    magm = df["measured_mag_auto_minus"].to_numpy(float)
    szp = df["measured_flux_radius_plus"].to_numpy(float)
    szm = df["measured_flux_radius_minus"].to_numpy(float)
    snp, snm = df["S/N_plus"].to_numpy(float), df["S/N_minus"].to_numpy(float)
    fin = np.isfinite(snp) & np.isfinite(snm) & np.isfinite(magp) & np.isfinite(magm) \
        & np.isfinite(szp) & np.isfinite(szm)
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    if rblend is not None:
        # Restrict BOTH sim and model to rows carrying an emulator R_blend. Keeping unmatched rows
        # and letting them default to 0 is exactly the silent-coverage failure mode found in job
        # 15366950; dropping them keeps the two sides on one population.
        fin &= np.isfinite(rblend)
    print(f"finite: {int(fin.sum()):,}   g={g:.4f}", flush=True)
    groups = {"ALL": fin}

    # ---- cut list: sim (per-leg arrays) and model (absolute thresholds) built together ----------
    up = args.sn_a * magp + args.sn_b * np.log10(np.maximum(szp, 1e-6))
    combo_pairs = []
    for spec in args.combo_cuts:
        try:
            mc_s, sc_s = spec.split(":")
            combo_pairs.append((float(mc_s), float(sc_s)))
        except ValueError:
            raise SystemExit(f"--combo-cuts expects 'MAG:SIZE' pairs, got {spec!r}")
    cuts = []
    for c in args.mag_cuts:
        cuts.append(dict(name=f"mag<{c:g}", proxy=False,
                         sim=[(magp, magm, c, False)],
                         conds=[dict(var="mag", thr=float(c), keep_high=False)]))
    for a in args.size_cuts:
        px = a / PX
        cuts.append(dict(name=f'R>{a:.2f}"', proxy=False,
                         sim=[(szp, szm, px, True)],
                         conds=[dict(var="size", thr=float(np.log(px)), keep_high=True)]))
    for mc, sc_ in combo_pairs:
        cuts.append(dict(name=f'mag<{mc:g} & R>{sc_:.2f}"', proxy=False,
                         sim=[(magp, magm, mc, False), (szp, szm, sc_ / PX, True)],
                         conds=[dict(var="mag", thr=float(mc), keep_high=False),
                                dict(var="size", thr=float(np.log(sc_ / PX)), keep_high=True)]))
    # TRUE-PROPERTY rows. THE decisive control for "is `dm` a response error or a selection-modelling
    # error". A measured cut is applied to the flow's own SAMPLED mag/size, so a too-broad predictive
    # distribution blurs the model's boundary and changes which objects it selects. A cut on TRUE mag
    # / TRUE Re cannot do that: those are conditioning inputs the model is handed, identical on both
    # sides, so the two select exactly the same galaxies and `dm` isolates the response error alone.
    # Compare a true row against the measured row at the same threshold: the difference between them
    # IS the selection-modelling contribution.
    fixed_masks = {}
    if args.true_cuts:
        tmag = df["r_input_p"].to_numpy(float)
        tre = df["Re_input_p"].to_numpy(float)
        for c in args.true_mag_cuts:
            key = f"tmag<{c:g}"
            fixed_masks[key] = tmag < c
            cuts.append(dict(name=f"{key} [T]", proxy=False,
                             sim=[(tmag, tmag, float(c), False)],
                             conds=[dict(var="fixed", key=key)]))
        for a in args.true_re_cuts:
            key = f"tRe>{a:g}"
            fixed_masks[key] = tre > a
            cuts.append(dict(name=f"{key} [T]", proxy=False,
                             sim=[(tre, tre, float(a), True)],
                             conds=[dict(var="fixed", key=key)]))
        # Joint TRUE cut. The measured two-axis row showed the magnitude and size boundary terms are
        # strongly SUB-ADDITIVE because both cuts remove the same faint-and-small galaxies
        # (2026-08-01k). That row still mixes boundary and response. Its true-property twin removes
        # the boundary entirely -- both sides select identical objects -- so `m` on this row is the
        # pure SHAPE-response bias of the joint population, and the difference against the measured
        # row at the same thresholds is the selection-modelling contribution.
        for spec in args.true_combo_cuts:
            try:
                mc_s, sc_s = spec.split(":")
                mc, sc_ = float(mc_s), float(sc_s)
            except ValueError:
                raise SystemExit(f"--true-combo-cuts expects 'MAG:RE' pairs, got {spec!r}")
            key = f"tmag<{mc:g} & tRe>{sc_:g}"
            fixed_masks[key] = (tmag < mc) & (tre > sc_)
            cuts.append(dict(name=f"{key} [T]", proxy=False,
                             sim=[(tmag, tmag, mc, False), (tre, tre, sc_, True)],
                             conds=[dict(var="fixed", key=key)]))

    # NEWCOMER / BOTH / DROPOUT rows. A measured magnitude cut and a true one at the SAME threshold
    # disagree in SIGN (measured `mag<25` -> m = +0.633, true `tmag<25` -> m = -0.406), which cannot
    # be the moving boundary: that term is only -0.077 with the model at -0.102. The two cuts simply
    # select different galaxies, because measurement brightens 71% of objects (median -0.090 mag) and
    # the brightening is driven by neighbour flux in MAG_AUTO. These rows split the measured-cut
    # population into the three disjoint pieces that make the difference explicit:
    #   both     = measured in AND true in      (the honest overlap)
    #   newcomer = measured in but true OUT     (brightened across the threshold)
    #   dropout  = true in but measured OUT     (dimmed across it)
    # All three are FROZEN masks -- identical on both legs -- so each row's boundary term is exactly
    # 0 and its `m` is the pure shape-response bias of that subpopulation, directly comparable to the
    # [T] rows. `both + newcomer` reconstitutes the measured cut; `both + dropout` the true cut.
    for c in args.newcomer_cuts:
        mm_ = 0.5 * (df["measured_mag_auto_plus"].to_numpy(float)
                     + df["measured_mag_auto_minus"].to_numpy(float))
        tm_ = df["r_input_p"].to_numpy(float)
        for tag, mkh, tkh in (("both", False, False), ("newcomer", False, True),
                              ("dropout", True, False)):
            key = f"{tag} mag<{c:g}"
            mmask = (mm_ > c) if mkh else (mm_ < c)
            tmask = (tm_ > c) if tkh else (tm_ < c)
            fixed_masks[key] = mmask & tmask
            cuts.append(dict(name=f"{key} [F]", proxy=False,
                             sim=[(mm_, mm_, float(c), mkh), (tm_, tm_, float(c), tkh)],
                             conds=[dict(var="fixed", key=key)]))

    # COMPLEMENT rows. Every cut above keeps the bright/large corner, so they cannot disagree in
    # sign: R(no cut) is the keep-weighted blend of a cut and its complement, so if `m` rises on the
    # kept side it MUST fall on the discarded side. Quoting "positive at every cut" as if the cuts
    # were independent evidence is therefore circular. These rows keep the OTHER side of the same
    # thresholds and make the constraint visible instead of leaving it implicit.
    if args.complements:
        for c in args.mag_cuts:
            cuts.append(dict(name=f"mag>{c:g} [c]", proxy=False,
                             sim=[(magp, magm, c, True)],
                             conds=[dict(var="mag", thr=float(c), keep_high=True)]))
        for a in args.size_cuts:
            px = a / PX
            cuts.append(dict(name=f'R<{a:.2f}" [c]', proxy=False,
                             sim=[(szp, szm, px, False)],
                             conds=[dict(var="size", thr=float(np.log(px)), keep_high=False)]))

    for thr in args.sn_cuts:
        # (*) real S/N on the sim; the flow cannot form it (no FLUX_AUTO/FLUXERR_AUTO output), so the
        # MODEL cuts the mag+size proxy at the quantile that keeps the SAME fraction -- the only
        # proxy left in this table. Matching by keep-fraction is required because the proxy is a
        # different variable, so the same numeric threshold would not select a comparable set.
        kf = float(np.mean(snp[fin] > thr))
        pthr = float(np.quantile(up[fin], 1.0 - kf))
        cuts.append(dict(name=f"S/N>{thr:g} (*)", proxy=True,
                         sim=[(snp, snm, float(thr), True)],
                         conds=[dict(var="proxy", a=args.sn_a, b=args.sn_b, thr=pthr,
                                     keep_high=True)]))

    # ---- sim ------------------------------------------------------------------------------------
    sim = {gn: {} for gn in groups}
    for gn, gm in groups.items():
        ng = max(int(gm.sum()), 1)
        sim[gn]["__nocut__"] = {k: leg_avg(float(a[gm].mean()), float(b[gm].mean()), g)
                                for k, a, b in kinds}
        # LEG SPLIT on the sim side, mirroring the model's `outx`. At no cut the two legs select the
        # identical (whole) sample by construction, so both boundary terms are exactly 0 -- kept
        # explicitly rather than implied, because the model's no-cut entry is 0 only for the same
        # reason and the two must be checkable side by side.
        _bnc = float(np.nanmean(rblend[gm])) if rblend is not None else np.nan
        sim[gn]["__nocut__"].update(keep=1.0, keep_plus=1.0, keep_minus=1.0, keep_resp=0.0,
                                    b_plus=_bnc, b_minus=_bnc, b_resp=0.0)
        for c in cuts:
            pp, pm = gm.copy(), gm.copy()
            for xp, xm, thr, kh in c["sim"]:
                pp &= (xp > thr) if kh else (xp < thr)
                pm &= (xm > thr) if kh else (xm < thr)
            sim[gn][c["name"]] = {k: leg_avg(float(a[pp].mean()), float(b[pm].mean()), g)
                                  for k, a, b in kinds}
            sim[gn][c["name"]]["keep"] = 0.5 * (pp.sum() + pm.sum()) / max(gm.sum(), 1)
            kp_, km_ = int(pp.sum()) / ng, int(pm.sum()) / ng
            bp_ = float(rblend[pp].mean()) if (rblend is not None and pp.any()) else np.nan
            bm_ = float(rblend[pm].mean()) if (rblend is not None and pm.any()) else np.nan
            sim[gn][c["name"]].update(
                keep_plus=kp_, keep_minus=km_, keep_resp=float(leg_avg(kp_, km_, g)),
                b_plus=bp_, b_minus=bm_, b_resp=float(leg_avg(bp_, bm_, g)))

    cuts_meta = [dict(name=c["name"], proxy=bool(c.get("proxy"))) for c in cuts]
    group_rows = {gn: int(gm.sum()) for gn, gm in groups.items()}
    fp = fingerprint(df, g, rblend, groups)

    # ---- ARRAY MODE: score ONE checkpoint and dump, leaving aggregation to the merge step --------
    # Seeds are independent given the prepared population, so 16 sequential scorings (~8 min each on
    # top of a ~8 min setup) is ~2h20 of wall clock for work that fans out perfectly. Each array task
    # rebuilds the population (the setup is duplicated, which is the price) and scores one seed, so
    # wall clock collapses to setup + one seed. The population fingerprint is written with every
    # dump and checked at merge, so a task built from a different catalogue or cut list cannot be
    # averaged in silently.
    if args.dump_per_seed:
        if len(args.ckpt) != 1:
            raise SystemExit(f"--dump-per-seed scores exactly one checkpoint, got {len(args.ckpt)}")
        ck = args.ckpt[0]
        bundle = load_measurement_model(ck, device=device)
        rf, rb, rk, rx = model_selected(bundle, df, g, gh1, gh2, (i1, i2), cuts, groups,
                                        args.n_samples, args.batch_size, args.flow_seed, device,
                                        sign=sign, rblend=rblend, fixed_masks=fixed_masks,
                                        sel_weight=selw)
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_per_seed)), exist_ok=True)
        payload = dict(rf=rf, rb=rb, rk=rk, rx=rx, ckpt=os.path.basename(ck), fingerprint=fp,
                       sim=sim, cuts=cuts_meta, group_rows=group_rows,
                       dom_mag_max=args.dom_mag_max, dom_re_min=args.dom_re_min)
        with open(args.dump_per_seed, "w") as fh:
            json.dump(payload, fh)
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"  dumped -> {args.dump_per_seed}")
        print("CG_ND_SEED_DONE", flush=True)
        return

    # ---- model ----------------------------------------------------------------------------------
    per = []
    for ck in args.ckpt:
        bundle = load_measurement_model(ck, device=device)
        rf, rb, rk, rx = model_selected(bundle, df, g, gh1, gh2, (i1, i2), cuts, groups,
                                        args.n_samples, args.batch_size, args.flow_seed, device,
                                        sign=sign, rblend=rblend, fixed_masks=fixed_masks,
                                        sel_weight=selw)
        per.append((rf, rb, rk, rx))
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
    report(sim, per, group_rows, cuts_meta, args.save_npz, args.dom_mag_max, args.dom_re_min)




if __name__ == "__main__":
    main()
