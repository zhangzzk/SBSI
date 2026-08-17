"""WHICH COMPONENT owns the per-bin response error: the FLOW's self response, or the EMULATOR's
R_blend?

WHY THIS EXISTS. The owner's reading of fig2 vs fig5: fig5 (flow alone vs half-shear self response)
looks consistent across most bins while fig2 (flow + emulator vs constgold TOTAL) does not, which
would point at the emulator. Reading two figures against each other is suggestive but not a
measurement -- they sit on DIFFERENT populations (2.36M half-shear rows vs 11.67M in-domain constgold)
and DIFFERENT response extractions (forward 0->+g vs antithetic +-g), and the extraction difference is
a known real effect, not a rounding detail. This script turns the comparison into numbers on ONE
population.

THE IDENTITY IT USES. On constgold, per bin:

    required_blend(bin) = <r_sim>(bin) - <R_flow>(bin)

is what R_blend WOULD have to be for the model to land on the simulation in that bin. Against the
emulator's actual prediction:

    gap(bin) = required_blend(bin) - <R_blend_emulator>(bin)
             = [true_blend - emulator_blend]  +  [true_self - flow_self]

**The gap is NOT the emulator's error on its own** -- it absorbs the flow's self-response error too.
That is exactly why fig5 is needed: it measures the second bracket independently. This script prints
the gap per bin; the fig5 residuals on the same axis are printed alongside so the two brackets can be
read together, WITHOUT pretending the subtraction is clean across two populations and two extractions.

WHAT WOULD SETTLE IT EITHER WAY:
  - gap large and structured, fig5 residual small on the same axis  -> EMULATOR
  - gap tracks the fig5 residual bin for bin                         -> FLOW
  - both large and uncorrelated                                      -> both, and neither alone fixes it

NO CUTS, NO TUNING, NO FIT. This is a diagnostic read of existing dumps. It introduces no threshold
and cannot promote or demote a model -- constgold is evaluation-only (the R_blend firewall).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plotting.plot_fid_flow_figures import (  # noqa: E402
    BLEND_LOOKUP, CONST_CAT, CROWD, SELFRESP, _read_key_table, load_dumps)

PANELS = [("S/N_plus", "primary flux  S/N", True, None, "SN"),
          ("Re_input_p", "primary size  Re [arcsec]", False, None, "Re_input_p"),
          ("nbr_flux_near", "neighbour / blend flux", True, 1e-3, "nbr_flux_near")]


def edges_for(x, nb, logx, xmin):
    m = np.isfinite(x)
    if xmin is not None:
        m &= x > xmin
    xx = np.log10(x[m]) if logx else x[m]
    return np.linspace(np.nanpercentile(xx, 0.5), np.nanpercentile(xx, 99.5), nb + 1), m


def fig5_residuals(col, nb, logx, xmin, edges=None):
    """Flow-ONLY residual vs half-shear self response, on the SAME axis as the constgold side.

    `edges` MUST normally be supplied. The first version of this function derived its own edges from
    the half-shear x distribution and the caller then paired the two tables BY BIN INDEX. Those are
    different populations (2.36M half-shear rows vs 11.67M in-domain constgold), so equal-percentile
    edges do NOT cover equal Re ranges, and bin i on one side was being subtracted from a different Re
    range on the other. Passing constgold's edges puts both curves on one physical grid and removes
    that misalignment; the two populations still differ, but at least each row of the table now refers
    to the same interval of true size.
    """
    if not os.path.exists(SELFRESP):
        return {}
    df = pd.read_feather(SELFRESP)
    if col not in df.columns:
        return {}
    sc = sorted([c for c in df.columns if c.startswith("R_flow_s")])
    ys = df["r_sim_self"].to_numpy(float)
    ym = np.column_stack([df[c].to_numpy(float) for c in sc])
    x = df[col].to_numpy(float)
    if edges is None:
        ed, m = edges_for(x, nb, logx, xmin)
    else:
        ed = edges
        m = np.isfinite(x)
        if xmin is not None:
            m &= x > xmin
    # Non-finite r_sim_self poisons the bin mean and returned NaN for every bin on the first run
    # (job 15474196). Mask it here, not in edges_for -- the bin EDGES must stay defined by the x
    # distribution alone so they remain comparable with the constgold side.
    m = m & np.isfinite(ys) & np.isfinite(ym).all(1)
    xx = np.log10(x[m]) if logx else x[m]
    idx = np.digitize(xx, ed) - 1
    ys, ym = ys[m], ym[m]
    out = {}
    for b in range(nb):
        s = idx == b
        if s.sum() < 30:
            continue
        sm = float(np.mean(ys[s]))
        out[b] = ((float(ym[s].mean()) / sm - 1.0) * 100.0, int(s.sum()), sm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--save-npz", default="results/blend_vs_flow_perbin.npz")
    args = ap.parse_args()

    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    key = ["case", "input_index"]
    cases = set(ref["case"].unique().tolist())
    ref = ref.merge(_read_key_table(CONST_CAT, key + ["Re_input_p", "S/N_plus"], cases),
                    on=key, how="left")
    ref = ref.merge(_read_key_table(CROWD, key + ["nbr_flux_near"], cases), on=key, how="left")

    rsim = ref["r_sim"].to_numpy(float)
    rb = ref["R_blend"].to_numpy(float)
    rf = flows.astype(np.float64)                       # [nseed, nobj]
    print(f"\nGLOBAL  <r_sim>={np.nanmean(rsim):+.4f}  <R_flow>={np.nanmean(rf):+.4f}  "
          f"<R_blend>={np.nanmean(rb):+.4f}  "
          f"model total={np.nanmean(rf) + np.nanmean(rb):+.4f}  "
          f"required blend={np.nanmean(rsim) - np.nanmean(rf):+.4f}")

    store = {}
    for col, lab, logx, xmin, f5col in PANELS:
        x = ref[col].to_numpy(float)
        ed, m = edges_for(x, args.nbins, logx, xmin)
        xx = np.log10(x[m]) if logx else x[m]
        idx = np.digitize(xx, ed) - 1
        rs_, rb_, rf_ = rsim[m], rb[m], rf[:, m]
        # constgold's edges, so both curves refer to the same intervals of the physical axis
        f5 = fig5_residuals(f5col, args.nbins, logx, xmin, edges=ed)

        print(f"\n{'='*126}\n{lab}\n{'='*126}")
        print(f"  {'bin x':>10}{'n':>11}{'sim tot':>9}{'flow':>9}{'emu bl':>9}"
              f"{'mdl tot':>9}{'resid%':>9} | {'req bl':>9}{'emu/req':>9}"
              f" | {'fig5 flow%':>11}{'flow expl%':>11}{'REMAIND%':>10}")
        rows = []
        for b in range(args.nbins):
            s = idx == b
            n = int(s.sum())
            if n < 30:
                continue
            cx = 10 ** (0.5 * (ed[b] + ed[b + 1])) if logx else 0.5 * (ed[b] + ed[b + 1])
            st = float(np.mean(rs_[s]))
            fl = float(rf_[:, s].mean())
            bl = float(np.nanmean(rb_[s]))
            tot = fl + bl
            res = (tot / st - 1.0) * 100.0
            req = st - fl
            # NOTE: `req - bl` is identically `st - tot`, i.e. the residual restated -- it carries NO
            # information the resid% column does not already have. The first version of this script
            # printed it as a "GAP" and it was redundant. What IS informative is the ratio emu/req
            # (is the emulator too flat?) and the attribution below.
            f5r = f5.get(b, (np.nan, 0, np.nan))[0]
            # Split the total-model residual: the part the flow's OWN measured error accounts for,
            # and the remainder that has to sit with the emulator. The flow enters R_model with
            # weight fl/(fl+bl), so a flow error of f5r% shows up in the total as f5r*share.
            share = fl / tot if tot else np.nan
            expl = f5r * share
            remd = res - expl
            print(f"  {cx:>10.4g}{n:>11,}{st:>9.4f}{fl:>9.4f}{bl:>9.4f}{tot:>9.4f}"
                  f"{res:>+9.2f} | {req:>+9.4f}{bl/req if req else np.nan:>9.2f}"
                  f" | {f5r:>+11.2f}{expl:>+11.2f}{remd:>+10.2f}")
            rows.append((cx, n, st, fl, bl, tot, res, req, f5r, expl, remd, share))
        a = np.array([[r[i] for i in range(12)] for r in rows], float)
        store[f"{col}"] = a
        res_, expl_, remd_, req_, bl_ = a[:, 6], a[:, 9], a[:, 10], a[:, 7], a[:, 4]
        ok = np.isfinite(expl_)
        print(f"  -> total-model residual rms {np.sqrt(np.mean(res_**2)):.2f} pt "
              f"(min {res_.min():+.2f} max {res_.max():+.2f})")
        print(f"  -> emulator R_blend spans {bl_.min():.4f}..{bl_.max():.4f} "
              f"(range {bl_.max()-bl_.min():.4f}); REQUIRED spans {req_.min():.4f}..{req_.max():.4f} "
              f"(range {req_.max()-req_.min():.4f}) -> "
              f"{(req_.max()-req_.min())/(bl_.max()-bl_.min()):.1f}x more structure than the emulator has")
        if ok.sum() > 2:
            print(f"  -> ATTRIBUTION over {int(ok.sum())} bins: explained by flow "
                  f"{np.sqrt(np.mean(expl_[ok]**2)):.2f} pt rms | REMAINDER (emulator) "
                  f"{np.sqrt(np.mean(remd_[ok]**2)):.2f} pt rms | "
                  f"corr(total, flow) = {np.corrcoef(res_[ok], a[ok,8])[0,1]:+.3f}")
            print("     CAVEAT: the flow column is measured on HALF-SHEAR with FORWARD extraction; "
                  "the rest is constgold ANTITHETIC. A size-dependent extraction difference would "
                  "contaminate this split. It also assumes sim additivity R_total = R_self + R_blend.")

    np.savez(args.save_npz, **store, seeds=np.array(seeds))
    print(f"\nsaved -> {args.save_npz}")
    print("REMINDER: `gap` = (true_blend - emulator_blend) + (true_self - flow_self). It is NOT the "
          "emulator error alone. The fig5 column is the second bracket, measured on a DIFFERENT "
          "population (half-shear, forward extraction) -- treat the pairing as indicative, not as a "
          "subtraction.")


if __name__ == "__main__":
    main()
