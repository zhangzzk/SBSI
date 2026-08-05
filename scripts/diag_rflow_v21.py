"""WHERE on the V2.1 domain is `R_flow` short? A per-bin closure map.

2026-08-05h exonerated the blend emulator for V2.1's positive `m`: inside the emulator's inference
box -- the population `m` is scored on -- all four emulators predict the summed `R_blend` to within
0.06% of each other and none is distinguishable from half-shear truth. The V2.1 decomposition then
leaves the residual on the flow: `R_sim = 0.9903`, `R_blend = 0.1179`, so closure needs
`R_flow = 0.8724` against an actual `0.8581`, i.e. the flow is 1.66% low.

WHAT THIS ADDS. That is one global number and it does not say WHERE. This bins the closure residual

    required_R_flow = R_sim - R_blend        (per bin, from the same dump rows)
    residual        = R_flow - required_R_flow

over the axes the V2.1 domain moved relative to the flow's own training domain (true mag < 26,
Re > 0.3). The V2.1 cut is an S/N CURVE with no magnitude ceiling, so it admits faint-but-large
primaries the flow never trained on -- the same out-of-training-domain mechanism that hurt the
emulator OUTSIDE its box. If that is what is happening, the residual will concentrate on rows failing
the flow's training box; if it is flat, the flow is uniformly miscalibrated and the domain is a red
herring.

WHAT MAY AND MAY NOT BE QUOTED FROM THIS. `R_sim` carries no seed dependence while `R_flow` does, so
the residual inherits the full per-seed flow offset and is NOT a difference in which that offset
cancels (AGENTS.md, "the seed offset cancels in a DIFFERENCE, not in an absolute m"). Only 8 V2.1
checkpoints exist (501-509, no 504) against the 16 that AGENTS.md requires for anything reporting
`m`, so this script prints a LOCALISATION, never an `m`, and it refuses to print at all if the caller
implies otherwise. Seeds 510-517 are training; once they land the same axes can carry a quotable
number.

FIREWALL: reads constgold dumps for EVALUATION only -- nothing here tunes, fits, or selects a model,
and no correction is derived or applied.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pyarrow.feather as pf

from sbs_shear import domain as sbs_domain
from scripts.eval_v2_indomain_m import catalogue_true_props


def binned(label, name, key, edges, rsim, rflow, rblend, mask):
    print(f"\n[{label}]")
    print(f"  {name:>22}{'R_sim':>9}{'R_flow':>9}{'R_blend':>9}{'needed':>9}"
          f"{'residual':>10}{'rel %':>9}{'N':>12}")
    for i in range(len(edges) - 1):
        m = mask & (key >= edges[i]) & (key < edges[i + 1])
        n = int(m.sum())
        if n < 2000:
            continue
        rs, rf, rb = rsim[m].mean(), rflow[m].mean(), rblend[m].mean()
        need = rs - rb
        rel = 100 * (rf / need - 1) if need else np.nan
        print(f"  [{edges[i]:>9.2f},{edges[i+1]:>8.2f}){rs:>9.4f}{rf:>9.4f}{rb:>9.4f}{need:>9.4f}"
              f"{rf-need:>+10.4f}{rel:>+9.2f}{n:>12,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-max", type=float, default=26.0, help="flow training box, upper mag")
    ap.add_argument("--re-min", type=float, default=0.3, help="flow training box, lower Re")
    ap.add_argument("--scope", choices=("v21", "train"), default="v21",
                    help="which population to bin over. 'v21' is the deliverable domain; 'train' is "
                         "the flow's whole training box, which SPANS the V2.1 boundary and is the "
                         "only scope that can show where the residual changes sign.")
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if not dumps:
        raise SystemExit(f"no dumps matching {args.dump_glob}")
    print(f"{len(dumps)} seed dumps -- LOCALISATION ONLY, no m is reported "
          f"(AGENTS.md requires 16 seeds for m; V2.1 has 8)", flush=True)

    tp = catalogue_true_props(args.catalogue, args.min_case, t0)
    mag = tp["r_input_p"].to_numpy(float)
    re_ = tp["Re_input_p"].to_numpy(float)
    v21 = sbs_domain.in_domain(mag, re_)

    acc = None
    for d in dumps:
        t = pf.read_table(d, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        if len(t) != len(tp):
            raise RuntimeError(f"{d}: {len(t):,} rows vs catalogue {len(tp):,}; cannot stack")
        cur = np.column_stack([t["r_sim"].to_numpy(zero_copy_only=False).astype(float),
                               t["R_flow"].to_numpy(zero_copy_only=False).astype(float),
                               t["R_blend"].to_numpy(zero_copy_only=False).astype(float)])
        acc = cur if acc is None else acc + cur
    acc /= len(dumps)
    rsim, rflow, rblend = acc[:, 0], acc[:, 1], acc[:, 2]

    intrain = (mag < args.mag_max) & (re_ > args.re_min)
    print(f"\n{'='*104}\nV2.1 CLOSURE RESIDUAL   residual = R_flow - (R_sim - R_blend)\n{'='*104}")
    print(f"V2.1 domain: {int(v21.sum()):,} rows.  Of those, INSIDE the flow's training box "
          f"(mag < {args.mag_max}, Re > {args.re_min}): {int((v21 & intrain).sum()):,} "
          f"({(v21 & intrain).sum()/max(v21.sum(),1):.2%})")
    for nm, m in (("V2.1 domain, ALL", v21),
                  ("V2.1 AND inside flow training box", v21 & intrain),
                  ("V2.1 but OUTSIDE flow training box", v21 & ~intrain)):
        n = int(m.sum())
        if not n:
            continue
        rs, rf, rb = rsim[m].mean(), rflow[m].mean(), rblend[m].mean()
        need = rs - rb
        print(f"  {nm:<38} N={n:>10,}  R_sim={rs:.4f}  R_flow={rf:.4f}  R_blend={rb:.4f}  "
              f"needed={need:.4f}  residual={rf-need:+.4f} ({100*(rf/need-1):+.2f}%)")
    print("\nIf the residual concentrates OUTSIDE the flow's training box, V2.1's problem is that its")
    print("S/N curve admits primaries the flow never trained on. If it is flat, the domain is not it.")

    if args.scope == "v21":
        binned("by PRIMARY TRUE MAG (the axis V2.1 opened up)", "r_input_p", mag,
               [18, 23, 24, 25, 25.5, 26, 26.5, 27, 28], rsim, rflow, rblend, v21)
        binned("by PRIMARY TRUE SIZE", "Re_input_p", re_,
               [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 3.0], rsim, rflow, rblend, v21)
        print("\nDIAG_RFLOW_V21_DONE", flush=True)
        return

    # 2026-08-05: the V2.1 half of the flow training box closes at -1.82% and its complement at
    # +2.86%, so somewhere between them the residual passes through zero. Binning INSIDE V2.1 can
    # never show that crossing -- it lives at the boundary. Bin the whole training box instead, on
    # both axes the V2.1 cut is built from, and let the sign change locate itself.
    sn = sbs_domain.sn_true(mag, re_)
    print(f"\n{'='*104}\nWHERE THE RESIDUAL CHANGES SIGN   (scope: the flow's whole training box, "
          f"mag < {args.mag_max}, Re > {args.re_min})\n{'='*104}")
    print("The V2.1 cut is Re > 0.5 AND sn_true > 10. A residual that crosses zero AT those values is")
    print("the cut splitting one smooth trend; a residual that crosses somewhere else is not.")
    binned("by PRIMARY TRUE SIZE, fine, spanning the Re = 0.5 cut", "Re_input_p", re_,
           [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00, 1.20, 1.50],
           rsim, rflow, rblend, intrain)
    binned("by PRIMARY TRUE S/N, spanning the sn_true = 10 cut", "sn_true", sn,
           [0, 5, 7.5, 10, 15, 20, 30, 50, 100, 1e6], rsim, rflow, rblend, intrain)
    binned("by PRIMARY TRUE MAG", "r_input_p", mag,
           [18, 22, 23, 24, 24.5, 25, 25.5, 26], rsim, rflow, rblend, intrain)

    # Which of the two V2.1 conditions actually carries the split? They overlap heavily, so the
    # marginal tables above cannot separate them; these four cells can.
    print(f"\n[the V2.1 cut is TWO conditions -- which one carries the split?]")
    print(f"  {'cell':>34}{'R_sim':>9}{'R_flow':>9}{'R_blend':>9}{'needed':>9}"
          f"{'residual':>10}{'rel %':>9}{'N':>12}")
    big, bright = re_ > sbs_domain.V21_RE_MIN, sn > sbs_domain.V21_SN_MIN
    for nm, m in ((f"Re>{sbs_domain.V21_RE_MIN} AND sn>{sbs_domain.V21_SN_MIN}  (= V2.1)",
                   intrain & big & bright),
                  (f"Re>{sbs_domain.V21_RE_MIN} but sn<{sbs_domain.V21_SN_MIN}", intrain & big & ~bright),
                  (f"Re<{sbs_domain.V21_RE_MIN} but sn>{sbs_domain.V21_SN_MIN}", intrain & ~big & bright),
                  (f"Re<{sbs_domain.V21_RE_MIN} AND sn<{sbs_domain.V21_SN_MIN}", intrain & ~big & ~bright)):
        n = int(m.sum())
        if n < 2000:
            continue
        rs, rf, rb = rsim[m].mean(), rflow[m].mean(), rblend[m].mean()
        need = rs - rb
        print(f"  {nm:>34}{rs:>9.4f}{rf:>9.4f}{rb:>9.4f}{need:>9.4f}"
              f"{rf-need:>+10.4f}{100*(rf/need-1):>+9.2f}{n:>12,}")
    print("\nDIAG_RFLOW_V21_DONE", flush=True)


if __name__ == "__main__":
    main()
