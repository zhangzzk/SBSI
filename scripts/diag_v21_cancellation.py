"""Is V2.1's positive `m` a NEW bias, or the fiducial's known bias with its cancellation removed?

WHAT TONIGHT ESTABLISHED. V2.1's `m` is not the emulator's fault (2026-08-05h: inside the box every
emulator agrees to 0.06% and none differs from half-shear truth) and not a flow-domain effect
(2026-08-05i: 100% of the V2.1 domain already sits inside the flow's training box, and the closure
residual is FLAT across true mag and size). It is also not specific to the V2.1 flow: on the V2.1
domain the FIDUCIAL dom6x6 flow gives `R_flow = 0.8574` against the V2.1 flow's `0.8581` -- a 0.08%
difference -- so both land near `+1.6%` while the fiducial flow reaches `-0.271%` on its own domain.

THE REMAINING HYPOTHESIS. AGENTS.md already records that the fiducial `m` is a CANCELLATION between
an under-predicting blended majority and an over-predicting remainder. The V2.1 domain (`Re > 0.5`,
`S/N > 10`) keeps 44.8% of the flow-training domain -- the well-resolved half. If it happens to
select the under-predicting side, then V2.1 invented no bias at all: it removed the population that
was hiding one. That reframes the whole V2.1 investigation, so it is worth testing directly rather
than inferring.

THE TEST. Split the flow's training domain into the V2.1 subset and its COMPLEMENT and form `m` in
each, from the same dump rows. If the complement comes out strongly negative and the two average
back to the fiducial value, the cancellation is confirmed.

SEEDS AND ERRORS. Uses the 16 FIDUCIAL dumps, so this reports a quotable `m` (AGENTS.md: the e
response takes 16 seeds; the V2.1 flow has only 8 checkpoints and cannot be used for this). Each
ratio is formed INSIDE each seed and the spread is taken across seeds -- building it from ensemble
means would discard exactly the cancellation being measured.

FIREWALL: constgold is read for EVALUATION only. Nothing is fitted, tuned, selected, or corrected.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pyarrow.feather as pf

from sbs_shear import domain as sbs_domain
from scripts.eval_v2_indomain_m import catalogue_true_props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--re-min", type=float, default=0.3)
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) < 16:
        raise SystemExit(f"REFUSING: {len(dumps)} dumps found; this reports m, which AGENTS.md "
                         f"requires 16 seeds for. Point --dump-glob at the fiducial dumps.")
    print(f"{len(dumps)} seed dumps", flush=True)

    tp = catalogue_true_props(args.catalogue, args.min_case, t0)
    mag = tp["r_input_p"].to_numpy(float)
    re_ = tp["Re_input_p"].to_numpy(float)
    train = (mag < args.mag_max) & (re_ > args.re_min)
    v21 = sbs_domain.in_domain(mag, re_)
    masks = {
        "FLOW TRAINING DOMAIN (the fiducial)": train,
        "  of which V2.1 (Re>0.5 & S/N>10)": train & v21,
        "  of which the COMPLEMENT": train & ~v21,
    }
    for k, v in masks.items():
        print(f"  {k:<38} N={int(v.sum()):>12,}  ({v.sum()/max(train.sum(),1):6.1%} of training domain)")

    per_seed = {k: [] for k in masks}
    comps = {k: [] for k in masks}
    for d in dumps:
        t = pf.read_table(d, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        if len(t) != len(tp):
            raise RuntimeError(f"{d}: {len(t):,} rows vs catalogue {len(tp):,}; cannot stack")
        rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        rf = t["R_flow"].to_numpy(zero_copy_only=False).astype(float)
        rb = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        for k, m in masks.items():                      # ratio formed INSIDE the seed
            a, b, c = rs[m].mean(), rf[m].mean(), rb[m].mean()
            per_seed[k].append(100 * (a / (b + c) - 1))
            comps[k].append((a, b, c))

    print(f"\n{'='*100}\nm PER POPULATION, 16 seeds, ratio formed inside each seed\n{'='*100}")
    print(f"  {'population':<38}{'m':>10}{'sem':>9}{'seed sd':>10}"
          f"{'R_sim':>9}{'R_flow':>9}{'R_blend':>9}")
    out = {}
    for k in masks:
        v = np.array(per_seed[k])
        sem = v.std(ddof=1) / np.sqrt(len(v))
        a, b, c = np.mean(comps[k], axis=0)
        out[k] = v.mean()
        print(f"  {k:<38}{v.mean():>+9.3f}%{sem:>9.3f}{v.std(ddof=1):>10.3f}"
              f"{a:>9.4f}{b:>9.4f}{c:>9.4f}")

    kt, kv, kc = list(masks)
    fv = masks[kv].sum() / masks[kt].sum()
    print(f"\n{'='*100}\nDOES IT CANCEL?\n{'='*100}")
    print(f"  V2.1 is {fv:.1%} of the training domain, the complement {1-fv:.1%}.")
    print(f"  population-weighted mean of the two parts: "
          f"{fv*out[kv] + (1-fv)*out[kc]:+.3f}%   vs the whole: {out[kt]:+.3f}%")
    print("  (weighting m's is approximate -- m is a ratio of means, not a mean -- so treat this as")
    print("   a consistency check on the SIGNS and rough sizes, not an identity.)")
    if out[kv] > 0 > out[kc]:
        print("\n  CONFIRMED: the two halves carry OPPOSITE signs. V2.1 did not create a bias -- it")
        print("  removed the population that was cancelling one. The fiducial's small m on the full")
        print("  training domain is hiding both.")
    else:
        print("\n  NOT the cancellation picture: the halves do not carry opposite signs.")
    print("\nDIAG_V21_CANCELLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
