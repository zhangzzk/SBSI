"""Attribution: is the selection-gate residual a SELECTION failure or shape-response bleed-through?

THE PROBLEM (owner, 2026-07-30). The Stage-2 gate projects the flow's SAMPLED MEASURED shape and
applies the cut to the flow's SAMPLED MEASURED size/mag. So every cut row mixes two errors:

    m_measured(cut)  =  shape-response error  +  selection-response error  +  cross term

and there is no way to attribute the -5.97% at size>4.4. Subtracting the no-cut row via the `shift`
columns is only an approximation -- the shape error sits in BOTH numerator and denominator of that
ratio, so it does not divide out cleanly.

THE FIX. Project the EXACT sheared INTRINSIC shape on BOTH sides, and let the model contribute only
the SELECTION:

    sim   : select on the CATALOGUE's measured size/mag -> average intrinsic sheared shapes
    model : select on the FLOW's SAMPLED measured size/mag -> average THE SAME intrinsic shapes

Both sides then average identical, exactly-known, noise-free shapes. The base response cancels in m,
so whatever survives comes only from WHICH objects were selected. That is the selection bias alone.

BUILT-IN CORRECTNESS CHECK. In intrinsic mode with NO CUT, sim and model average the same
deterministic per-object numbers over the same objects, so

    m_intrinsic(NO CUT) == 0 EXACTLY.

Any deviation beyond float noise means the mode is mis-wired -- this is checked and reported, and it
is the reason to trust (or not trust) every other row in the intrinsic column.

WHAT THIS DOES *NOT* DO -- state it before reading the table. Selection and shape are NOT
independent: the cut is on measured size/mag, which are correlated with measured shape inside the
flow's joint. Replacing measured shape with intrinsic shape deliberately BREAKS that correlation, so
this measures "selection bias with exact shapes". The total is therefore NOT
shape-term + selection-term; a cross term is set aside by construction. Do not expect the two columns
to add up to m_measured, and do not quote their difference as "the shape term" -- it is
shape + cross.

BONUS. In intrinsic mode the projection is deterministic per object, so the sampled-shape noise
(per-draw sd ~0.3) leaves the estimator entirely; only the pass/fail indicator stays stochastic. The
intrinsic column should therefore be far better converged at the same n_samples.

FIREWALL: half-shear legs, ISOLATED (R_blend ~ 0). No emulator, no constgold. Trains nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from eval_selection_response import (  # noqa: E402
    CAT, CROWD, NN, build_base, model_selected_response, truth_selected_response,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[2.5, 2.9, 3.5, 4.4])
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.0, 24.5, 25.0])
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--qmc", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpts={len(args.ckpt)}  n_samples={args.n_samples}  "
          f"qmc={args.qmc}", flush=True)

    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                    args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    cuts = ([dict(name=f"size>{c:g}", dim=3, thr=float(np.log(c)), keep_high=True, raw=c, kind="size")
             for c in args.size_cuts] +
            [dict(name=f"mag<{c:g}", dim=2, thr=float(c), keep_high=False, raw=c, kind="mag")
             for c in args.mag_cuts])
    names = ["__nocut__"] + [c["name"] for c in cuts]

    bundles = [load_measurement_model(c, device=device) for c in args.ckpt]
    out = {}
    for mode in ("measured", "intrinsic"):
        intr = (mode == "intrinsic")
        # --- truth ---
        sim = {}
        R_nc, _, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso,
                                                "measured_flux_radius", -1e9, True, intrinsic=intr)
        sim["__nocut__"] = R_nc
        for c in cuts:
            xcol = "measured_flux_radius" if c["kind"] == "size" else "measured_mag_auto"
            sim[c["name"]], _, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, xcol,
                                                              c["raw"], c["keep_high"], intrinsic=intr)
        # --- model (ensemble over ckpts) ---
        per = []
        for b in bundles:
            per.append(model_selected_response(b, base, gh1, gh2, gmed, iso, cuts, args.n_samples,
                                               args.batch_size, args.flow_seed, device,
                                               qmc=args.qmc, intrinsic=intr))
        mod = {k: float(np.mean([p[k][0] for p in per])) for k in names}
        frac = {k: float(np.mean([p[k][1] for p in per])) for k in names}
        out[mode] = dict(sim=sim, mod=mod, frac=frac)
        print(f"  [{mode}] done ({time.time()-t0:.0f}s)  nocut: R_sim={sim['__nocut__']:+.5f} "
              f"R_model={mod['__nocut__']:+.5f}", flush=True)

    # ---- correctness check on the intrinsic mode -------------------------------------------
    si = out["intrinsic"]["sim"]["__nocut__"]; mi = out["intrinsic"]["mod"]["__nocut__"]
    dev = abs(mi / si - 1.0) if si else np.nan
    print("\n" + "=" * 100)
    print("INTRINSIC-MODE CORRECTNESS CHECK (must be ~0: same shapes, same objects, no cut)")
    print(f"  R_sim(nocut)={si:+.8f}   R_model(nocut)={mi:+.8f}   rel dev = {dev*100:.5f}%")
    ok = np.isfinite(dev) and dev < 1e-4
    print("  -> PASS: intrinsic mode is wired correctly; the column below is meaningful."
          if ok else
          "  -> **FAIL**: intrinsic no-cut m is not zero. The mode is MIS-WIRED; ignore the table.")

    # ---- attribution table -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ATTRIBUTION -- m = R_sim/R_model - 1  (ISOLATED)")
    print("=" * 100)
    print(f"  {'cut':>10} {'fracM':>6} | {'m_sel':>8} {'m_flow':>8} | {'m_measured':>11} {'difference':>11}")
    print(f"  {'':>10} {'':>6} | {'REAL sel':>8} {'model':>8} | {'end-to-end':>11} {'shape+cross':>11}")
    rows = []
    a, b = out["measured"], out["intrinsic"]
    nc = "__nocut__"
    for k in names:
        m_meas = a["sim"][k] / a["mod"][k] - 1.0 if a["mod"][k] else np.nan
        m_intr = b["sim"][k] / b["mod"][k] - 1.0 if b["mod"][k] else np.nan
        # m_sel: how much selection bias EXISTS in the sim, with exact shapes (a DATA property).
        # m_flow: whether the flow reproduces that selected response (a MODEL property).
        # Reporting m_flow ALONE is misleading -- a small m_flow where m_sel ~ 0 means "nothing was
        # being tested", not "the model works". See WORKLOG 2026-07-30j.
        m_sel = b["sim"][k] / b["sim"][nc] - 1.0 if b["sim"][nc] else np.nan
        m_flow = b["mod"][k] / b["sim"][k] - 1.0 if b["sim"][k] else np.nan
        print(f"  {k:>10} {a['frac'][k]:>6.2f} | {m_sel*100:>+7.2f}% {m_flow*100:>+7.2f}% | "
              f"{m_meas*100:>+10.2f}% {(m_meas-m_intr)*100:>+10.2f}%")
        rows.append((k, m_meas, m_intr))
    print("\n  m_sel  = R_sim(cut)/R_sim(nocut)-1 with EXACT shapes: the selection bias that really")
    print("           exists in the sim. If this is ~0, that cut has NO selection bias to test.")
    print("  m_flow = R_flow(cut)/R_sim(cut)-1: whether the flow reproduces it. Judge the model on")
    print("           this ONLY where m_sel is non-negligible.")
    print("  The last column is shape-response error PLUS the shape-selection cross term -- it is")
    print("  NOT 'the shape term'.")

    if args.output:
        np.savez(args.output, names=np.array(names),
                 m_measured=np.array([r[1] for r in rows]),
                 m_intrinsic=np.array([r[2] for r in rows]),
                 **{f"{m}_{w}": np.array([out[m][w][k] for k in names])
                    for m in ("measured", "intrinsic") for w in ("sim", "mod", "frac")})
        print(f"\nsaved {args.output}")
    print("SELECTION_ATTRIBUTION_DONE", flush=True)


if __name__ == "__main__":
    main()
