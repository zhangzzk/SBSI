"""Can the flow predict an S/N-cut selection bias? Proxy S/N on BOTH sides, so it is self-consistent.

THE PROPOSAL (owner). The flow outputs measured shape, magnitude and size; S/N is (nearly) a function
of measured magnitude and size; so the flow can predict the selection probability of an S/N cut for
every true galaxy, without ever being trained on S/N.

WHAT THE GATE ALREADY ESTABLISHED (`eval_sn_proxy_check.py`, job 15364971):
  * `log10 S/N = a*mag + b*log10(R) + c` fits the REAL SExtractor S/N with R^2 = 0.9944
    (a = -0.3676, b = -0.7736), so mag+size really do determine S/N; BUT
  * cutting on that proxy does NOT reproduce the selection bias of cutting on the real S/N -- it
    OVERSTATES it by 0.25-0.52 points at every keep-fraction, because the 0.6% of variance the proxy
    misses is preferentially the SHEAR-CORRELATED part, and that is the only part selection bias
    depends on.

SO WHAT THIS SCRIPT DOES AND DOES NOT TEST. It applies the SAME proxy on both sides -- sim cuts on
the proxy built from the CATALOGUE's measured mag/size, model cuts on the proxy built from the FLOW's
SAMPLED measured mag/size. That is self-consistent, and it answers:

    "does the flow reproduce a cut it can actually represent?"        <- YES, this is tested here

It does NOT answer "does the flow reproduce a real S/N cut". The ~0.4-point proxy-vs-real gap from
the gate sits ON TOP of whatever m_flow comes out below, and closing it would need the flow to output
flux_auto and fluxerr_auto as extra dimensions (a retrain, not post-processing).

DETECTION IS SEPARATED, as requested: `build_base` keeps only objects detected in BOTH legs.

VALIDATION BUILT IN: the same run also reports a plain measured-size cut, whose m_sel must reproduce
the already-published dense-grid numbers (e.g. +5.40% at R>0.70"). That is the check that the new
`lin` cut path did not disturb the existing one.

FIREWALL: half-shear legs, ISOLATED. No emulator, no constgold.
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
import eval_selection_response as ESR  # noqa: E402
from eval_selection_response import (  # noqa: E402
    CAT, CROWD, NN, model_selected_response, truth_selected_response,
)

PX = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--sn-a", type=float, default=-0.3676, help="fitted mag coefficient")
    ap.add_argument("--sn-b", type=float, default=-0.7736, help="fitted log10(R) coefficient")
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.95, 0.85, 0.70, 0.50, 0.30])
    ap.add_argument("--size-check", type=float, nargs="+", default=[3.5],
                    help="plain size cuts (px) rerun as a regression check on the untouched path")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ESR.MEAS = list(dict.fromkeys(ESR.MEAS))
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]
    print(f"device={device}  ckpts={len(args.ckpt)}  n_samples={args.n_samples}  "
          f"proxy: log10 S/N = {args.sn_a:+.4f}*mag {args.sn_b:+.4f}*log10(R)", flush=True)

    # Thresholds on the proxy, placed at the requested keep-fractions of the g=0 leg.
    m0 = base["measured_mag_auto_0"].to_numpy(float)
    r0 = base["measured_flux_radius_0"].to_numpy(float)
    u0 = args.sn_a * m0 + args.sn_b * np.log10(np.maximum(r0, 1e-6))
    ok = iso & np.isfinite(u0)
    cuts = []
    for kf in args.keep_fracs:
        thr = float(np.quantile(u0[ok], 1.0 - kf))
        cuts.append(dict(name=f"SN keep{kf:.2f}", dim=None, thr=thr, keep_high=True,
                         raw=thr, kind="sn", lin=(args.sn_a, args.sn_b)))
    for c in args.size_check:
        cuts.append(dict(name=f'R>{c*PX:.2f}"', dim=3, thr=float(np.log(c)), keep_high=True,
                         raw=c, kind="size", lin=None))
    names = ["__nocut__"] + [c["name"] for c in cuts]

    # ---- truth, EXACT intrinsic shapes -> pure selection --------------------------------------
    sim = {}
    sim["__nocut__"], _, _, _ = truth_selected_response(
        base, gh1, gh2, gmed, iso, "measured_flux_radius", -1e9, True, intrinsic=True)
    for c in cuts:
        xcol = (("lin", args.sn_a, args.sn_b) if c["kind"] == "sn" else
                ("measured_flux_radius" if c["kind"] == "size" else "measured_mag_auto"))
        sim[c["name"]], _, _, _ = truth_selected_response(
            base, gh1, gh2, gmed, iso, xcol, c["raw"], c["keep_high"], intrinsic=True)

    # ---- model ---------------------------------------------------------------------------------
    per = []
    for ck in args.ckpt:
        b = load_measurement_model(ck, device=device)
        per.append(model_selected_response(b, base, gh1, gh2, gmed, iso, cuts, args.n_samples,
                                           args.batch_size, args.flow_seed, device,
                                           intrinsic=True))
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
    mod = {k: float(np.mean([p[k][0] for p in per])) for k in names}
    sem = {k: (float(np.std([p[k][0] for p in per], ddof=1) / np.sqrt(len(per)))
               if len(per) > 1 else np.nan) for k in names}

    nc = "__nocut__"
    print("\n" + "=" * 100)
    print("PROXY-S/N CUT: does the flow reproduce a cut it CAN represent?  (ISOLATED, both-detected)")
    print("=" * 100)
    print(f"  {'cut':>14} | {'m_sel [%]':>12} {'m_flow [%]':>16}")
    for k in names:
        if k == nc:
            continue
        m_sel = (sim[k] / sim[nc] - 1.0) * 100.0
        m_flow = (mod[k] / sim[k] - 1.0) * 100.0
        e = sem[k] / abs(sim[k]) * 100.0 if np.isfinite(sem[k]) else np.nan
        print(f"  {k:>14} | {m_sel:>+11.3f}% {m_flow:>+11.3f} +- {e:<.3f}")
    print("\n  REGRESSION CHECK: the plain size row must reproduce the published dense-grid m_sel")
    print("  (R>0.70\" -> +5.40%). If it does not, the new `lin` cut path broke the existing one.")
    print("\n  SCOPE: this is the PROXY cut on both sides. The gate (job 15364971) showed the proxy")
    print("  overstates the REAL-S/N selection bias by 0.25-0.52 pts; that gap is NOT included here.")
    print("SN_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
