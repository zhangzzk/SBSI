"""Per-object SELF-response on the half-shear legs: sim vs flow. Feeds figure 5.

WHY A SEPARATE DUMP. Figure 2 compares the FULL response on constgold, so its model curve is
`R_flow + R_blend`. Figure 5 asks the narrower question -- is the flow's OWN (self) response right? --
and that needs a regime where the self term is isolated. constgold cannot do it: every object's
response there contains its neighbours' contribution. The half-shear legs can, because each galaxy
carries its own RANDOM shear direction, so projecting the leg-to-leg shape difference on the PRIMARY's
direction ghat_p keeps the primary's own response and averages the neighbours' away (their ghat_s is
uncorrelated with ghat_p). That is the standard component-separation ruler for this project.

    R_self = < (e_g - e_0) . ghat_p > / g          measured ngmix shapes, both legs

EXTRACTION MUST MATCH ON BOTH SIDES -- this is a recorded trap, not a theoretical worry. A previous
Stage-1 investigation chased a 0.49-vs-0.60 self-response gap that turned out to be pure extraction
convention: antithetic (+/-g) against forward (0 -> +g) on image-identical sims. The half-shear sim
here is FORWARD (leg 0 is g=0, leg g is g=0.05), so the flow is scored FORWARD too -- legs s=0 and
s=+gmed, never +/-gmed. Changing this silently reintroduces that gap.

COMMON RANDOM NUMBERS: both legs are reseeded identically, so flow sampling noise cancels in the
difference instead of being amplified by 1/g ~ 20x.

Output: one feather with per-object `r_sim_self`, `R_flow_s{seed}` for every seed, and the three
figure-2 axes (primary S/N, primary true Re, neighbour flux). No model is trained; no constgold is
touched.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
import eval_selection_response as ESR  # noqa: E402
from eval_selection_response import CAT, CROWD, NN  # noqa: E402

RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
EXTRA = ["measured_flux_auto", "measured_fluxerr_auto"]


@torch.no_grad()
def flow_self_response(bundle, base, gh1, gh2, gmed, n_samples, batch_size, seed, chunk=100_000):
    """Per-object R_flow, FORWARD extraction (s=0 -> s=+gmed), CRN across legs."""
    names = bundle.target_transform.target_names
    i1, i2 = names.index("measured_ngmix_g1"), names.index("measured_ngmix_g2")
    i1i = base["e1_input_rot0_p"].to_numpy(float)
    i2i = base["e2_input_rot0_p"].to_numpy(float)
    n = len(base)
    out = np.empty(n, dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        means = {}
        for leg, s in ((0, 0.0), (1, +gmed)):
            fr = base.iloc[lo:hi].reset_index(drop=True).copy()
            e1s, e2s = apply_shear_to_ellipticity(i1i[lo:hi], i2i[lo:hi],
                                                  s * gh1[lo:hi], s * gh2[lo:hi])
            fr["e1_input_rot0_p"], fr["e2_input_rot0_p"] = e1s, e2s
            fr = rescale(fr, **RK)
            torch.manual_seed(seed)                     # CRN: identical latents in both legs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)
            proj = d[:, :, i1] * gh1[lo:hi, None] + d[:, :, i2] * gh2[lo:hi, None]
            means[leg] = np.nanmean(proj, axis=1)
        out[lo:hi] = ((means[1] - means[0]) / gmed).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--out", default="results/halfshear_selfresp.feather")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ESR.MEAS = list(dict.fromkeys(ESR.MEAS + EXTRA))
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"]
    print(f"device={device}  ckpts={len(args.ckpt)}  N={len(base):,}  g={gmed:.4f}", flush=True)

    # ---- sim self-response: FORWARD, projected on the PRIMARY's shear direction ---------------
    e1_0 = base["measured_ngmix_g1_0"].to_numpy(float)
    e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    e1_g = base["measured_ngmix_g1_g"].to_numpy(float)
    e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    p0 = e1_0 * gh1 + e2_0 * gh2
    pg = e1_g * gh1 + e2_g * gh2
    r_sim_self = (pg - p0) / gmed

    # ---- axes, matching figure 2 ---------------------------------------------------------------
    f0 = base["measured_flux_auto_0"].to_numpy(float)
    fe0 = base["measured_fluxerr_auto_0"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = np.where(fe0 > 0, f0 / fe0, np.nan)

    out = pd.DataFrame({
        "case": base["case"].to_numpy(),
        "input_index": base["input_index"].to_numpy(),
        "r_sim_self": r_sim_self.astype(np.float32),
        "SN": sn.astype(np.float32),
        "Re_input_p": base["Re_input_p"].to_numpy(np.float32),
        "nbr_flux_near": base["nbr_flux_near"].to_numpy(np.float32),
    })
    print(f"  sim <R_self> = {np.nanmean(r_sim_self):+.5f}  ({time.time()-t0:.0f}s)", flush=True)

    seen = set()
    for ck in args.ckpt:
        # Parse the seed with an anchored regex. Splitting on "_s" also matches "_swaavg", which
        # made every checkpoint resolve to the same label, so all 16 wrote one column and
        # overwrote each other -- the dump silently held only the last seed.
        mo = re.search(r"_s(\d+)_", os.path.basename(ck))
        if not mo:
            raise SystemExit(f"cannot parse a seed from {os.path.basename(ck)}")
        sd = mo.group(1)
        if sd in seen:
            raise SystemExit(f"duplicate seed label {sd!r} -- columns would overwrite")
        seen.add(sd)
        b = load_measurement_model(ck, device=device)
        rf = flow_self_response(b, base, gh1, gh2, gmed, args.n_samples, args.batch_size,
                               args.flow_seed)
        out[f"R_flow_s{sd}"] = rf
        print(f"  s{sd}: <R_flow>={np.nanmean(rf):+.5f}  ({time.time()-t0:.0f}s)", flush=True)

    cols = [c for c in out.columns if c.startswith("R_flow_s")]
    if len(cols) != len(args.ckpt):
        raise SystemExit(f"REFUSING to save: {len(cols)} seed columns for {len(args.ckpt)} "
                         "checkpoints -- seed labels collided.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_feather(args.out)
    print(f"\nsaved {len(out):,} rows x {len(cols)} seeds -> {args.out}")
    print(f"  ensemble <R_flow> = {np.nanmean([out[c].mean() for c in cols]):+.5f}  "
          f"vs sim <R_self> = {np.nanmean(r_sim_self):+.5f}")
    print("HALFSHEAR_SELFRESP_DONE", flush=True)


if __name__ == "__main__":
    main()
