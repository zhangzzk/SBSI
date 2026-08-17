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

Output: one feather with per-object `r_sim_self`, `R_flow_s{seed}` for every seed, and the
diagnostic axes (primary S/N, primary true magnitude/Re, neighbour flux). No model is trained; no
constgold is touched.
"""
from __future__ import annotations

import argparse
import json
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
def flow_self_response(bundle, base, gh1, gh2, gmed, n_samples, batch_size, seed, chunk=100_000,
                       extraction="forward", ext_delta=None):
    """Per-object R_flow, CRN across legs.

    extraction="forward" (default, unchanged): the chord s=0 -> s=+gmed, matching how the SIM truth
    `r_sim_self` and the ruler's SNC label are built.

    extraction="central": the symmetric slope (s=-d -> s=+d)/2d at `ext_delta`. This exists to test a
    TRAIN/SCORE MISMATCH. The response pin supervises the mean head's CENTRAL difference at
    --response-delta (0.02 fiducial), while this dump grades the FORWARD chord over 0->0.05. If the
    response is nonlinear in g those are different functionals, so the pin can be satisfied exactly
    while the graded number still misses -- which would explain why the per-object pin is insensitive
    to its weight over 100..8000 (WORKLOG 2026-08-03p/r). Scoring central-extracted model response
    against the forward-extracted sim truth is deliberately the SAME comparison training tried to
    equate, so agreement there localises the deficit to EXTRACTION rather than to the fit.
    """
    names = bundle.target_transform.target_names
    i1, i2 = names.index("measured_ngmix_g1"), names.index("measured_ngmix_g2")
    i1i = base["e1_input_rot0_p"].to_numpy(float)
    i2i = base["e2_input_rot0_p"].to_numpy(float)
    n = len(base)
    out = np.empty(n, dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        means = {}
        d_ext = float(ext_delta if ext_delta is not None else gmed)
        legs = ((0, 0.0), (1, +gmed)) if extraction == "forward" else ((0, -d_ext), (1, +d_ext))
        denom = gmed if extraction == "forward" else 2.0 * d_ext
        for leg, s in legs:
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
        out[lo:hi] = ((means[1] - means[0]) / denom).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    # Optional LOWER case bound. Default None == the original behaviour (cases 0..max-case).
    # Given, the dump covers a case WINDOW, which is what lets the 160 fresh cases be fanned out
    # over a Slurm array by case chunk instead of by checkpoint: each task then pays the (CPU,
    # IO-bound) population build for its own chunk only, while the GPU work -- 16 checkpoints on
    # 1/N of the rows -- is unchanged in total.
    ap.add_argument("--min-case", type=int, default=None)
    # BLIND MODE. The dump normally prints <R_self> and <R_flow> as a sanity read-out. When the
    # dump is being produced as FRESH data for a pre-registered gate, printing any response --
    # even a global mean -- puts a number in a log that a later analysis choice could be tuned
    # against. --blind replaces every response print with counts and finite fractions only.
    # It changes nothing that is written to the feather.
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--extraction", choices=["forward", "central"], default="forward",
                    help="how the MODEL's response is extracted. forward = the 0->gmed chord "
                         "(default, matches the sim truth). central = the symmetric slope at "
                         "--extraction-delta, matching what the response PIN actually supervises.")
    ap.add_argument("--extraction-delta", type=float, default=None,
                    help="half-width for --extraction central; defaults to gmed. Set 0.02 to match "
                         "the trainer's --response-delta.")
    ap.add_argument("--out", default="results/halfshear_selfresp.feather")
    # BASE CACHE. Building the matched both-detected population is pure CPU and IO-bound: it reads
    # the two ~72 GB half-shear legs. Scoring is GPU and linear in checkpoints. Fanning the 16
    # checkpoints out over a Slurm array WITHOUT a cache would re-read ~115 GB sixteen times.
    # --build-base-only runs the build once and writes the population (plus the per-object shear
    # directions and gmed) to --base-cache; the array tasks then pass --base-cache and skip the
    # read entirely. The cached frame IS the built frame -- nothing is recomputed from it.
    ap.add_argument("--base-cache", default=None)
    ap.add_argument("--build-base-only", action="store_true")
    # Build the population in CASE CHUNKS and concatenate. Every cut is per-object and the legs are
    # matched WITHIN a case, so chunking is exact; the only non-per-object quantity is gmed, which
    # is asserted identical across chunks rather than assumed. Purely a peak-memory control.
    ap.add_argument("--case-chunk", type=int, default=0)
    args = ap.parse_args()

    if not args.build_base_only and not args.ckpt:
        raise SystemExit("--ckpt is required unless --build-base-only is given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ESR.MEAS = list(dict.fromkeys(ESR.MEAS + EXTRA))

    cache_meta = None if args.base_cache is None else os.path.splitext(args.base_cache)[0] + ".json"
    if args.base_cache and os.path.exists(args.base_cache) and not args.build_base_only:
        base = pd.read_feather(args.base_cache)
        with open(cache_meta) as fh:
            cm = json.load(fh)
        gmed = float(cm["gmed"])
        gh1 = base.pop("_ghat1").to_numpy(float)
        gh2 = base.pop("_ghat2").to_numpy(float)
        print(f"loaded base cache {args.base_cache}: N={len(base):,}  g={gmed:.4f}  "
              f"cases=[{cm['case_lo']},{cm['case_hi']}]  ({time.time()-t0:.0f}s)", flush=True)
    else:
        lo0 = 0 if args.min_case is None else args.min_case
        step = args.case_chunk if args.case_chunk and args.case_chunk > 0 else (args.max_case - lo0 + 1)
        parts, gmeds = [], []
        for clo in range(lo0, args.max_case + 1, step):
            chi = min(clo + step - 1, args.max_case)
            ru = ESR.build_base(args.g0_leg, args.gS_leg, chi, args.true_re_min,
                                args.true_mag_max, args.iso_radius, CROWD, NN, t0,
                                min_case=(clo if args.min_case is not None else None))
            b = ru["base"]
            b["_ghat1"] = ru["gh1"]
            b["_ghat2"] = ru["gh2"]
            parts.append(b)
            gmeds.append(float(ru["gmed"]))
            print(f"  chunk cases [{clo},{chi}]: N={len(b):,}  gmed={gmeds[-1]:.6f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if max(gmeds) - min(gmeds) > 1e-9:
            raise SystemExit(f"REFUSING: gmed differs across case chunks {gmeds} -- the chunks are "
                             "not combinable, the shear magnitude is not constant.")
        gmed = gmeds[0]
        base = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        del parts
        gh1 = base.pop("_ghat1").to_numpy(float)
        gh2 = base.pop("_ghat2").to_numpy(float)
        if args.base_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.base_cache)), exist_ok=True)
            sav = base.copy()
            sav["_ghat1"] = gh1
            sav["_ghat2"] = gh2
            sav.reset_index(drop=True).to_feather(args.base_cache)
            with open(cache_meta, "w") as fh:
                json.dump(dict(n_rows=int(len(sav)), gmed=gmed,
                               case_lo=int(sav["case"].min()), case_hi=int(sav["case"].max()),
                               n_cases=int(sav["case"].nunique()),
                               min_case=args.min_case, max_case=args.max_case,
                               g0_leg=args.g0_leg, gS_leg=args.gS_leg,
                               true_re_min=args.true_re_min, true_mag_max=args.true_mag_max),
                          fh, indent=1, sort_keys=True)
            print(f"wrote base cache -> {args.base_cache}  ({time.time()-t0:.0f}s)", flush=True)
            del sav
        if args.build_base_only:
            print(f"BASE_CACHE_DONE  N={len(base):,}  cases=[{int(base['case'].min())},"
                  f"{int(base['case'].max())}]  n_cases={base['case'].nunique()}", flush=True)
            return

    if args.min_case is not None and len(base):
        lo = int(base["case"].min())
        if lo < args.min_case:
            raise SystemExit(f"case window violated: min case {lo} < --min-case {args.min_case}")
    print(f"device={device}  ckpts={len(args.ckpt)}  N={len(base):,}  g={gmed:.4f}  "
          f"cases=[{int(base['case'].min())},{int(base['case'].max())}]", flush=True)

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
        "r_input_p": base["r_input_p"].to_numpy(np.float32),
        "Re_input_p": base["Re_input_p"].to_numpy(np.float32),
        "nbr_flux_near": base["nbr_flux_near"].to_numpy(np.float32),
    })
    if args.blind:
        print(f"  sim r_self: finite {np.isfinite(r_sim_self).mean():.6f}  "
              f"({time.time()-t0:.0f}s)  [BLIND: value withheld]", flush=True)
    else:
        print(f"  sim <R_self> = {np.nanmean(r_sim_self):+.5f}  ({time.time()-t0:.0f}s)", flush=True)

    seen = set()
    for ck in args.ckpt:
        # Parse the seed with an anchored regex. Splitting on "_s" also matches "_swaavg", which
        # made every checkpoint resolve to the same label, so all 16 wrote one column and
        # overwrote each other -- the dump silently held only the last seed.
        # `_s<seed>_` matches the SWA name (..._s501_swaavg.pt); `_s<seed>.pt` matches the RAW
        # final-epoch checkpoint. Both are needed to compare SWA against no-SWA, and the seed must
        # still never come from "_swaavg" itself -- that bug once made all 16 seeds write one column.
        mo = re.search(r"_s(\d+)(?:_|\.pt$)", os.path.basename(ck))
        if not mo:
            raise SystemExit(f"cannot parse a seed from {os.path.basename(ck)}")
        sd = mo.group(1)
        if sd in seen:
            raise SystemExit(f"duplicate seed label {sd!r} -- columns would overwrite")
        seen.add(sd)
        b = load_measurement_model(ck, device=device)
        rf = flow_self_response(b, base, gh1, gh2, gmed, args.n_samples, args.batch_size,
                               args.flow_seed, extraction=args.extraction,
                               ext_delta=args.extraction_delta)
        out[f"R_flow_s{sd}"] = rf
        if args.blind:
            print(f"  s{sd}: finite {np.isfinite(rf).mean():.6f}  ({time.time()-t0:.0f}s)  "
                  "[BLIND: value withheld]", flush=True)
        else:
            print(f"  s{sd}: <R_flow>={np.nanmean(rf):+.5f}  ({time.time()-t0:.0f}s)", flush=True)

    cols = [c for c in out.columns if c.startswith("R_flow_s")]
    if len(cols) != len(args.ckpt):
        raise SystemExit(f"REFUSING to save: {len(cols)} seed columns for {len(args.ckpt)} "
                         "checkpoints -- seed labels collided.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_feather(args.out)
    # Sidecar: everything the merge needs to refuse an inconsistent concatenation. Carries NO
    # response value -- gmed, counts, case window, seed labels and finite fractions only.
    meta = {
        "out": os.path.abspath(args.out),
        "g0_leg": args.g0_leg, "gS_leg": args.gS_leg,
        "min_case": args.min_case, "max_case": args.max_case,
        "case_lo": int(out["case"].min()), "case_hi": int(out["case"].max()),
        "n_cases": int(out["case"].nunique()),
        "n_rows": int(len(out)),
        "gmed": float(gmed),
        "true_re_min": args.true_re_min, "true_mag_max": args.true_mag_max,
        "iso_radius": args.iso_radius,
        "n_samples": args.n_samples, "flow_seed": args.flow_seed,
        "seeds": [c[len("R_flow_s"):] for c in cols],
        "ckpts": [os.path.basename(c) for c in args.ckpt],
        "finite_frac": {c: float(np.isfinite(out[c].to_numpy()).mean())
                        for c in ["r_sim_self"] + cols},
    }
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print(f"\nsaved {len(out):,} rows x {len(cols)} seeds -> {args.out}")
    if args.blind:
        print("  [BLIND] ensemble response means withheld")
    else:
        print(f"  ensemble <R_flow> = {np.nanmean([out[c].mean() for c in cols]):+.5f}  "
              f"vs sim <R_self> = {np.nanmean(r_sim_self):+.5f}")
    print("HALFSHEAR_SELFRESP_DONE", flush=True)


if __name__ == "__main__":
    main()
