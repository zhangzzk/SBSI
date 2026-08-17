"""One pair list, BOTH models: the confound-free `R_blend` comparison on constgold.

WHY. Every flow-#2-vs-BlendEMU constgold number so far has mixed a model difference with a
bookkeeping difference (WORKLOG 2026-08-02e). `build_blend_lookup.py` hands the whole field to
`predict_response`, which applies the emulator's own stored neighbour selection -- 18 <= mag_s < 26,
0.3 <= Re_s <= 1.5, r_max 10", k 20 -- while `build_blend_lookup_flow.py` sums a 7" KD-tree. Two
different pair sets, so the difference in `<R_blend>` is not attributable:

    lookup level : flow native 0.2388, flow matched 0.1936, emulator 0.1358
    scored rows  : flow native 0.1551 (+14.2% vs emu), flow matched 0.1280 (-22.3% vs emu)

The flow sits ABOVE the emulator on one construction and BELOW it on another. That spread is the
confound, not a result.

WHAT THIS DOES. Builds ONE pair list per case and scores BOTH models on it, pair for pair:
  * flow #2 via its neighbour-shear derivative (as `build_blend_lookup_flow.py`);
  * BlendEMU via `predict_on_pairs`, the pair-level API that scores exactly the rows it is handed
    instead of re-deriving its own neighbour list.
Then each is summed per primary. The two summed columns differ ONLY by the model, so `m` computed
from them is finally a model comparison.

APERTURE -- WHY 7" AND NOT 10". The emulator natively sums to 10", the flow trained on the ap7 legs
and has never seen a pair beyond 7". Matching at 10" would extrapolate the flow in separation;
matching at 7" merely evaluates the emulator on a subset of the range it covers, which is the
direction that costs nothing. So the emulator is brought DOWN to the flow's aperture, not the reverse.
The 7-10" annulus is then absent from both, and equally so -- it is excluded from the comparison,
not silently attributed to either model.

DOMAIN. `--restrict-to-emu-domain` (default on) keeps only pairs inside the emulator's stored
regression cuts, so neither model is extrapolated on the neighbour axis: the emulator is inside its
training box by construction, and the flow -- trained to mag 29 / Re 0.01 -- is comfortably inside
its own. Turning it off scores both models on the full 7" pair list, which is the wider-population
question; then the emulator IS extrapolating and its column must be read as such. Either way the two
columns cover the same pairs.

FIREWALL: reads constgold INPUT (true properties/positions) to SCORE two already-trained models.
Nothing is fitted, tuned or selected. constgold remains evaluation-only, and no choice between the
models may be made from the `m` this feeds -- that is settled on the per-pair ruler
(`scripts/eval_faint_neighbour_contribution.py`, `scripts/eval_blend_flow.py`).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.blend_flow import (  # noqa: E402
    build_features, load_blend_flow, response_from_contexts, shifted_shape_columns)
from build_blend_lookup_flow import CBASE, TILE, input_feather, pairs_within, select_neighbours  # noqa: E402

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)

# The emulator's stored regression selection, read back from its metadata at run time rather than
# retyped here -- a constant pasted from another file is exactly how the two lookups drifted apart.
EMU_FEATURES = ["Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
                "sersic_n_input_p", "sersic_n_input_s", "distance"]


def emulator_selection(tag):
    """(mag_s range, Re_s range, r_max, k) from the emulator's own metadata sidecar."""
    import json
    p = f"{BLEND_MODELS}/emulator_metadata_{tag}.json"
    with open(p) as fh:
        meta = json.load(fh)
    reg = meta["tasks"]["regression"]
    cuts = reg["cuts"]          # [mag_p, mag_s, Re_p, Re_s, distance]
    return tuple(cuts[1]), tuple(cuts[3]), float(reg["r_max"]), int(reg["k"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True, nargs="+",
                    help="one or more flow-#2 checkpoints. With several, EVERY seed is scored on the "
                         "SAME pair list in ONE pass -- the catalogue reads, pair construction and "
                         "the emulator pass happen once -- and the output carries one "
                         "`R_blend_flow_s<seed>` column per seed plus their mean as `R_blend_flow`. "
                         "Per-seed columns exist because AGENTS.md requires `m` to be formed INSIDE "
                         "each seed and the spread taken across seeds; building it from the ensemble "
                         "mean and propagating one term's scatter is the documented error.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default=CBASE)
    ap.add_argument("--aperture", type=float, default=7.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_indom_tuned")
    ap.add_argument("--batch-size", type=int, default=500000)
    ap.add_argument("--restrict-to-emu-domain", dest="restrict", action="store_true", default=True)
    ap.add_argument("--no-restrict-to-emu-domain", dest="restrict", action="store_false")
    ap.add_argument("--emu-device", default="cpu", choices=["cpu", "cuda"],
                    help="XGBoost inference device. cpu is the safe default -- cuda shares the GPU "
                         "with the flow's torch allocator and can OOM on a vGPU slice.")
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    for ck in args.checkpoint:
        model, std, yscale, meta = load_blend_flow(ck, device=dev)
        if bool(meta.get("crowding", False)):
            raise SystemExit(
                f"REFUSING: {ck} was trained with the CROWDING block "
                "(nbr_flux_near/far/max, log_k), which is built from a primary's FULL neighbour list "
                "via `pid`/`k`. The frames assembled here are per-case pair lists that do not carry "
                "those columns, so the features cannot be reproduced -- and building them from "
                "whatever rows happen to be present would report a systematically under-crowded "
                "galaxy. Use a non-crowding checkpoint, or extend this script to carry `pid`/`k`.")
        members.append(dict(path=ck, model=model, std=std, yscale=yscale, meta=meta,
                            seed=int(meta.get("seed", -1))))

    # Everything that shapes the PAIR LIST or the response CONVENTION must agree across members, or
    # the per-seed columns would not be the same quantity and averaging them would be meaningless.
    ref = members[0]["meta"]
    keys = ("delta", "derived", "shape_indices", "primary_mag_max", "primary_re_min")
    for mm in members[1:]:
        bad = [k for k in keys if mm["meta"].get(k) != ref.get(k)]
        if bad:
            raise SystemExit(
                f"REFUSING: {mm['path']} disagrees with {members[0]['path']} on {bad}. These control "
                "the finite-difference convention and the training domain, so the seeds would not be "
                "measuring the same thing and their mean would not be an ensemble of anything.")
    seeds = [mm["seed"] for mm in members]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"REFUSING: duplicate seeds in {seeds} -- the per-seed columns would "
                         "collide and the ensemble mean would double-count a member.")
    meta = ref
    delta = float(meta["delta"])
    derived = bool(meta.get("derived", True))
    i0, i1 = meta["shape_indices"]
    dom_mag = float(meta.get("primary_mag_max", 26.0))
    dom_re = float(meta.get("primary_re_min", 0.3))

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred_emu = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND,
                                      device=args.emu_device, load_self=False)
    mag_rng, re_rng, emu_rmax, emu_k = emulator_selection(args.tag)

    print(f"flow #2   : {len(members)} checkpoint(s), seeds {seeds}")
    for mm in members:
        print(f"    {mm['path']}")
    print(f"  delta={delta} derived={derived} device={dev}")
    print(f"  primary training domain: mag < {dom_mag}, Re > {dom_re}")
    print(f"BlendEMU  : tag {args.tag}")
    print(f"  its own neighbour selection: mag_s {mag_rng}, Re_s {re_rng}, "
          f"r_max {emu_rmax}\", k {emu_k}")
    print(f"aperture  : {args.aperture}\" for BOTH models "
          f"(the flow has never seen a pair beyond 7\"; bringing the emulator down costs nothing, "
          f"extrapolating the flow up would not)")
    print(f"pair set  : {'emulator domain (mag_s/Re_s/k applied to both)' if args.restrict else 'ALL pairs in aperture -- the emulator is EXTRAPOLATED outside its cuts'}")

    parts, tot_rows, tot_in = [], 0, 0
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True)
            continue
        t = pf.read_table(fp).to_pandas()
        ra, dec = t["RA_input"].to_numpy(float), t["DEC_input"].to_numpy(float)
        i, j, d = pairs_within(ra, dec, args.aperture)
        n_raw = len(i)
        if args.restrict:
            i, j, d, _ = select_neighbours(i, j, d, t["r_input"].to_numpy(float),
                                           t["Re_input"].to_numpy(float),
                                           mag_rng, re_rng, emu_k)
        if not len(i):
            print(f"case{c}: no pairs survive", flush=True)
            continue

        frame = pd.DataFrame({
            "r_input_p": t["r_input"].to_numpy(float)[i],
            "Re_input_p": t["Re_input"].to_numpy(float)[i],
            "sersic_n_input_p": t["sersic_n_input"].to_numpy(float)[i],
            "e1_input_rot0_p": t["e1_input_rot0"].to_numpy(float)[i],
            "e2_input_rot0_p": t["e2_input_rot0"].to_numpy(float)[i],
            "r_input_s": t["r_input"].to_numpy(float)[j],
            "Re_input_s": t["Re_input"].to_numpy(float)[j],
            "sersic_n_input_s": t["sersic_n_input"].to_numpy(float)[j],
            "e1_input_rot0_s": t["e1_input_rot0"].to_numpy(float)[j],
            "e2_input_rot0_s": t["e2_input_rot0"].to_numpy(float)[j],
            "distance": d})

        # ---- flow #2, per pair, ONE ROW PER SEED ----
        # The raw design matrix and the shifted shapes depend only on the frame and on `delta`, which
        # every member shares (checked above), so they are built ONCE and only the standardization
        # and the forward pass repeat per seed.
        Xraw = build_features(frame, derived=derived)
        sh = shifted_shape_columns(frame, delta)
        r_flow_seed = np.empty((len(members), len(frame)), dtype=np.float64)
        for mi, mm in enumerate(members):
            std, model, yscale = mm["std"], mm["model"], mm["yscale"]
            X = std.transform(Xraw)
            m0, s0 = std.mean[i0], std.scale[i0]
            m1, s1 = std.mean[i1], std.scale[i1]
            with torch.no_grad():
                for s in range(0, len(frame), args.batch_size):
                    e = min(s + args.batch_size, len(frame))
                    c0 = torch.as_tensor(X[s:e]).to(dev)
                    ctxs = {}
                    for k in ("e1+", "e1-", "e2+", "e2-"):
                        cc = c0.clone()
                        cc[:, i0] = torch.as_tensor(
                            ((sh[k][0][s:e] - m0) / s0).astype(np.float32)).to(dev)
                        cc[:, i1] = torch.as_tensor(
                            ((sh[k][1][s:e] - m1) / s1).astype(np.float32)).to(dev)
                        ctxs[k] = cc
                    r_flow_seed[mi, s:e] = response_from_contexts(
                        model, c0, ctxs, (yscale[0], yscale[1]), delta).cpu().numpy()
        r_flow = r_flow_seed.mean(axis=0)

        # ---- BlendEMU, the SAME pairs ----
        r_emu = np.empty(len(frame), dtype=np.float64)
        for s in range(0, len(frame), args.batch_size):
            e = min(s + args.batch_size, len(frame))
            got = pred_emu.predict_on_pairs(frame.iloc[s:e][EMU_FEATURES].copy(), task="response",
                                            warn_extrapolation=(s == 0))
            r_emu[s:e] = got["response"].to_numpy(float)

        npr = len(t)
        rb_f = np.bincount(i, weights=r_flow, minlength=npr)
        rb_e = np.bincount(i, weights=r_emu, minlength=npr)
        nk = np.bincount(i, minlength=npr)
        indom = (t["r_input"].to_numpy(float) < dom_mag) & (t["Re_input"].to_numpy(float) > dom_re)
        keep = nk > 0
        block = pd.DataFrame({
            "case": c, "input_index": t["index_input"].to_numpy(np.int64)[keep],
            "R_blend_flow": rb_f[keep], "R_blend_emu": rb_e[keep],
            "n_neighbours": nk[keep], "in_domain": indom[keep]})
        # Per-seed SUMS, so `m` can be formed inside each seed downstream. Summing per seed and then
        # averaging is not the same as averaging per pair and then summing only if the pair sets
        # differed between seeds -- they do not here -- but the per-seed columns are needed anyway to
        # get the SPREAD, which is the whole reason the ensemble exists.
        if len(members) > 1:
            for mi, mm in enumerate(members):
                block[f"R_blend_flow_s{mm['seed']}"] = np.bincount(
                    i, weights=r_flow_seed[mi], minlength=npr)[keep]
        parts.append(block)
        tot_rows += int(keep.sum())
        tot_in += int((keep & indom).sum())
        print(f"case{c}: {int(keep.sum()):,} primaries, {len(i):,} pairs of {n_raw:,} in aperture, "
              f"<k>={nk[keep].mean():.2f} | flow {rb_f[keep].mean():.4f}  "
              f"emu {rb_e[keep].mean():.4f}", flush=True)

    if not parts:
        raise SystemExit("REFUSING: no cases produced pairs")
    out = pd.concat(parts, ignore_index=True)
    dm = out["in_domain"].to_numpy(bool)
    print(f"\ntotal {tot_rows:,} primaries; {100*tot_in/max(tot_rows,1):.2f}% inside the flow's "
          f"primary training domain")
    print(f"  {'':<22}{'all rows':>12}{'in-domain':>12}")
    for nm, col in (("<R_blend> flow #2", "R_blend_flow"), ("<R_blend> BlendEMU", "R_blend_emu")):
        print(f"  {nm:<22}{out[col].mean():>12.4f}{out.loc[dm, col].mean():>12.4f}")
    rat_all = out["R_blend_flow"].mean() / max(out["R_blend_emu"].mean(), 1e-12)
    rat_in = out.loc[dm, "R_blend_flow"].mean() / max(out.loc[dm, "R_blend_emu"].mean(), 1e-12)
    print(f"  {'flow / emulator':<22}{rat_all:>12.3f}{rat_in:>12.3f}")
    print("\n  Both columns are sums over the SAME pairs, so this ratio is the model difference with")
    print("  no aperture, k-cap or neighbour-selection confound left in it.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_feather(args.output)
    print(f"\nwrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases "
          f"({time.time()-t0:.0f}s)")
    print("BLEND_LOOKUP_BOTH_DONE", flush=True)


if __name__ == "__main__":
    main()
