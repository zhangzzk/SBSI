"""Apples-to-apples self-response gap: score V2 (SetConditionedForwardModel) AND V1-ladder
(ConditionalMeanFlow) checkpoints on the SAME ruler.

The ruler = base + truth R_hs + isolation mask, built ONCE by
scripts.eval_selfresp_gap.load_ruler (matched both-detected g0.05<->g0.0 acceptance objects,
true-cut Re>0.3 & r<26, R_hs from measured_ngmix_g1/g2, nn_bright>7" isolation). Every model is
scored against this identical base, so the ONLY difference between a V1-ladder number and a V2
number is the model's own response readout -- removing the confound that made the earlier V2
"-5.1%" (from eval_ens_halfshear.py's own nn-join/population) incomparable to the V1 ladder.

Model readouts (each model's NATIVE self-response, primary-only intrinsic shear):
  * V2  ckpt (has 'primary_preprocessor'): main-checkout loader + central-secant readout
        `eval_constgold_closure.load_model` / `run_model_on` (delta stored in ckpt = 0.05),
        i.e. EXACTLY the readout that produced the earlier V2 number.
  * V1  ckpt (has 'condition_preprocessor'): eval_selfresp_gap.model_selfresp (mean-head finite
        diff; --v1-difference/--v1-delta; forward/0.05 by default = what produced the ladder
        numbers). --v1-both-stencils also prints the central/0.05 readout so the stencil effect
        is visible and V2's central number is comparable.

Prints the identical flow/R_hs-1 OVERALL + by-true-size table for each model group.
FIREWALL: nothing trains; only scores existing ckpts against the det_meas half-shear truth.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

# --- import paths: worktree first (sbs_shear + eval_selfresp_gap), then MAIN checkout scripts
#     (eval_constgold_closure -> train_joint_forward). sbs_shear model modules are byte-identical
#     across the two trees, so worktree-first is safe for loading V2 state dicts. ---
SBSI_WT = "/home/z/Zekang.Zhang/SBSI-ablation"
SBSI_MAIN = "/home/z/Zekang.Zhang/SBSI"
for p in [os.path.join(SBSI_MAIN, "scripts"), SBSI_MAIN,
          os.path.join(SBSI_WT, "scripts"), SBSI_WT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_selfresp_gap import (  # noqa: E402  (worktree/scripts)
    CAT, CROWD, NN, SIZE_EDGES,
    load_ruler, print_size_table, model_selfresp,
)
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from eval_constgold_closure import load_model as load_v2, run_model_on as run_v2  # noqa: E402  (main/scripts)


def _kind(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "primary_preprocessor" in d:
        return "v2"
    if "condition_preprocessor" in d:
        return "v1"
    raise ValueError(f"cannot classify ckpt (no primary/condition preprocessor): {path}")


def score_v2(base, ckpts, device, t0):
    Rf_seeds = []
    delta_used = None
    for c in ckpts:
        model, pp, ns, sc, delta = load_v2(c, device)
        delta_used = delta
        Rf, _Rd = run_v2(base, model, pp, ns, sc, delta, device)
        Rf_seeds.append(Rf)
        print(f"  [V2] {os.path.basename(c)}  delta={delta}  "
              f"<R_flow>={np.nanmean(Rf[np.isfinite(Rf)]):+.4f}  ({time.time()-t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.nanmean(np.stack(Rf_seeds, 0), axis=0), Rf_seeds, delta_used


def score_v1(base, ckpts, device, delta, difference, t0):
    Rf_seeds = []
    for c in ckpts:
        bundle = load_measurement_model(c, device=device)
        Rf = model_selfresp(base, bundle, delta, difference, device)
        Rf_seeds.append(Rf)
        print(f"  [V1/{difference}] {os.path.basename(c)}  "
              f"<R_flow>={np.nanmean(Rf[np.isfinite(Rf)]):+.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return np.nanmean(np.stack(Rf_seeds, 0), axis=0), Rf_seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-ckpt", nargs="+", default=None, help="V2 SetConditionedForwardModel ckpt(s)")
    ap.add_argument("--v2-glob", default=None, help="glob for V2 ckpts (ensemble)")
    ap.add_argument("--v1-ckpt", nargs="+", default=None, help="V1-ladder ConditionalMeanFlow ckpt(s) (control)")
    ap.add_argument("--v1-glob", default=None, help="glob for V1-ladder ckpts (control)")
    ap.add_argument("--v1-difference", choices=["forward", "central"], default="forward",
                    help="V1 finite-difference stencil (forward matches the ladder run)")
    ap.add_argument("--v1-delta", type=float, default=0.05)
    ap.add_argument("--v1-both-stencils", action="store_true",
                    help="also print the V1 control under the OTHER stencil (isolates the stencil effect)")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--nn", default=NN)
    ap.add_argument("--output", default=None, help="optional .npz with per-object arrays")
    args = ap.parse_args()

    v2_ckpts = list(args.v2_ckpt or []) + (sorted(glob.glob(args.v2_glob)) if args.v2_glob else [])
    v1_ckpts = list(args.v1_ckpt or []) + (sorted(glob.glob(args.v1_glob)) if args.v1_glob else [])
    if not v2_ckpts and not v1_ckpts:
        ap.error("give at least one of --v2-ckpt/--v2-glob/--v1-ckpt/--v1-glob")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  V2 ckpts={len(v2_ckpts)}  V1 ckpts={len(v1_ckpts)}", flush=True)
    # classify (defensive: warn if a ckpt is in the wrong bucket)
    for c in v2_ckpts:
        assert _kind(c) == "v2", f"--v2 ckpt is not a V2 model: {c}"
    for c in v1_ckpts:
        assert _kind(c) == "v1", f"--v1 ckpt is not a V1 model: {c}"

    # ---- SHARED ruler built ONCE (identical to eval_selfresp_gap) ----
    ru = load_ruler(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min, args.true_mag_max,
                    args.iso_radius, args.crowd, args.nn, t0=t0)
    base, R_hs, iso, size = ru["base"], ru["R_hs"], ru["iso"], ru["size"]
    iso_label = "ISOLATED acceptance set (nn_bright>%.0f\")" % args.iso_radius
    allsel = np.ones(len(base), bool)
    save = {"R_hs": R_hs, "size": size, "iso": iso, "gmed": ru["gmed"],
            "case": base["case"].to_numpy(int), "input_index": base["input_index"].to_numpy(int),
            "size_edges": SIZE_EDGES}

    if v2_ckpts:
        print("\n" + "=" * 78 + f"\nV2 ensemble ({len(v2_ckpts)} seeds) -- central-secant readout, delta from ckpt\n" + "=" * 78, flush=True)
        Rf_v2, Rf_v2_seeds, d2 = score_v2(base, v2_ckpts, device, t0)
        print(f"<R_flow V2 ENS>={np.nanmean(Rf_v2[np.isfinite(Rf_v2)]):+.4f}  (delta={d2})", flush=True)
        good = np.isfinite(R_hs) & np.isfinite(Rf_v2)
        print_size_table(R_hs, Rf_v2, size, iso, good, "V2  " + iso_label)
        print_size_table(R_hs, Rf_v2, size, allsel, good, "V2  ALL objects (diagnostic)")
        save["R_flow_v2"] = Rf_v2
        save["R_flow_v2_seeds"] = np.stack(Rf_v2_seeds, 0)

    if v1_ckpts:
        stencils = [args.v1_difference]
        if args.v1_both_stencils:
            stencils += [("central" if args.v1_difference == "forward" else "forward")]
        for diff in stencils:
            print("\n" + "=" * 78 + f"\nV1-ladder control ({len(v1_ckpts)} seeds) -- {diff} readout, delta={args.v1_delta}\n" + "=" * 78, flush=True)
            Rf_v1, Rf_v1_seeds = score_v1(base, v1_ckpts, device, args.v1_delta, diff, t0)
            print(f"<R_flow V1 ENS>={np.nanmean(Rf_v1[np.isfinite(Rf_v1)]):+.4f}", flush=True)
            good = np.isfinite(R_hs) & np.isfinite(Rf_v1)
            print_size_table(R_hs, Rf_v1, size, iso, good, f"V1[{diff}]  " + iso_label)
            print_size_table(R_hs, Rf_v1, size, allsel, good, f"V1[{diff}]  ALL objects (diagnostic)")
            save[f"R_flow_v1_{diff}"] = Rf_v1
            save[f"R_flow_v1_{diff}_seeds"] = np.stack(Rf_v1_seeds, 0)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, **save)
        print(f"\nsaved {args.output}")
    print("SELFRESP_GAP_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
