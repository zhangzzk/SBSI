"""CPU-only recompute of the TRUTH selection response R_sim + matched-pair error per flux/size cut.

Same matched both-detected det_meas base and cuts as eval_selection_response.py, but no flow -> fast.
Used to give the sim dots proper (matched-pair) error bars in plot_selection_bias.py; the flow side
(R_model + per-seed spread) already comes from the GPU npz.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from eval_selection_response import build_base, truth_selected_response, CROWD, NN  # noqa: E402
from sbs_shear.paths import CATALOGUES as CAT



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.02_test.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.5, 25.0, 25.5, 26.0, 26.5])
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                                        "truth_selerr_v2.npz")
    args = ap.parse_args()
    t0 = time.time()
    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, 0.3, 26.0, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    cuts = ([dict(name=f"size>{c:g}", raw=c, kind="size", xcol="measured_flux_radius", keep_high=True)
             for c in args.size_cuts]
            + [dict(name=f"mag<{c:g}", raw=c, kind="mag", xcol="measured_mag_auto", keep_high=False)
               for c in args.mag_cuts])

    rows = []
    _, _, Rnc, Rnc_err = truth_selected_response(base, gh1, gh2, gmed, iso, "measured_flux_radius",
                                                 -1e9, True)
    rows.append(dict(cut="NO CUT", raw=np.nan, kind="nocut", R_sim=Rnc, R_sim_err=Rnc_err))
    print(f"NO CUT  R_sim={Rnc:+.4f} +- {Rnc_err:.4f}")
    for c in cuts:
        R, frac, _, R_err = truth_selected_response(base, gh1, gh2, gmed, iso, c["xcol"],
                                                    c["raw"], c["keep_high"])
        rows.append(dict(cut=c["name"], raw=c["raw"], kind=c["kind"], R_sim=R, R_sim_err=R_err))
        print(f"{c['name']:>10}  R_sim={R:+.4f} +- {R_err:.4f}  frac={frac:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, gmed=gmed, pixel_size=0.2,
             cut=np.array([r["cut"] for r in rows]),
             kind=np.array([r["kind"] for r in rows]),
             raw=np.array([r["raw"] for r in rows], float),
             R_sim=np.array([r["R_sim"] for r in rows], float),
             R_sim_err=np.array([r["R_sim_err"] for r in rows], float))
    print(f"saved {args.output}  ({time.time()-t0:.1f}s)")
    print("TRUTH_SELERR_DONE", flush=True)


if __name__ == "__main__":
    main()
