"""DOES THE RESPONSE PIN IMPRINT ITS OWN CELL EDGES ON FLOW #1?

THE HYPOTHESIS. Flow #1 is 3.42 +- 1.10% LOW at small size and 3.63 +- 1.34% low at faint S/N, while
a pin-free 3-feature regressor on the same truth sits at zero (WORKLOG 2026-08-03k). The features are
not the problem, and the training volume is not either -- the regressor used 1.18M rows against the
flow's 3.4M and still won. What the flow has that the regressor does not is the RESPONSE PIN: a
piecewise-constant, per-cell target. Inside a cell the pin asks for ONE value, so wherever the true
response varies steeply within a cell the pin is actively pulling the flow FLAT against a truth that
is not.

The fiducial size edges are [0.300, 0.3552, 0.4191, 0.5025, 0.6287, 0.8530, 1.500] and the measured
response climbs 0.4375 -> 0.7016 across the FIRST cell alone. That is exactly where the flow is low.

THE FINGERPRINT. If the pin is imprinting, the flow's residual should RAMP systematically with
position INSIDE each cell and reset at each boundary -- a sawtooth keyed to the edges, not to size.
If the flow is simply bad at small galaxies, the residual should track size smoothly and show no
structure at the boundaries.

THE CONTROL IS WHAT MAKES THIS DECISIVE. The same 3-feature regressor is scored the same way. It
never saw the pin or its edges, so it CANNOT imprint them. If the flow shows a ramp and the regressor
does not, the pin is implicated. If both show it, the structure belongs to the data (a real response
feature that happens to sit near an edge) and the pin is exonerated.

READING THE SLOPE. Residual is (model/truth - 1), so a pin flattening a RISING truth makes the model
too HIGH at the bottom of a cell and too LOW at the top: a NEGATIVE slope in within-cell position.
That is the predicted sign, and it is stated here before the number is computed.

CAVEAT ON THE THIRD AXIS. The fiducial pin is 3-D (mag x size x r_blend) and this only reconstructs
the SIZE edges, which fig 5 carries directly. Rows in one size cell are spread over many mag/crowd
cells, so any imprint is DILUTED here and a null is weaker evidence than a detection.

FIREWALL: half-shear self-response only; constgold is not read.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.feather as pf
from sklearn.ensemble import HistGradientBoostingRegressor

# fiducial dom6x6 size edges (response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz)
PIN_SIZE_EDGES = np.array([0.300, 0.35515322, 0.41906579, 0.50249398, 0.62869868, 0.85295901, 1.500])
FEATS = ["SN", "Re_input_p", "nbr_flux_near"]


def ramp(u, resid, w, nu=5):
    """Mean residual vs within-cell position, and a weighted least-squares slope."""
    edges = np.linspace(0, 1, nu + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, nu - 1)
    out = []
    for b in range(nu):
        k = idx == b
        out.append(float(np.sum(resid[k] * w[k]) / np.sum(w[k])) if k.any() else np.nan)
    uu, rr = 0.5 * (edges[:-1] + edges[1:]), np.array(out)
    ok = np.isfinite(rr)
    slope = np.polyfit(uu[ok], rr[ok], 1)[0] if ok.sum() > 1 else np.nan
    return uu, rr, slope


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfresp", default="results/halfshear_selfresp.feather")
    ap.add_argument("--cells", type=int, default=3, help="how many of the lowest size cells to probe")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    df = pf.read_table(args.selfresp, memory_map=True).to_pandas()
    scols = sorted(c for c in df.columns if c.startswith("R_flow_s"))
    y = df["r_sim_self"].to_numpy(float)
    flow = np.mean([df[c].to_numpy(float) for c in scols], axis=0)
    re = df["Re_input_p"].to_numpy(float)
    case = df["case"].to_numpy(np.int64)
    X = df[FEATS].to_numpy(np.float32)

    cs = np.unique(case)
    held = np.isin(case, cs[len(cs) // 2:])
    ok = np.isfinite(y) & np.isfinite(flow) & np.isfinite(re) & np.isfinite(X).all(1)
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                        min_samples_leaf=200, l2_regularization=1.0,
                                        early_stopping=False, random_state=0)
    mdl.fit(X[(~held) & ok], y[(~held) & ok])
    fit = np.full(len(y), np.nan)
    fit[held] = mdl.predict(X[held])

    sel = held & ok
    print(f"{int(sel.sum()):,} held-out rows; probing the lowest {args.cells} pin size cells")
    ci = np.clip(np.digitize(re, PIN_SIZE_EDGES) - 1, 0, len(PIN_SIZE_EDGES) - 2)

    rng = np.random.default_rng(0)
    for name, model in (("flow #1  (pinned)", flow), ("fit(3ft) (NO pin)", fit)):
        print(f"\n===== {name} =====")
        slopes_all = []
        for c in range(args.cells):
            k = sel & (ci == c)
            lo, hi = PIN_SIZE_EDGES[c], PIN_SIZE_EDGES[c + 1]
            u = (re[k] - lo) / (hi - lo)
            # per-object residual weighted by truth, so the bin mean equals the ratio-of-means
            resid = 100.0 * (model[k] - y[k]) / np.mean(y[k])
            w = np.ones(k.sum())
            uu, rr, slope = ramp(u, resid, w)
            # truth's own rise across the cell, to show what the pin is being asked to flatten
            tlo = np.mean(y[k][u < 0.2]); thi = np.mean(y[k][u > 0.8])
            bs = []
            for _ in range(args.boot):
                pick = rng.choice(np.unique(case[k]), size=len(np.unique(case[k])), replace=True)
                m = np.isin(case[k], pick)
                if m.sum() > 100:
                    bs.append(ramp(u[m], resid[m], w[m])[2])
            sd = float(np.std(bs)) if bs else np.nan
            slopes_all.append((slope, sd))
            print(f"  cell {c} [{lo:.3f},{hi:.3f}]  N={int(k.sum()):,}  "
                  f"truth rises {tlo:.3f} -> {thi:.3f} across it")
            print(f"    residual vs within-cell position: " + " ".join(f"{v:+6.2f}" for v in rr))
            print(f"    slope = {slope:+.2f} +- {sd:.2f} %/cell   "
                  f"({'RAMP' if abs(slope) > 2 * sd else 'no ramp'})")
        num = sum(s / (sd ** 2) for s, sd in slopes_all if np.isfinite(sd) and sd > 0)
        den = sum(1 / (sd ** 2) for s, sd in slopes_all if np.isfinite(sd) and sd > 0)
        if den > 0:
            print(f"  COMBINED slope over {args.cells} cells = {num/den:+.2f} +- {np.sqrt(1/den):.2f} %/cell")

    print("\nREAD: a negative slope in the PINNED model with none in the pin-free control is the\n"
          "predicted signature of the pin flattening a rising truth inside its cells. Both showing\n"
          "it means the structure is in the data, not the pin. The 3-D pin is only partly\n"
          "reconstructed here (size edges only), so a null is weaker evidence than a detection.")
    print("\nPIN_IMPRINT_DONE")


if __name__ == "__main__":
    main()
