"""Are different neighbours of the SAME primary given INDEPENDENT shear directions?

WHY THIS DECIDES WHETHER THE PER-PAIR RULER IS VALID AT ALL
-----------------------------------------------------------
The per-pair blending label

    truth_j = ((e1_both - e1_0) * ghat1_j + (e2_both - e2_0) * ghat2_j) / |g_j|

is claimed to be an UNBIASED estimator of pair j's own response. The whole claim rests on one
assumption (`eval_rblend_gap.py`, `eval_rblend_gap_summed.py`): the other neighbours enter only
through `cos 2(theta_k - theta_j)`, "which has zero mean over their independent directions". If the
simulation instead assigns ONE secondary shear direction per scene -- or merely correlated ones --
then

    E[truth_j] = R_blend(j) + sum_{k != j} R_blend(k) * E[cos 2(theta_k - theta_j)]

and the label measures something between the pair's response and the primary's SUMMED response. Every
per-pair number in this project inherits that: the -41.5% close-pair deficit, the emulator's apparent
accuracy at 2-4", and flow #2's training labels.

WHY THE EXISTING NULL TEST DOES NOT COVER THIS. The 45-degree null projects on the spin-2 orthogonal
direction, giving `sum_k R_k sin 2(theta_k - theta_j)`. Alignment is symmetric under `theta -> -theta`
about the common direction, so the SINE channel averages to zero even when the COSINE channel does
not. A passing null test is therefore consistent with fully aligned neighbour shears. This has to be
measured separately, and it never has been.

WHAT IS MEASURED. For every primary with at least two annotated neighbours, the mean of
`cos 2(theta_j - theta_k)` over distinct neighbour pairs (j, k).

    ~0  -> directions are independent; the per-pair label is unbiased as documented.
    ~1  -> one shear direction per scene; the per-pair label is really the SUMMED response and the
           per-pair ruler cannot be read pair by pair.
    in between -> partial contamination, scaling with how many neighbours a primary has.

A companion check reports the mean over ALL primaries of `sum_{k != j} cos 2(theta_k - theta_j)`,
which is the actual multiplier on the contamination term.

FIREWALL: reads half-shear input columns only. Nothing is fitted; constgold is never opened.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sbs_shear.paths import CATALOGUES as CAT

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

COLS = ["case", "input_index", "detected", "distance",
        "gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", default=CAT + "det_meas_ngmix_ap7_g0.05_pilot.feather")
    ap.add_argument("--max-case", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    parts = []
    with pa.memory_map(args.leg, "rb") as src:
        rd = ipc.open_file(src)
        for i in range(rd.num_record_batches):
            b = pa.Table.from_batches([rd.get_batch(i)]).select(COLS).to_pandas()
            if args.max_case is not None:
                b = b[b["case"] < args.max_case]
            gp = np.hypot(b["gamma1_input_p"].to_numpy(float), b["gamma2_input_p"].to_numpy(float))
            gs = np.hypot(b["gamma1_input_s"].to_numpy(float), b["gamma2_input_s"].to_numpy(float))
            b = b[(gp > 1e-6) & (gs > 1e-6)]
            if len(b):
                parts.append(b)
    d = pd.concat(parts, ignore_index=True)
    print(f"both-sheared rows: {len(d):,}  cases={d['case'].nunique()}  ({time.time()-t0:.0f}s)")

    g1 = d["gamma1_input_s"].to_numpy(float)
    g2 = d["gamma2_input_s"].to_numpy(float)
    gm = np.hypot(g1, g2)
    h1, h2 = g1 / gm, g2 / gm            # spin-2 unit vector of each NEIGHBOUR's shear

    key = d["case"].to_numpy(np.int64) * 1_000_003 + d["input_index"].to_numpy(np.int64)
    _, pid = np.unique(key, return_inverse=True)
    npid = pid.max() + 1
    k = np.bincount(pid, minlength=npid)
    print(f"primaries: {npid:,}   <neighbours per primary> = {k.mean():.3f}   "
          f"with >=2 neighbours: {int((k >= 2).sum()):,}")

    # For a primary with neighbours j, sum_j h_j is the resultant. Then
    #   sum_{j != k} cos 2(theta_j - theta_k) = |sum_j h_j|^2 - k
    # exactly, because cos 2(theta_j - theta_k) = h_j . h_k in the spin-2 basis. That gives the
    # whole cross-term budget without ever forming O(k^2) pairs.
    S1 = np.bincount(pid, weights=h1, minlength=npid)
    S2 = np.bincount(pid, weights=h2, minlength=npid)
    cross = S1 ** 2 + S2 ** 2 - k                      # sum over ORDERED distinct pairs
    npairs = k * (k - 1)                               # number of ordered distinct pairs
    m = npairs > 0
    mean_cos = cross[m].sum() / npairs[m].sum()
    # error over PRIMARIES (the independent unit)
    per = np.where(m, cross / np.maximum(npairs, 1), 0.0)[m]
    sem = per.std(ddof=1) / np.sqrt(m.sum())

    print(f"\n{'='*100}")
    print("MEAN cos 2(theta_j - theta_k) OVER DISTINCT NEIGHBOUR PAIRS OF THE SAME PRIMARY")
    print(f"{'='*100}")
    print(f"  primaries with >=2 neighbours : {int(m.sum()):,}")
    print(f"  ordered distinct pairs        : {int(npairs[m].sum()):,}")
    print(f"  mean cos2(dtheta)             : {mean_cos:+.5f} +- {sem:.5f} "
          f"({abs(mean_cos)/sem:.1f} sigma from zero)")

    print(f"\n  INTERPRETATION")
    if abs(mean_cos) < 3 * sem:
        print("  Consistent with ZERO -> neighbour shear directions ARE independent, so the")
        print("  per-pair label is unbiased as documented and the per-pair ruler is valid. The")
        print("  contamination from other neighbours is variance only, not bias.")
    else:
        print("  NOT consistent with zero -> the other neighbours DO contribute to the per-pair")
        print("  label in the mean. Every per-pair number in this project is affected, including")
        print("  the -41.5% close-pair deficit and the emulator's apparent accuracy at 2-4\".")
        print(f"  The contamination multiplier is <sum_{{k!=j}} cos2> = "
              f"{cross[m].sum()/m.sum():+.4f} per primary.")

    # The same statistic against the PRIMARY's direction, which the documented decorrelation
    # argument also relies on (and which HAS been measured before: mean cos = -0.0000, sd = 0.7071).
    p1 = d["gamma1_input_p"].to_numpy(float) / np.hypot(d["gamma1_input_p"].to_numpy(float),
                                                        d["gamma2_input_p"].to_numpy(float))
    p2 = d["gamma2_input_p"].to_numpy(float) / np.hypot(d["gamma1_input_p"].to_numpy(float),
                                                        d["gamma2_input_p"].to_numpy(float))
    cps = p1 * h1 + p2 * h2
    print(f"\n  cross-check, primary vs its own neighbour: mean cos2 = {cps.mean():+.5f} "
          f"+- {cps.std(ddof=1)/np.sqrt(len(cps)):.5f}  (sd {cps.std():.4f}; the recorded value is "
          f"mean -0.0000, sd 0.7071)")

    # Does the effect grow with multiplicity? A per-SCENE shear would give exactly 1.0 in every bin.
    print(f"\n  {'k':>4}{'primaries':>12}{'mean cos2(dtheta)':>20}")
    for kk in range(2, 9):
        s = m & (k == kk)
        if s.sum() < 50:
            continue
        print(f"  {kk:>4}{int(s.sum()):>12,}{cross[s].sum()/npairs[s].sum():>20.5f}")

    print(f"\nNEIGHBOUR_SHEAR_INDEPENDENCE_DONE  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
