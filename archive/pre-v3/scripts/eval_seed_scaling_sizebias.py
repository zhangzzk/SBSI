"""DIAGNOSTIC (not a gate): would MORE FLOW SEEDS remove the measured-size selection bias?

WHY THIS EXISTS. The 2026-08-01i re-gate measured a selection excess dm(A2) = +2.10 pt on the
measured-size cut and showed the realisation-aware head recovers only 8% of it. The natural follow-up
is whether the excess is simply seed noise that a larger ensemble would average away.

It cannot be, if the excess is a SYSTEMATIC: seeds average over training-initialisation randomness,
so they shrink the SPREAD of dm across checkpoints as 1/sqrt(N) but leave its MEAN untouched. This
script measures both, so the claim rests on a number rather than on the argument.

WHAT IT COMPUTES, PER SEED (never from ensemble means -- forming the ratio inside each seed and then
taking the spread across seeds is the convention in AGENTS.md, and building it from ensemble means is
exactly the bug that returns the same error bar on every row):

    m_s(S)  = R_sim(S) / R_flow_s(S) - 1          on population S
    dm_s(S) = m_s(S) - m_s(all)                   the selection-induced excess for seed s

then mean, std and sem across the 16 seeds, plus the ensemble-mean value the gate actually reported
(R_model = mean_s R_flow_s), and the seed count that would be needed for the seed error alone to
bring |dm| under the 0.30 pt deliverable budget -- which is only meaningful if the MEAN is already
inside the budget, and the script says so explicitly when it is not.

HALF-SHEAR MODEL SIDE IS R_flow ALONE (no R_blend) -- same convention as the re-gate, so these dm
values are directly comparable to results/regate_result_h.json.

DIAGNOSTIC STATUS. The re-gate on cases 40-199 is already recorded as FAIL; this script cannot change
that verdict and is not a re-gate. It introduces no criterion and no threshold.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/"
SEEDS = [501, 502, 503, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517]
PIX = 0.2  # arcsec / pixel; measured_flux_radius is in PIXELS
BUDGET_PT = 0.30  # the deliverable budget, GOALS.md realistic-cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", default=HS + "base_c40-199.feather")
    ap.add_argument("--selfresp", default=HS + "halfshear_selfresp_c40-199.feather")
    args = ap.parse_args()

    sr = pd.read_feather(args.selfresp)
    bs = pd.read_feather(args.base_cache, columns=["case", "input_index",
                                                   "measured_flux_radius_0", "measured_mag_auto_0"])
    # The dump was written from this exact cache, so the keys are elementwise identical. Assert it
    # rather than assume it -- a silent misalignment here would corrupt every number below.
    assert len(sr) == len(bs), f"row count mismatch {len(sr)} vs {len(bs)}"
    assert (sr["case"].to_numpy() == bs["case"].to_numpy()).all(), "case column not aligned"
    assert (sr["input_index"].to_numpy() == bs["input_index"].to_numpy()).all(), "index not aligned"
    print(f"rows {len(sr):,}  cases [{sr['case'].min()},{sr['case'].max()}]  keys aligned elementwise")

    rsim = sr["r_sim_self"].to_numpy(float)
    rflow = np.column_stack([sr[f"R_flow_s{s}"].to_numpy(float) for s in SEEDS])
    rad = bs["measured_flux_radius_0"].to_numpy(float)
    mag = bs["measured_mag_auto_0"].to_numpy(float)

    good = np.isfinite(rsim) & np.isfinite(rflow).all(1) & np.isfinite(rad) & np.isfinite(mag)
    rsim, rflow, rad, mag = rsim[good], rflow[good], rad[good], mag[good]
    print(f"good rows {good.sum():,} / {good.size:,} ({good.mean():.6f})")

    cuts = {
        "A2  size > 3.0 px (0.60\")": rad > 3.0,
        "A4  size > 3.5 px (0.70\")": rad > 3.5,
        "A1  mag < 26.0": mag < 26.0,
    }

    for name, cut in cuts.items():
        print(f"\n{'='*74}\n{name}   keep fraction {cut.mean():.6f}\n{'='*74}")
        # per seed: form the ratio INSIDE the seed, then take the spread across seeds
        dm = np.array([
            (rsim[cut].mean() / rflow[cut, i].mean() - 1) - (rsim.mean() / rflow[:, i].mean() - 1)
            for i in range(len(SEEDS))]) * 100.0
        # the ensemble value the gate reported: R_model = mean over seeds, THEN the ratio
        rm_all, rm_cut = rflow.mean(1), rflow[cut].mean(1)
        dm_ens = ((rsim[cut].mean() / rm_cut.mean() - 1) - (rsim.mean() / rm_all.mean() - 1)) * 100.0

        print(f"  per-seed dm (pt): min {dm.min():+.4f}  max {dm.max():+.4f}")
        for s, v in zip(SEEDS, dm):
            print(f"    s{s}: {v:+.4f}")
        sd, sem = dm.std(ddof=1), dm.std(ddof=1) / np.sqrt(len(SEEDS))
        print(f"  mean over {len(SEEDS)} seeds = {dm.mean():+.4f} pt   std {sd:.4f}   sem {sem:.4f}")
        print(f"  ensemble-mean R_model value  = {dm_ens:+.4f} pt  (this is what the gate reports)")
        print(f"  every seed has the same sign: {bool((np.sign(dm) == np.sign(dm.mean())).all())}")
        print(f"  signal-to-seed-noise |mean|/sem = {abs(dm.mean())/sem:.1f}")

        # Could MORE seeds bring |dm| inside the budget? Only if the MEAN is already inside it --
        # extra seeds shrink sem, never the mean. Say so rather than printing a misleading N.
        if abs(dm.mean()) <= BUDGET_PT:
            print(f"  mean is already within the {BUDGET_PT} pt budget; seeds are not the obstacle")
        else:
            excess = abs(dm.mean()) - BUDGET_PT
            print(f"  NO SEED COUNT HELPS: the MEAN alone is {abs(dm.mean()):.4f} pt, which is "
                  f"{excess:.4f} pt OUTSIDE the {BUDGET_PT} pt budget before any error bar is added. "
                  f"sem -> 0 as N -> inf leaves {abs(dm.mean()):.4f} pt.")
            print(f"  for scale, mean/std across seeds = {abs(dm.mean())/sd:.1f}: the checkpoints "
                  f"agree with EACH OTHER {abs(dm.mean())/sd:.0f}x better than any of them agrees "
                  f"with the simulation.")

    # The diagnosed mechanism: the response in the smallest measured-size bin, per seed.
    print(f"\n{'='*74}\nsub-3px (0.60\") measured-size bin -- the diagnosed defect, per seed\n{'='*74}")
    small = rad <= 3.0
    print(f"  population fraction {small.mean():.5f}   N {small.sum():,}")
    print(f"  sim  response in bin: {rsim[small].mean():+.4f}   (population mean {rsim.mean():+.4f})")
    ratios = []
    for s, i in zip(SEEDS, range(len(SEEDS))):
        mo = rflow[small, i].mean()
        ratios.append(mo / rsim[small].mean())
        print(f"    s{s}: model {mo:+.4f}   model/sim {ratios[-1]:+.3f}x")
    ratios = np.array(ratios)
    print(f"  model/sim over-prediction: mean {ratios.mean():.3f}x  std {ratios.std(ddof=1):.3f}  "
          f"min {ratios.min():.3f}  max {ratios.max():.3f}")
    print("  -> a consistent over-prediction across ALL seeds is a SYSTEMATIC, not seed noise; "
          "averaging more checkpoints converges to it, not away from it.")


if __name__ == "__main__":
    main()
