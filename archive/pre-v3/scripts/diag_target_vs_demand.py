"""Why is the V2.2 response TARGET ~2.5% below what constgold demands?

THE FINDING THIS FOLLOWS UP. WORKLOG 2026-08-07c localized the V2.2 closure deficit to a uniform
~1% shortfall in `R_flow`, flat in size, magnitude and S/N. 2026-08-07d then measured the pieces on
the V2.2 box and found the flow is NOT the culprit:

    half-shear self-response, |g|=0.02  R = 0.8161 +- 0.0036   (this is the target's own level)
    constgold demand, R_sim - R_blend   R = 0.8369              (what closure requires)

The flow sits slightly ABOVE its target and still falls short, because the TARGET ITSELF is ~2.5%
below the demand. Shear nonlinearity is excluded as the carrier (+0.16% +- 0.39,
`diag_response_nonlinearity.py`). So the gap is in one of three places, and this script separates
the first from the other two:

  (1) POPULATION -- the half-shear legs (cases 0-99) and constgold (min-case 40) do not have the
      same true (mag, size) mix inside the box, so the two averages weight cells differently.
  (2) LEVEL, R_blend -- the emulator's R_blend is subtracted to form the demand. If it is too small,
      the demand is too large. The independent per-pair ruler already reports the emulator LOW
      (v22 -9.78%, `_ho` -6.89%), the right sign, though neither is significant on its own.
  (3) LEVEL, extraction convention -- half-shear is forward, constgold antithetic (CONVENTIONS.md
      6c). 2026-08-05u bounds that term at -0.55% +- 0.61 at |g|=0.02.

HOW (1) IS SEPARATED. The half-shear self-response is re-averaged using CONSTGOLD's (mag, size)
cell counts instead of its own. That reweighting happens entirely inside the half-shear catalogue --
no convention is crossed and nothing is differenced across catalogues -- so whatever it moves IS the
population term, cleanly. What survives the reweighting is level, i.e. (2) + (3) together, and the
per-bin table says whether the survivor is FLAT (a level offset, consistent with (2)/(3)) or TILTED
(something population-like that the 2D reweighting did not capture, e.g. crowding, which constgold
has no column for).

WHAT THIS SCRIPT DOES NOT DO. It does not correct anything, and it does not attribute the surviving
gap to R_blend. Separating (2) from (3) needs the per-pair ruler and the antithetic bound, not this
comparison. Reporting a gap is not licence to close it: per AGENTS.md "Numerical Integrity", no
offset found here may be folded into any reported quantity.

SEEDS. R_sim and R_blend are both seed-independent, so ONE dump carries the whole demand; the flow's
R_flow is read from the same dump for context only and is labelled single-seed. No `m` is reported.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbs_shear.halfshear import (LEG_PATHS, case_blocked, load_g0_lookup, load_leg, pack_key,
                                 self_response_per_object, verify_independent_shear_directions)

CONSTGOLD = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
             "constant_response_catalogue_train.feather")
DUMPS = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
         "v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather")


def load_constgold_demand(dump_glob, catalogue, min_case, mag_max, re_min):
    """Per-object r_sim / R_blend / R_flow from one dump, joined to true size from the catalogue."""
    path = sorted(glob.glob(dump_glob))[0]
    d = pd.read_feather(path)
    d = d[d["case"].to_numpy() >= min_case]

    sizes = []
    with ipc.open_file(pa.memory_map(catalogue)) as reader:
        for bi in range(reader.num_record_batches):
            b = (pa.Table.from_batches([reader.get_batch(bi)])
                 .select(["case", "input_index", "Re_input_p"]).to_pandas())
            sizes.append(b[b["case"].to_numpy() >= min_case])
    sizes = pd.concat(sizes, ignore_index=True)

    key_d, key_s = pack_key(d["case"], d["input_index"]), pack_key(sizes["case"],
                                                                   sizes["input_index"])
    order = np.argsort(key_s)
    key_s, re_s = key_s[order], sizes["Re_input_p"].to_numpy(float)[order]
    pos = np.clip(np.searchsorted(key_s, key_d), 0, len(key_s) - 1)
    match = key_s[pos] == key_d
    d = d.assign(Re_input_p=np.where(match, re_s[pos], np.nan))
    print(f"  dump {path.split('/')[-1]}: {len(d):,} rows, size join matched {match.mean():.2%}")

    d = d[np.isfinite(d["Re_input_p"].to_numpy())]
    d = d[(d["r_input_p"].to_numpy() < mag_max) & (d["Re_input_p"].to_numpy() > re_min)]
    return d.reset_index(drop=True)


def cell_index(mag, size, mag_edges, size_edges):
    i = np.clip(np.digitize(mag, mag_edges) - 1, 0, len(mag_edges) - 2)
    j = np.clip(np.digitize(size, size_edges) - 1, 0, len(size_edges) - 2)
    return i * (len(size_edges) - 1) + j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=0.02, choices=sorted(LEG_PATHS))
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--n-mag", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()

    mag_edges = np.linspace(18.0, args.mag_max, args.n_mag + 1)
    size_edges = np.linspace(args.re_min, 1.5, args.n_size + 1)
    print(f"V2.2 box: true mag < {args.mag_max}, Re > {args.re_min}")
    print(f"cells: {args.n_mag} mag x {args.n_size} size "
          f"(size grid truncated at 1.5, matching the target's own edges)\n")

    print("HALF-SHEAR SELF-RESPONSE (forward, SNC, ghat_p projection)")
    g0 = load_g0_lookup()
    leg = load_leg(LEG_PATHS[args.g], args.mag_max, args.re_min, args.max_case)
    conc = verify_independent_shear_directions(leg)
    print(f"  within-case shear-direction concentration = {conc:.4f} "
          f"({'OK: directions spread, projection is self-only' if conc < 0.1 else 'WARNING'})")
    hs, st = self_response_per_object(leg, args.g, g0, keep_cols=("r_input_p", "Re_input_p"))
    del leg
    print(f"  |g|={args.g}: {st['n_rows']:,} rows -> {len(hs):,} objects, "
          f"SNC matched {st['snc_matched']:.2%}  ({time.time()-t0:.0f}s)", flush=True)

    print("\nCONSTGOLD DEMAND (R_sim - R_blend), single dump; both terms seed-independent")
    cg = load_constgold_demand(DUMPS, CONSTGOLD, args.min_case, args.mag_max, args.re_min)
    print(f"  {len(cg):,} objects in box  ({time.time()-t0:.0f}s)", flush=True)

    hs_R, hs_e, ncase = case_blocked(hs["R"], hs["case"])
    demand = cg["r_sim"].to_numpy(float) - cg["R_blend"].to_numpy(float)
    dm_R, dm_e, _ = case_blocked(demand, cg["case"])
    rsim, _, _ = case_blocked(cg["r_sim"], cg["case"])
    rbl, _, _ = case_blocked(cg["R_blend"], cg["case"])
    rfl, _, _ = case_blocked(cg["R_flow"], cg["case"])

    print(f"\n  half-shear self-response   = {hs_R:.5f} +- {hs_e:.5f}   ({ncase} cases)")
    print(f"  constgold R_sim            = {rsim:.5f}")
    print(f"  constgold R_blend (v22)    = {rbl:.5f}")
    print(f"  constgold DEMAND           = {dm_R:.5f} +- {dm_e:.5f}")
    print(f"  GAP (half-shear - demand)  = {hs_R-dm_R:+.5f}  = {100*(hs_R-dm_R)/dm_R:+.2f}%")
    print(f"  [context, single seed] R_flow = {rfl:.5f}, i.e. {100*(rfl-hs_R)/hs_R:+.2f}% vs its "
          f"own target level")

    ci_h = cell_index(hs["r_input_p"].to_numpy(), hs["Re_input_p"].to_numpy(), mag_edges, size_edges)
    ci_c = cell_index(cg["r_input_p"].to_numpy(), cg["Re_input_p"].to_numpy(), mag_edges, size_edges)
    ncell = (len(mag_edges) - 1) * (len(size_edges) - 1)
    n_h = np.bincount(ci_h, minlength=ncell).astype(float)
    n_c = np.bincount(ci_c, minlength=ncell).astype(float)
    R_h = np.divide(np.bincount(ci_h, weights=hs["R"], minlength=ncell), n_h,
                    out=np.full(ncell, np.nan), where=n_h > 0)
    R_c = np.divide(np.bincount(ci_c, weights=demand, minlength=ncell), n_c,
                    out=np.full(ncell, np.nan), where=n_c > 0)

    ok = (n_h > 0) & (n_c > 0) & np.isfinite(R_h) & np.isfinite(R_c)
    hs_rw = float(np.sum(R_h[ok] * n_c[ok]) / np.sum(n_c[ok]))
    hs_own = float(np.sum(R_h[ok] * n_h[ok]) / np.sum(n_h[ok]))
    dm_own = float(np.sum(R_c[ok] * n_c[ok]) / np.sum(n_c[ok]))

    print(f"\nPOPULATION vs LEVEL   ({ok.sum()}/{ncell} cells populated in both)")
    print(f"  half-shear, its OWN population        = {hs_own:.5f}")
    print(f"  half-shear, CONSTGOLD's population    = {hs_rw:.5f}")
    print(f"  -> population term                    = {100*(hs_rw-hs_own)/dm_own:+.2f}%")
    print(f"  -> LEVEL term that survives it        = {100*(hs_rw-dm_own)/dm_own:+.2f}%")
    print("  (the population term is an internal reweighting of the half-shear catalogue only:")
    print("   no convention crossed, so it is clean. The survivor is R_blend + convention.)")

    print("\nIS THE SURVIVOR FLAT OR TILTED?  per-cell (half-shear - demand) / demand, %")
    hdr = "  mag \\ Re  " + "".join(f"{size_edges[j]:>8.2f}" for j in range(len(size_edges) - 1))
    print(hdr)
    for i in range(len(mag_edges) - 1):
        row = f"  [{mag_edges[i]:.1f},{mag_edges[i+1]:.1f})"
        for j in range(len(size_edges) - 1):
            c = i * (len(size_edges) - 1) + j
            row += f"{100*(R_h[c]-R_c[c])/R_c[c]:>8.1f}" if ok[c] else "       ."
        print(row)
    rel = 100 * (R_h[ok] - R_c[ok]) / R_c[ok]
    w = n_c[ok] / n_c[ok].sum()
    mean = float(np.sum(w * rel))
    spread = float(np.sqrt(np.sum(w * (rel - mean) ** 2)))
    print(f"\n  count-weighted mean {mean:+.2f}%, spread {spread:.2f} pt across cells")
    print("  A FLAT survivor (spread << |mean|) points at a LEVEL cause -- R_blend or the")
    print("  forward/antithetic convention. A TILTED one points at a population axis this 2D")
    print("  grid misses; crowding is the obvious candidate, and constgold carries no crowding")
    print("  column, so that would need one built before it could be tested.")
    print("\nDIAG_TARGET_VS_DEMAND_DONE", flush=True)


if __name__ == "__main__":
    main()
