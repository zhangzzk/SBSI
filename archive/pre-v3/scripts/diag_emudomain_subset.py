"""WHO are the 22% of primaries the emulator-domain pair set drops, and why does R_flow move?

Job 15483389 scored `m` on 9,107,497 of the fiducial 11,674,408 primaries and got +4.666% for the
BlendEMU reference, against the fiducial -0.123%. The three ingredients moved like this:

               R_sim    R_flow   R_blend        m
  fiducial    0.8605    0.7268    0.1358   -0.123%
  that run    0.8605    0.6886    0.1336   +4.666%

`R_sim` did not move AT ALL while `R_flow` fell 5.3%. A subset cut that leaves the simulated response
untouched but shifts the model's prediction of it by 5% is not a normal population effect -- either
the dropped primaries are a very specific population, or the two sides are not describing the same
rows. This script decides which, by splitting the SAME dump rows into KEPT / DROPPED and reporting
every ingredient plus the properties that would identify the group.

It changes nothing and fits nothing; it is a read-and-print over existing products.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from plotting.plot_fid_flow_figures import dump_paths, load_dumps  # noqa: E402

EMUDOM = "results/blend_lookup_ens16_emudomain_c40-139.feather"
EXTRA = ["r_input_p", "neighbored", "distance"]


def block(name, mask, ref, flows, rb_emudom):
    n = int(mask.sum())
    if n == 0:
        print(f"  {name:>26}{0:>12}")
        return
    rsim = float(ref["r_sim"].to_numpy(float)[mask].mean())
    rflow = float(flows[:, mask].mean())
    rbl = float(ref["R_blend"].to_numpy(float)[mask].mean())
    # `m` per seed, then the mean -- AGENTS.md: never from ensemble means.
    ms = np.array([rsim / (float(flows[s, mask].mean()) + rbl) - 1.0 for s in range(flows.shape[0])])
    mag = float(ref["r_input_p"].to_numpy(float)[mask].mean())
    nb = float(ref["neighbored"].to_numpy()[mask].mean())
    dmask = mask & np.isfinite(ref["distance"].to_numpy(float))
    dist = float(ref["distance"].to_numpy(float)[dmask].mean()) if dmask.any() else np.nan
    rbe = rb_emudom[mask]
    rbe_m = float(np.nanmean(rbe)) if np.isfinite(rbe).any() else np.nan
    print(f"  {name:>26}{n:>12,}{100*n/len(ref):>8.2f}%{rsim:>10.4f}{rflow:>10.4f}{rbl:>10.4f}"
          f"{rbe_m:>11.4f}{100*ms.mean():>+9.3f}{mag:>9.3f}{100*nb:>8.1f}%{dist:>9.3f}")


def main():
    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    flows = flows.astype(np.float64)
    key = ["case", "input_index"]

    # `load_dumps` reads only case/input_index/r_sim/R_blend, so the identifying properties have to
    # come back off the dump file. Merged on the key rather than assumed positional, because `ref`
    # has already been masked down to the tuned emulator's coverage.
    paths = dump_paths()
    props = pf.read_table(paths[sorted(paths)[0]], columns=key + EXTRA).to_pandas()
    n_before = len(ref)
    ref = ref.merge(props, on=key, how="left")
    if len(ref) != n_before:
        raise SystemExit(f"REFUSING: property merge changed the row count {n_before:,} -> "
                         f"{len(ref):,}; `flows` is positional and would misalign.")
    miss = int((~np.isfinite(ref["r_input_p"].to_numpy(float))).sum())
    if miss:
        print(f"  note: {miss:,} rows have no r_input_p")

    lk = pf.read_table(EMUDOM, memory_map=True).to_pandas()
    n_lk = len(lk)
    n_uniq = len(lk[key].drop_duplicates())
    print(f"\nemudomain lookup: {n_lk:,} rows, {n_uniq:,} unique (case,input_index)")
    if n_lk != n_uniq:
        raise SystemExit("REFUSING: duplicate keys in the lookup -- a left merge would lengthen the "
                         "frame and silently misalign the mask against r_sim/R_flow.")

    j = ref[key].merge(lk[key + ["R_blend_emu", "in_domain", "n_neighbours"]], on=key, how="left")
    if len(j) != len(ref):
        raise SystemExit(f"REFUSING: merge changed the row count {len(ref):,} -> {len(j):,}; the "
                         "positional mask against `flows` would be invalid.")
    rb_emudom = j["R_blend_emu"].to_numpy(float)
    matched = np.isfinite(rb_emudom)
    indom = np.where(matched, j["in_domain"].to_numpy(), False).astype(bool)
    keep = matched & indom

    print(f"matched by the emudomain lookup : {matched.sum():,} ({100*matched.mean():.2f}%)")
    print(f"  ... and flagged in_domain     : {keep.sum():,} ({100*keep.mean():.2f}%)")
    print(f"matched but NOT in_domain       : {int((matched & ~indom).sum()):,}")

    print(f"\n{'='*128}")
    print("The SAME dump rows, split by whether the emulator-domain pair set covers them.")
    print("`R_blend fid` is the tuned emulator's native (whole-field) prediction already in `ref`;")
    print("`R_bl emud` is the emulator re-scored inside the 7\" / neighbour-cut pair set.")
    print(f"{'='*128}")
    print(f"  {'group':>26}{'N':>12}{'share':>9}{'R_sim':>10}{'R_flow':>10}{'R_bl fid':>10}"
          f"{'R_bl emud':>11}{'m %':>9}{'mag_p':>9}{'nbrd':>9}{'dist':>9}")
    block("ALL (fiducial)", np.ones(len(ref), bool), ref, flows, rb_emudom)
    block("KEPT (scored)", keep, ref, flows, rb_emudom)
    block("DROPPED", ~keep, ref, flows, rb_emudom)
    block("  of which: unmatched", ~matched, ref, flows, rb_emudom)
    block("  of which: not in_domain", matched & ~indom, ref, flows, rb_emudom)

    print("\nREAD THIS AS: if DROPPED has a much HIGHER R_flow than KEPT while its R_sim is similar,")
    print("the subset is genuinely a population where flow #1 predicts a smaller self response, and")
    print("+4.666% is a real property of that population -- not a wiring fault. If instead DROPPED")
    print("looks unremarkable, the 5.3% R_flow shift cannot come from the split and something else")
    print("is wrong.")
    print("\nDIAG_EMUDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
