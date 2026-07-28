"""Why does the response catalogue not join to the half-shear catalogue? Diagnose the shared keys.

`eval_shear_amplitude.py` matched ZERO pairs on (case, input_index, neighbour RA/DEC) even though
both catalogues are supposed to come from the same simulation suite (lsst_sims_fs2_25876) and both
store double-precision positions copied from an input catalogue. Either the primary key means
different things in the two files, or the two files describe different galaxies.

This walks the key hierarchy from the outside in and prints where the agreement stops:

  1. do the (case, input_index) sets overlap at all?
  2. for shared input_index values, do the PRIMARY input positions agree?
  3. if they do, how close are the neighbour positions -- exact, or merely nearby?

The answer decides whether the per-pair comparison can be rescued with a looser key (rounding, or a
nearest-position match) or whether the two catalogues genuinely describe different populations, in
which case every cross-catalogue number in this investigation needs re-examining.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.compute as pc

RULER = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather"
RESP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"


def read_case(path, case, cols):
    """Rows of one case, read batch-by-batch so nothing large is materialised."""
    parts = []
    with ipc.open_file(path) as r:
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] == case]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def via_load_legs(case):
    """Where do the shared primaries go once the ruler's own selection is applied?

    The raw catalogues share 100% of the response catalogue's primaries (part 1 below), yet
    eval_shear_amplitude joined ZERO rows on (case, input_index). So the loss happens inside
    `load_legs`: the both-sheared-leg cut, the deliverable-domain cut, or the both-detected match.
    This walks those three in order against the response catalogue's primary set.
    """
    import sys as _sys
    _sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot")
    from scripts.eval_rblend_gap import GAMMA, NGMIX, PAIR_FEATURES
    import pyarrow.feather as pf

    resp_ids = set(read_case(RESP, case, ["case", "input_index"])["input_index"].to_numpy(int))
    print(f"\n4) tracing load_legs' selection against {len(resp_ids):,} response primaries")

    cols = ["case", "input_index", "detected", "neighbored"] + PAIR_FEATURES + NGMIX + GAMMA
    d = pf.read_table(RULER, columns=cols).to_pandas()
    d = d[d["case"] == case]
    step = [("raw rows in this case", d)]

    gp = np.hypot(d["gamma1_input_p"].to_numpy(float), d["gamma2_input_p"].to_numpy(float))
    gs = np.hypot(d["gamma1_input_s"].to_numpy(float), d["gamma2_input_s"].to_numpy(float))
    step.append(("primary sheared", d[gp > 1e-6]))
    step.append(("neighbour sheared", d[gs > 1e-6]))
    nb = d[(gp > 1e-6) & (gs > 1e-6)]
    step.append(("BOTH sheared", nb))
    nb2 = nb[(nb["Re_input_p"].to_numpy(float) > 0.3) & (nb["r_input_p"].to_numpy(float) < 26.0)]
    step.append(("+ deliverable domain", nb2))
    step.append(("+ detected", nb2[nb2["detected"].astype(bool)]))

    for name, frame in step:
        ids = set(frame["input_index"].to_numpy(int))
        print(f"   {name:<24} rows={len(frame):>9,}  primaries={len(ids):>9,}  "
              f"shared with response={len(ids & resp_ids):>9,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--trace-load-legs", action="store_true")
    args = ap.parse_args()

    cols = ["case", "input_index", "RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s",
            "distance"]
    print(f"reading case {args.case} from both catalogues ...", flush=True)
    a = read_case(RULER, args.case, cols)
    b = read_case(RESP, args.case, cols)
    print(f"  ruler    rows={len(a):,}  unique primaries={a['input_index'].nunique():,}")
    print(f"  response rows={len(b):,}  unique primaries={b['input_index'].nunique():,}")
    if not len(a) or not len(b):
        print("one side is empty -- case numbering differs")
        return

    sa = set(a["input_index"].to_numpy(int))
    sb = set(b["input_index"].to_numpy(int))
    inter = sa & sb
    print(f"\n1) input_index overlap: |ruler|={len(sa):,}  |response|={len(sb):,}  "
          f"shared={len(inter):,} ({100*len(inter)/max(len(sb),1):.1f}% of response)")
    print(f"   ruler    input_index range  [{min(sa)}, {max(sa)}]")
    print(f"   response input_index range  [{min(sb)}, {max(sb)}]")

    if not inter:
        print("   -> the primary key does not even share a value range; different id conventions")
        return

    # 2) do the primary positions agree for shared ids?
    ka = a.drop_duplicates("input_index").set_index("input_index")
    kb = b.drop_duplicates("input_index").set_index("input_index")
    ids = np.array(sorted(inter))[:200000]
    dra = (ka.loc[ids, "RA_input_p"].to_numpy(float) - kb.loc[ids, "RA_input_p"].to_numpy(float))
    dde = (ka.loc[ids, "DEC_input_p"].to_numpy(float) - kb.loc[ids, "DEC_input_p"].to_numpy(float))
    sep = np.hypot(dra * np.cos(np.deg2rad(ka.loc[ids, "DEC_input_p"].to_numpy(float))), dde) * 3600
    print(f"\n2) PRIMARY input position, same input_index (N={len(ids):,}):")
    print(f"   median |offset| = {np.median(sep):.6f}\"   mean = {sep.mean():.6f}\"   "
          f"max = {sep.max():.4f}\"")
    print(f"   exact matches (<1e-6\") = {100*(sep < 1e-6).mean():.2f}%")
    if np.median(sep) > 1e-6:
        print("   -> SAME id, DIFFERENT galaxy: the two catalogues do not share an input catalogue")
        return

    # 3) neighbour positions for a shared primary
    m = a[["input_index", "RA_input_s", "DEC_input_s", "distance"]].merge(
        b[["input_index", "RA_input_s", "DEC_input_s", "distance"]],
        on="input_index", suffixes=("_a", "_b"))
    d = np.hypot((m["RA_input_s_a"] - m["RA_input_s_b"]) * np.cos(np.deg2rad(m["DEC_input_s_a"])),
                 m["DEC_input_s_a"] - m["DEC_input_s_b"]).to_numpy(float) * 3600
    print(f"\n3) NEIGHBOUR position, all response rows for a shared primary (N={len(m):,}):")
    print(f"   exact matches (<1e-6\") = {100*(d < 1e-6).mean():.3f}%   "
          f"(rows per primary: {len(m)/max(len(inter),1):.1f})")
    ex = m[d < 1e-6]
    print(f"   rows where the SAME neighbour is named: {len(ex):,}")
    if len(ex):
        print(f"   their distance columns: ruler mean={ex['distance_a'].mean():.4f}\"  "
              f"response mean={ex['distance_b'].mean():.4f}\"  "
              f"<resp-ruler>={ (ex['distance_b']-ex['distance_a']).mean():+.4f}\"")

    if args.trace_load_legs:
        via_load_legs(args.case)
    print("DIAG_JOIN_DONE", flush=True)


if __name__ == "__main__":
    main()
