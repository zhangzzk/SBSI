"""Look up a flux x size x blend response grid per-object on the constgold cert catalogue.

Produces a {case, input_index, value} override .npz (same format as harvest_joint_rflow.py)
consumable by eval_selection_robustness.py --rflow-override / --rblend-override.

This is the VALIDATION-side application of a grid built ONLY from the half-shear sims
(compute_deltaet_target.py). The constgold catalogue supplies only the per-object
(flux,size,blend) COORDINATES for the lookup -- its measured response (r_sim) is never used
here, so no constgold information leaks into R_flow/R_blend. Bin assignment replicates
compute_deltaet_target.py EXACTLY (same edges from the grid npz; isolated -> blend bin 0).
"""
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="npz from compute_deltaet_target.py")
    ap.add_argument("--catalogue", required=True, help="constgold cert catalogue (coords only)")
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--isolated-zero", action="store_true",
                    help="set value=0 for neighbored=False objects (BLEND grid: an isolated object "
                         "has no neighbour, hence no neighbour-shear leakage). Omit for the SELF grid.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.grid, allow_pickle=True)
    Rsim = z["Rsim"].astype(float)
    ef = z["edges_flux"].astype(float)
    es = z["edges_size"].astype(float)
    ed = z["edges_dist"].astype(float)
    nf, ns, nblend = Rsim.shape
    n_dist = nblend - 1
    gm = float(z["global_R"]) if "global_R" in z.files else float(np.nanmean(Rsim))
    Rsim = np.where(np.isfinite(Rsim), Rsim, gm)
    print(f"grid {nf}x{ns}x{nblend} <R>={gm:.4f} from {args.grid}")

    need = ["case", "input_index", "r_input_p", "Re_input_p", "distance", "neighbored"]
    cases, idxs, vals = [], [], []
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names)
        missing = set(need) - avail
        if missing:
            raise KeyError(f"catalogue missing columns: {sorted(missing)}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(need).to_pandas()
            if args.min_case is not None:
                b = b[b["case"] >= args.min_case]
            if args.max_case is not None:
                b = b[b["case"] <= args.max_case]
            if len(b) == 0:
                continue
            flux = b["r_input_p"].to_numpy(float)
            size = b["Re_input_p"].to_numpy(float)
            nbflag = b["neighbored"].astype(bool).to_numpy()
            dist = b["distance"].to_numpy(float)
            fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
            si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
            if nblend == 1:                       # pure flux x size grid (SELF): every object -> bin 0
                di = np.zeros(len(flux), dtype=int)
            else:                                 # isolated bin 0 + distance quantiles (BLEND)
                di = np.where(nbflag, 1 + np.clip(np.digitize(dist, ed) - 1, 0, n_dist - 1), 0)
            v = Rsim[fi, si, di]
            if args.isolated_zero:
                v = np.where(nbflag, v, 0.0)      # isolated -> no neighbour-shear leakage
            cases.append(b["case"].to_numpy(np.int64))
            idxs.append(b["input_index"].to_numpy(np.int64))
            vals.append(v.astype(float))
    case = np.concatenate(cases); idx = np.concatenate(idxs); val = np.concatenate(vals)
    np.savez(args.out, case=case, input_index=idx, value=val)
    print(f"harvested {len(val):,} objects, cases [{case.min()},{case.max()}], "
          f"<value>={np.mean(val):.4f} range=[{val.min():.4f},{val.max():.4f}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
