"""Property-resolved response grid R(flux x size x blend) from a blendemu delta_et catalogue.

TRAIN-ON-SHEARED-SIM, VALIDATE-ON-CONSTGOLD correction (WORKLOG cont.67+).

Unlike compute_response_target_blend.py (which recomputes e.ghat/g from a sheared PAIR
catalogue and, in its --antithetic mode, reads the constgold VALIDATION sample -- the
train-on-validation error), this reads a blendemu half-shear RESPONSE catalogue whose
`delta_et1` is ALREADY the shear-rotated finite-difference response et(sheared)-et(g=0):

  * SELF  response  (self_response_catalogue_train*.feather): target = SHEARED secondary,
    neighbour fixed unsheared -> clean self-response.  g = 0.05.
  * BLEND response  (response_catalogue_train*.feather):      target = UNSHEARED primary,
    neighbour sheared -> clean neighbour-shear leakage.       g = 0.2.

R(cell) = <delta_et1>/g, binned exactly like compute_response_target_blend.py (flux=r_input_p,
size=Re_input_p, blend = isolated bin 0 + neighbour-distance quantiles) so the output npz is
drop-in for train_forward_prototype.py --target-npz AND for harvest_grid_perobj.py.

Constgold is NEVER read here -- it stays the acceptance metric only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True, help="blendemu self/blend delta_et response catalogue")
    ap.add_argument("--nominal-g", type=float, required=True, help="0.05 for self, 0.2 for blend")
    ap.add_argument("--deltaet-col", default="delta_et1", help="pre-projected tangential response column")
    ap.add_argument("--n-flux", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--size-edges", default=None,
                    help="comma-separated explicit size-bin edges (overrides --n-size quantile edges). "
                         "A-priori physics choice to resolve the steep large-size response, not |m|-tuning.")
    ap.add_argument("--n-dist", type=int, default=3, help="distance bins for BLENDED gals (isolated = bin 0)")
    ap.add_argument("--min-count", type=float, default=200,
                    help="cells with fewer EFFECTIVE (unique-target, sum-of-weights) galaxies fall back to global R.")
    ap.add_argument("--max-case", type=int, default=None, help="keep only case <= max-case")
    ap.add_argument("--max-rows", type=int, default=0, help="read at most this many raw rows (0=all)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # `neighbored` is optional: the blend catalogue (all rows are primary-neighbour PAIRS) omits it;
    # when absent every row is treated as neighboured (which it is, by construction).
    need = {args.deltaet_col, "r_input_p", "Re_input_p", "distance", "input_index", "case"}
    parts = []
    nread = 0
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names)
        missing = need - avail
        if missing:
            raise KeyError(f"catalogue missing columns: {sorted(missing)}")
        has_nbr = "neighbored" in avail
        cols = sorted(need | ({"neighbored"} if has_nbr else set()))
        for bi in range(r.num_record_batches):
            if args.max_rows and nread >= args.max_rows:
                break
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if not has_nbr:
                b["neighbored"] = True          # blend catalogue: every row is a pair
            nread += len(b)
            if args.max_case is not None:
                b = b[b["case"] <= args.max_case]
                if len(b) == 0:
                    continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            parts.append(b.reset_index(drop=True))
    df = pd.concat(parts, ignore_index=True)

    # Per-(case,target) weighting: blend catalogues are ALL-PAIRS (a primary appears once per
    # neighbour) -> weight each pair-row 1/n_pairs so each (case,target) contributes as ONE
    # independent measurement. Self catalogues are one-row-per-target -> n_pairs=1 -> weights 1.
    ii = df["input_index"].to_numpy(np.int64)
    key = df["case"].to_numpy(np.int64) * 1_000_003 + ii
    _, inv, npairs = np.unique(key, return_inverse=True, return_counts=True)
    w = (1.0 / npairs[inv]).astype(float)

    proj = df[args.deltaet_col].to_numpy(float)        # ALREADY et(sheared)-et(g=0), shear-rotated
    flux = df["r_input_p"].to_numpy(float)
    size = df["Re_input_p"].to_numpy(float)
    nbflag = df["neighbored"].astype(bool).to_numpy()
    dist = df["distance"].to_numpy(float)
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(proj)
    flux, size, proj, nbflag, dist, w = flux[fin], size[fin], proj[fin], nbflag[fin], dist[fin], w[fin]

    g = args.nominal_g
    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    if args.size_edges:
        es = np.array([float(x) for x in args.size_edges.split(",")], dtype=float)
        es = np.sort(es); es[0] -= 1e-6; es[-1] += 1e-6
        args.n_size = len(es) - 1
        print(f"custom size edges ({args.n_size} bins): {np.round(es, 4).tolist()}")
    else:
        es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    if args.n_dist == 0:
        # pure flux x size (nblend=1). Used for the SELF grid: the self catalogue has NO isolated
        # rows (100% neighboured within 3"), so blend/distance is marginalised and the constgold
        # self-response is looked up by flux x size alone.
        ed = np.array([0.0, 1.0])
        di = np.zeros(len(flux), dtype=int)
        nblend = 1
    else:
        # isolated bin 0 + neighbour-distance quantiles. Used for the BLEND grid.
        db = dist[nbflag & np.isfinite(dist)]
        ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
        di = np.where(nbflag, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
        nblend = args.n_dist + 1

    Rsim = np.full((args.n_flux, args.n_size, nblend), np.nan)
    cnt = np.zeros((args.n_flux, args.n_size, nblend), dtype=np.float64)
    rawcnt = np.zeros((args.n_flux, args.n_size, nblend), dtype=np.int64)
    gm = float(np.average(proj, weights=w)) / g
    for a in range(args.n_flux):
        for b in range(args.n_size):
            for c in range(nblend):
                m = (fi == a) & (si == b) & (di == c)
                rawcnt[a, b, c] = int(m.sum())
                cnt[a, b, c] = float(w[m].sum())
                if cnt[a, b, c] > args.min_count:
                    Rsim[a, b, c] = float(np.average(proj[m], weights=w[m])) / g
    Rsim = np.where(np.isfinite(Rsim), Rsim, gm)

    np.savez(args.output, edges_flux=ef, edges_size=es, edges_dist=ed, Rsim=Rsim,
             counts=cnt, raw_counts=rawcnt, nominal_g=g, global_R=gm, n_dist=args.n_dist,
             crowd_col="", edges_crowd=ed, response_estimator="deltaet",
             deltaet_col=args.deltaet_col, catalogue=os.path.basename(args.catalogue),
             max_case=(-1 if args.max_case is None else args.max_case))
    print(f"R(flux x size x blend) from {args.deltaet_col}/g, g={g}, "
          f"N_pairs={len(proj):,}, N_eff(unique-target)={w.sum():.0f}, global R={gm:.4f}")
    print(f"grid {args.n_flux}x{args.n_size}x{nblend} (blend bin 0=isolated, then by distance)")
    for c in range(nblend):
        tag = "ISOLATED" if c == 0 else f"blend d-bin {c}"
        mincell = int(np.nanmin(cnt[:, :, c])) if np.isfinite(cnt[:, :, c]).any() else 0
        print(f"  {tag}: R {np.nanmin(Rsim[:, :, c]):.3f}..{np.nanmax(Rsim[:, :, c]):.3f}  "
              f"(N_eff={cnt[:, :, c].sum():.0f}, min-cell N_eff={mincell})")
    print(f"R spans {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
