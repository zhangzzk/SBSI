"""COHERENT response target R_sim(flux x size x crowd) from the CONSTANT (gold) render, for a
cross-validated response-aware calibration of the coherent response (self + collective blend).

R_sim = <(e_+g - e_-g).ghat> / (2g)   (two-sided antithetic; cancels intrinsic shape + noise).
Bin by true flux x size x crowding (r_blend). Use --max-case to hold out cases for validation
(e.g. target on cases 0-19, validate the trained flow on 20-39): NOT circular m-removal, a
per-property response calibration cross-validated across noise realisations.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CBASE + "constant_response_catalogue_train.feather")
    ap.add_argument("--blend-lookup", required=True, help="per-(case,input_index) R_blend (crowding axis)")
    ap.add_argument("--min-case", type=int, default=None)
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--n-flux", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--n-crowd", type=int, default=5)
    ap.add_argument("--min-count", type=float, default=200)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    need = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "distance", "input_index", "case",
            "axis_ratio_input_p", "position_angle_input_p", "r_input_p", "Re_input_p"]
    parts = []
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names); cols = [c for c in need if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if args.min_case is not None:
                b = b[b["case"] >= args.min_case]
            if args.max_case is not None:
                b = b[b["case"] <= args.max_case]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)

    rb = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
    crowd = df.merge(rb, on=["case", "input_index"], how="left")["R_blend"].fillna(0.0).to_numpy(float)

    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gm = np.hypot(df["applied_g1"], df["applied_g2"]).to_numpy(float)
    gh1 = df["applied_g1"].to_numpy(float) / gm; gh2 = df["applied_g2"].to_numpy(float) / gm
    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    proj = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / 2.0     # /g done below to match R_sim=proj/g
    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(proj) & np.isfinite(crowd)
    flux, size, proj, crowd = flux[fin], size[fin], proj[fin], crowd[fin]
    w = np.ones(len(flux))

    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    ec = np.quantile(crowd, np.linspace(0, 1, args.n_crowd + 1)); ec[0] -= 1e-9; ec[-1] += 1e-9
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    ci = np.clip(np.digitize(crowd, ec) - 1, 0, args.n_crowd - 1)
    nf, ns, nc = args.n_flux, args.n_size, args.n_crowd
    Rsim = np.full((nf, ns, nc), np.nan); cnt = np.zeros((nf, ns, nc))
    gmr = float(np.average(proj, weights=w)) / g
    for a in range(nf):
        for b in range(ns):
            for c in range(nc):
                m = (fi == a) & (si == b) & (ci == c)
                cnt[a, b, c] = m.sum()
                if m.sum() > args.min_count:
                    Rsim[a, b, c] = float(np.average(proj[m], weights=w[m])) / g
    Rsim = np.where(np.isfinite(Rsim), Rsim, gmr)
    np.savez(args.output, edges_flux=ef, edges_size=es, edges_dist=ec, edges_crowd=ec, Rsim=Rsim,
             counts=cnt, raw_counts=cnt.astype(np.int64), nominal_g=g, global_R=gmr, n_dist=nc,
             crowd_col="r_blend")
    print(f"COHERENT target from constant, |g|={g:.4f}, N={len(flux):,}, global R={gmr:.4f}, grid {nf}x{ns}x{nc}")
    print(f"R_sim spans {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, min-cell N={int(cnt[cnt>0].min())}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
