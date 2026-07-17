"""Verify the blend structure of the half render as USED by the flow (--targets secondaries).

The measured objects are the sheared SECONDARIES (2nd half of the input list, |g|>0).
Question: are their NEIGHBOURS unsheared PRIMARIES (|g|=0), or sheared secondaries, or a mix?
And if a neighbour is a secondary, is its shear direction correlated with the target's?

For each sheared secondary, find its nearest input neighbour and report:
  - fraction of nearest neighbours that are primaries (|g|=0) vs secondaries (|g|>0)
  - for secondary neighbours: mean cos(2*dphi) between target & neighbour shear axes (~0 = incoherent)
  - same, weighted by how close the neighbour is (closer neighbours blend more)
"""
import argparse
import numpy as np
import pyarrow.feather as f
from scipy.spatial import cKDTree

HALF = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
TILE = "tile180.0_-0.5"


def analyse(casedir):
    inp = f.read_table(f"{HALF}/{casedir}/real0/catalogues/input/gals_info_{TILE}.feather").to_pandas()
    n = len(inp); half = n // 2
    g1 = inp["gamma1_input"].to_numpy(float); g2 = inp["gamma2_input"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    ra = inp["RA_input"].to_numpy(float); dec = inp["DEC_input"].to_numpy(float)
    dec0 = np.median(dec)
    x = (ra - np.median(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    tree = cKDTree(np.c_[x, y])
    d, idx = tree.query(np.c_[x, y], k=2)      # self + nearest
    nn_dist = d[:, 1]; nn_idx = idx[:, 1]

    is_sec = np.arange(n) >= half              # secondaries = sheared targets
    tgt = is_sec & (gmag > 1e-6)               # sheared secondary targets
    nn = nn_idx[tgt]
    nn_sheared = gmag[nn] > 1e-6               # is the nearest neighbour sheared?
    nn_is_primary = nn < half
    # shear-axis alignment for secondary (sheared) neighbours
    tgt_ang = np.arctan2(g2[tgt], g1[tgt])     # 2*phi of target
    nn_ang = np.arctan2(g2[nn], g1[nn])
    cos2d = np.cos(tgt_ang - nn_ang)           # 1=aligned, 0=random, -1=anti
    sec_nbr = nn_sheared
    return dict(
        n_tgt=tgt.sum(),
        frac_nbr_primary=nn_is_primary.mean(),
        frac_nbr_unsheared=(~nn_sheared).mean(),
        frac_nbr_sheared=nn_sheared.mean(),
        cos2d_shearednbr=(cos2d[sec_nbr].mean() if sec_nbr.any() else np.nan),
        nn_dist_med=np.median(nn_dist[tgt]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shear", default="0.05")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    agg = {}
    for c in args.cases:
        try:
            r = analyse(f"case{c}_{args.shear}")
            for k, v in r.items():
                agg.setdefault(k, []).append(v)
        except Exception as e:
            print(f"  case{c}: SKIP ({type(e).__name__}: {e})")
    print(f"\nHALF @{args.shear}: nearest-neighbour shear status of the SHEARED SECONDARY targets")
    print(f"  targets/case (mean): {np.mean(agg['n_tgt']):,.0f}   median NN distance: {np.mean(agg['nn_dist_med']):.2f}\"")
    print(f"  nearest neighbour is an UNSHEARED primary : {np.mean(agg['frac_nbr_primary'])*100:.1f}%")
    print(f"  nearest neighbour is UNSHEARED (any)      : {np.mean(agg['frac_nbr_unsheared'])*100:.1f}%")
    print(f"  nearest neighbour is SHEARED (secondary)  : {np.mean(agg['frac_nbr_sheared'])*100:.1f}%")
    print(f"  for SHEARED neighbours: <cos(dphi2)> shear-axis alignment = {np.nanmean(agg['cos2d_shearednbr']):+.3f}  (0=random/incoherent, 1=coherent)")


if __name__ == "__main__":
    main()
