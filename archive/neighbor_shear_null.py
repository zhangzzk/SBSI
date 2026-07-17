"""NULL TEST for the all-pairs self-response caveat (user Point 3): a sheared secondary target
often has a *secondary* neighbour that is ALSO sheared (random direction). Its blended light
injects a shear into the target's measured shape. Assumption: over the ensemble this averages
to noise and does not bias the self-response R_sim.

Test on the g=0.05 all-pairs catalogue, restricted to pairs whose NEIGHBOUR is itself sheared
(|gamma_s|>0). Bin the target response proj = <e_target . ghat_target>/g by the target<->neighbour
shear-axis alignment a = cos(2*dphi) (a=+1 aligned, -1 anti, 0 orthogonal):
  - if the neighbour shear leaks, R_sim rises with a (aligned neighbour boosts the target response);
  - the caveat is BENIGN iff the alignment distribution is uniform (so the population mean is the
    a=0 value) -> report the alignment histogram and the count-weighted mean vs the a~0 bin.
"""
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=20_000_000)
    args = ap.parse_args()

    need = ["measured_ngmix_g1", "measured_ngmix_g2", "gamma1_input_p", "gamma2_input_p",
            "gamma1_input_s", "gamma2_input_s", "detected", "input_index", "case"]
    proj = []; align = []; n = 0
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names)
        cols = [c for c in need if c in avail]
        miss = [c for c in need if c not in avail]
        if miss:
            print(f"WARNING missing columns: {miss}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            gp1 = b["gamma1_input_p"].to_numpy(float); gp2 = b["gamma2_input_p"].to_numpy(float)
            gs1 = b["gamma1_input_s"].to_numpy(float); gs2 = b["gamma2_input_s"].to_numpy(float)
            gpm = np.hypot(gp1, gp2); gsm = np.hypot(gs1, gs2)
            e1 = b["measured_ngmix_g1"].to_numpy(float); e2 = b["measured_ngmix_g2"].to_numpy(float)
            ok = (gpm > 1e-6) & (gsm > 1e-6) & np.isfinite(e1)   # target & neighbour both sheared
            if "detected" in b:
                ok &= b["detected"].astype(bool).to_numpy()
            gh1, gh2 = gp1[ok] / gpm[ok], gp2[ok] / gpm[ok]
            p = (e1[ok] * gh1 + e2[ok] * gh2) / args.nominal_g
            # spin-2 alignment cos(2 dphi) = ghat_p . ghat_s
            a = (gp1[ok] / gpm[ok]) * (gs1[ok] / gsm[ok]) + (gp2[ok] / gpm[ok]) * (gs2[ok] / gsm[ok])
            proj.append(p); align.append(a); n += ok.sum()
            if n >= args.max_rows:
                break
    proj = np.concatenate(proj); align = np.concatenate(align)
    print(f"pairs with target & neighbour both sheared: {len(proj):,}")
    print(f"alignment a=cos(2dphi): mean={align.mean():+.4f} (should be ~0 = uniform/independent)")
    # uniformity of alignment
    h, edges = np.histogram(align, bins=args.nbins, range=(-1, 1))
    print("alignment histogram (uniform => flat):")
    for i in range(args.nbins):
        print(f"  a[{edges[i]:+.2f},{edges[i+1]:+.2f}): frac={h[i]/len(align):.3f}  R_sim={proj[(align>=edges[i])&(align<edges[i+1])].mean():.3f}")
    # slope of R_sim vs alignment (blend coupling) and the population mean vs a~0
    A = np.c_[np.ones_like(align), align]
    coef, *_ = np.linalg.lstsq(A, proj, rcond=None)
    print(f"\nR_sim(a) linear fit: intercept(a=0)={coef[0]:.4f}  slope={coef[1]:+.4f}")
    print(f"population mean R_sim = {proj.mean():.4f}   (BENIGN iff ~= intercept, i.e. alignment mean ~0)")
    print(f"=> neighbour-shear bias on R_sim ~ slope*<a> = {coef[1]*align.mean():+.5f} "
          f"({coef[1]*align.mean()/max(abs(proj.mean()),1e-6)*100:+.2f}% of R_sim)")


if __name__ == "__main__":
    main()
