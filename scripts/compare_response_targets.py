"""Confirm-first curvature test for the flow's response-supervision target.

Two SNC secants from zero, R(g) = <[e(g)-e(0)].ghat>/g per (flux x size x r_blend) cell, are EQUAL
iff the true self-response is linear in shear over [0, g]. So comparing the g=0.02 target to the
existing g=0.05 target, cell by cell, directly tests the linearity assumption baked into training:

  agree within noise  -> response is linear over [0, 0.05]; the 0.05 target is already SNR-optimal and
                         the amplitude mismatch (target@0.05 vs supervision/validation@0.02) is a
                         non-issue -> DO NOT bother retraining at g=0.02 (it only adds target noise).
  differ systematically-> curvature is real; the amplitude fix is justified -> proceed to the retrain
                         (job_train_conc_g02tgt.sh) and eat/offset the extra target noise.

The two targets bin on shear-INDEPENDENT axes (true flux, true size, emulator r_blend), so for the
same 100 cases their edges should coincide and cells are directly comparable. Usage:

  python scripts/compare_response_targets.py \
    --a results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz \
    --b results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="g=0.02 target npz")
    ap.add_argument("--b", required=True, help="g=0.05 target npz (existing)")
    ap.add_argument("--rel-thresh", type=float, default=0.05,
                    help="flag cells whose |R_a-R_b|/|R_b| exceeds this")
    args = ap.parse_args()

    A = np.load(args.a); B = np.load(args.b)
    Ra, Rb = A["Rsim"], B["Rsim"]
    ca, cb = A["counts"], B["counts"]
    ga, gb = float(A["nominal_g"]), float(B["nominal_g"])
    print(f"A = {args.a}\n    nominal_g={ga}  shape={Ra.shape}  global_R={float(A['global_R']):.4f}")
    print(f"B = {args.b}\n    nominal_g={gb}  shape={Rb.shape}  global_R={float(B['global_R']):.4f}")

    if Ra.shape != Rb.shape:
        print(f"!! shape mismatch {Ra.shape} vs {Rb.shape} -- cannot compare cell-by-cell"); return
    for k in ("edges_flux", "edges_size", "edges_crowd"):
        if k in A.files and k in B.files and not np.allclose(A[k], B[k], rtol=0.02, atol=1e-3):
            print(f"!! WARNING {k} differ between targets -> cells are NOT aligned:")
            print(f"     A[{k}]={np.array2string(A[k], precision=3)}")
            print(f"     B[{k}]={np.array2string(B[k], precision=3)}")

    diff = Ra - Rb
    rel = np.where(np.abs(Rb) > 1e-6, diff / np.abs(Rb), np.nan)
    w = np.minimum(ca, cb).astype(float)              # weight by the thinner cell
    good = np.isfinite(rel) & (w > 0)
    wmean_absrel = float(np.average(np.abs(rel[good]), weights=w[good]))
    wmean_rel = float(np.average(rel[good], weights=w[good]))

    print(f"\nper-cell R(0.02) - R(0.05):  count-weighted <|rel|>={wmean_absrel:.3%}   "
          f"<rel(signed)>={wmean_rel:+.3%}   (n_cells={good.sum()}/{Ra.size})")
    print("  <|rel|> small & <rel> ~0  => LINEAR (skip retrain);  systematic sign => CURVATURE (retrain)\n")

    print(f"  {'cell (f,s,c)':>14} {'R_0.02':>9} {'R_0.05':>9} {'diff':>9} {'rel':>8} {'N_02':>9} {'N_05':>9}")
    flagged = 0
    for idx in np.ndindex(Ra.shape):
        r = rel[idx]
        flag = np.isfinite(r) and abs(r) > args.rel_thresh
        flagged += int(flag)
        mark = "  <==" if flag else ""
        print(f"  {str(idx):>14} {Ra[idx]:>9.4f} {Rb[idx]:>9.4f} {diff[idx]:>+9.4f} "
              f"{(r if np.isfinite(r) else float('nan')):>+8.2%} {ca[idx]:>9,.0f} {cb[idx]:>9,.0f}{mark}")
    print(f"\n{flagged}/{Ra.size} cells exceed |rel|>{args.rel_thresh:.0%}. "
          f"If concentrated by sign/severity -> real curvature; if scattered -> target noise.")


if __name__ == "__main__":
    main()
