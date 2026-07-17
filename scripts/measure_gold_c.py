"""Additive bias c on the gold constant render (the dimension the m-analysis ignores).

The constant render is antithetic +/-g, so the shear RESPONSE (∝g) cancels in the mean
(e_plus + e_minus)/2, leaving c + <intrinsic>. Averaged over the detected sample this is the
additive shear bias: fixed-frame c1,c2 (PSF/detector residual + mean intrinsic/IA) and the
shear-frame projection c_par (along g) / c_cross (perpendicular). Stage-IV wants |c|<~5e-4.

Reports fixed-frame and shear-projected c with per-case bootstrap errors, and by true magnitude
(faint galaxies carry more blend/IA). Flow-independent: pure sim truth.
"""
import argparse, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from sbs_shear.preprocessing import source_select_selection, DEFAULT_SELECTION_CUTS  # noqa
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa

CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--max-rows", type=int, default=12000000)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    need = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "case", "r_input_p", "Re_input_p", "distance", "neighbored",
            "axis_ratio_input_p", "position_angle_input_p"]
    parts = []; n = 0
    with ipc.open_file(args.catalogue) as r:
        cols = [c for c in need if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            parts.append(b); n += len(b)
            if n >= args.max_rows:
                break
    df = pd.concat(parts, ignore_index=True)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    print(f"N={len(df):,}  cases={df['case'].nunique()}")

    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    c1 = 0.5 * (e1p + e1m); c2 = 0.5 * (e2p + e2m)          # shear-independent, fixed frame
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gh1 = df["applied_g1"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    gh2 = df["applied_g2"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    c_par = c1 * gh1 + c2 * gh2                              # along shear
    c_cross = -c1 * gh2 + c2 * gh1                           # perpendicular

    case_arr = df["case"].to_numpy(np.int64); ucases = np.unique(case_arr)

    def boot(x):
        per = {c: (float(x[case_arr == c].sum()), int((case_arr == c).sum())) for c in ucases}
        rng = np.random.default_rng(0); vals = []
        for _ in range(args.n_boot):
            pick = rng.choice(ucases, size=len(ucases), replace=True)
            s = sum(per[c][0] for c in pick); nn = sum(per[c][1] for c in pick)
            vals.append(s / nn)
        return float(np.mean(x)), float(np.std(vals))

    print(f"\n|g|={g:.4f}   (Stage-IV target |c| < ~5e-4)")
    for name, x in [("c1 (fixed)", c1), ("c2 (fixed)", c2), ("c_par (‖g)", c_par), ("c_cross (⊥g)", c_cross)]:
        mu, err = boot(x)
        print(f"  {name:>14} = {mu:+.5f} +/- {err:.5f}   ({mu/err:+.1f}σ)")

    print(f"\n--- c_cross by true r-mag (IA/blend grows to the faint end) ---")
    rmag = df["r_input_p"].to_numpy(float)
    for lo, hi in [(18, 24), (24, 25), (25, 26), (26, 28)]:
        msk = (rmag >= lo) & (rmag < hi)
        if msk.sum() < 5000:
            continue
        mu = c_cross[msk].mean(); e = c_cross[msk].std() / np.sqrt(msk.sum())
        print(f"  r∈[{lo},{hi}): c_cross={mu:+.5f} +/- {e:.5f}  c_par={c_par[msk].mean():+.5f}  N={int(msk.sum()):,}")


if __name__ == "__main__":
    main()
