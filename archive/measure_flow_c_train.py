"""Is the flow's mean-head offset a CONSISTENT g=0-measurable property (=> a g=0-derived per-mag
correction legitimately fixes the gold additive bias c), or gold-specific?

Measure, on the g=0 TRAINING catalogue (det_meas_crowd_g0.0), the flow's predicted mean e vs the
ACTUAL measured_ngmix mean, per r-mag bin = the mean-head error. If it matches the gold offsets
(off_c2 -0.0018/-0.0045/-0.0048/-0.0052 by mag), the error is a stable flow property -> correctable
from g=0 alone. Flow at s=0 (no shear), same intrinsic-e handling as validate.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI); sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from sbs_shear.measurement_model import load_measurement_model  # noqa
from sbs_shear.preprocessing import rescale, source_select_selection, DEFAULT_SELECTION_CUTS  # noqa
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa

RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_c0-39.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--model-max-rows", type=int, default=1500000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    need = ["measured_ngmix_g1", "measured_ngmix_g2", "r_input_p", "Re_input_p", "distance",
            "neighbored", "axis_ratio_input_p", "position_angle_input_p", "redshift_input_p",
            "sersic_n_input_p", "Re_input_s", "r_input_s", "axis_ratio_input_s",
            "position_angle_input_s", "sersic_n_input_s", "redshift_input_s",
            "nbr_flux_near", "nbr_flux_far", "case", "input_index"]
    parts = []; n = 0
    with ipc.open_file(args.catalogue) as r:
        cols = [c for c in need if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            parts.append(b); n += len(b)
            if n >= 4 * args.model_max_rows:
                break
    df = pd.concat(parts, ignore_index=True)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    if len(df) > args.model_max_rows:
        df = df.sample(n=args.model_max_rows, random_state=0).reset_index(drop=True)
    for c in ("nbr_flux_near", "nbr_flux_far"):
        if c not in df.columns:
            df[c] = 0.0
    df["r_blend"] = 0.0

    act1 = df["measured_ngmix_g1"].to_numpy(float); act2 = df["measured_ngmix_g2"].to_numpy(float)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_measurement_model(args.measurement_model, device=dev)
    frame = rescale(df.copy(), **RK)
    draws = bundle.sample(frame, n_samples=args.n_samples, batch_size=args.batch_size)
    mean = bundle.target_transform.inverse_transform_array(draws.mean(axis=1))
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    fc1, fc2 = mean[:, i1], mean[:, i2]
    rmag = df["r_input_p"].to_numpy(float)

    print(f"N={len(df):,}  (g=0 TRAINING data: actual = measured_ngmix mean)")
    print(f"  GLOBAL: actual c2={act2.mean():+.5f} flow c2={fc2.mean():+.5f} off_c2={act2.mean()-fc2.mean():+.5f}  "
          f"| actual c1={act1.mean():+.5f} flow c1={fc1.mean():+.5f} off_c1={act1.mean()-fc1.mean():+.5f}")
    print(f"\n{'r-mag':>10} {'act_c2':>8} {'flow_c2':>8} {'off_c2':>8} {'off_c1':>8} {'N':>9}   (compare off to gold)")
    for lo, hi in [(18, 24), (24, 25), (25, 26), (26, 28)]:
        m_ = (rmag >= lo) & (rmag < hi)
        if m_.sum() < 5000:
            continue
        print(f"  [{lo},{hi})   {act2[m_].mean():+8.5f} {fc2[m_].mean():+8.5f} {act2[m_].mean()-fc2[m_].mean():+8.5f} "
              f"{act1[m_].mean()-fc1[m_].mean():+8.5f} {int(m_.sum()):>9,}")
    print("\ngold off_c2 by mag was: -0.00179 / -0.00449 / -0.00475 / -0.00522  (if train matches -> g=0-correctable)")


if __name__ == "__main__":
    main()
