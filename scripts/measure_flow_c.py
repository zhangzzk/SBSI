"""Does the FLOW reproduce the gold's additive bias c? (c1=-0.00016, c2=+0.00386 measured.)

The SBSI likelihood recovers shear against the g=0 flow model. If the flow's predicted mean
ellipticity (at zero shear) matches the sim's c, then c is a MODELED offset the likelihood
absorbs -> the calibration additive RESIDUAL is only sim_c - flow_c (expected ~O(g^2)~4e-4).
If the flow predicts c~0 while the sim has +0.0039, that's an uncorrected additive bias.

Flow c = <E[e_hat | intrinsic, no shear]> (raw mean, not projected). Compared to sim c on the
SAME subsample.
"""
import argparse, os, sys
import numpy as np
import torch

SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI); sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from scripts.validate_constant_with_blend import load  # noqa
from sbs_shear.measurement_model import load_measurement_model  # noqa
from sbs_shear.preprocessing import rescale  # noqa
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa
import pyarrow.feather as pf  # noqa

RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--catalogue", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather")
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_c0-39.feather")
    ap.add_argument("--meas-prim-lookup", default=None,
                    help="join measured_mag_auto/flux_radius/class_star etc. for the realistic szfl flows")
    ap.add_argument("--model-max-rows", type=int, default=1500000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    df = load(args.catalogue, 12000000)
    if len(df) > args.model_max_rows:
        df = df.sample(n=args.model_max_rows, random_state=0).reset_index(drop=True)
    # crowd-flux features for the crowdflux flow
    if os.path.exists(args.crowd_flux_lookup):
        cf = pf.read_table(args.crowd_flux_lookup).to_pandas()
        nbrcols = [c for c in ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"] if c in cf.columns]
        cm = df[["case", "input_index"]].merge(cf[["case", "input_index", *nbrcols]], on=["case", "input_index"], how="left")
        for c in nbrcols:
            df[c] = cm[c].fillna(0.0).to_numpy(float)
    if args.meas_prim_lookup and os.path.exists(args.meas_prim_lookup):
        mp = pf.read_table(args.meas_prim_lookup).to_pandas()
        mcols = [c for c in mp.columns if c.startswith("measured_")
                 and c not in ("measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus")]
        mm = df[["case", "input_index"]].merge(mp[["case", "input_index", *mcols]], on=["case", "input_index"], how="left")
        for c in mcols:  # fill OOD/missing (few % of rows) with the column median for this global-c diagnostic
            col = mm[c].to_numpy(float)
            df[c] = np.where(np.isfinite(col), col, np.nanmedian(col))
    df["r_blend"] = 0.0
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0

    # sim c on this subsample (fixed frame; shear here is along g1 so c1=c_par, c2=c_cross)
    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    sim_c1 = float(np.mean(0.5 * (e1p + e1m))); sim_c2 = float(np.mean(0.5 * (e2p + e2m)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_measurement_model(args.measurement_model, device=dev)
    frame = df.copy()
    frame["e1_input_rot0_p"] = df["e1_input_rot0_p"].to_numpy(float)   # intrinsic, NO shear applied (s=0)
    frame["e2_input_rot0_p"] = df["e2_input_rot0_p"].to_numpy(float)
    frame = rescale(frame, **RK)
    draws = bundle.sample(frame, n_samples=args.n_samples, batch_size=args.batch_size)  # (N, n_samples, dim)
    mean = draws.mean(axis=1)
    mean = bundle.target_transform.inverse_transform_array(mean)  # engineered -> RAW ngmix e units
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    flow_c1 = float(np.mean(mean[:, i1])); flow_c2 = float(np.mean(mean[:, i2]))
    n = len(df)
    print(f"N={n:,}  (Stage-IV |c|<~5e-4)")
    print(f"  sim  c1={sim_c1:+.5f}  c2={sim_c2:+.5f}")
    print(f"  flow c1={flow_c1:+.5f}  c2={flow_c2:+.5f}")
    print(f"  RESIDUAL (sim-flow) c1={sim_c1-flow_c1:+.5f}  c2={sim_c2-flow_c2:+.5f}")

    # is the flow mean-head offset CONSTANT (const re-center fixes it) or magnitude-dependent
    # (needs a property-resolved mean head)? offset = sim - flow per r-mag bin.
    sc1 = 0.5 * (e1p + e1m); sc2 = 0.5 * (e2p + e2m)   # per-object sim c
    fc1 = mean[:, i1]; fc2 = mean[:, i2]               # per-object flow c (raw units)
    rmag = df["r_input_p"].to_numpy(float)
    print(f"\n{'r-mag':>10} {'sim_c2':>8} {'flow_c2':>8} {'off_c2':>8} {'sim_c1':>8} {'flow_c1':>8} {'off_c1':>8} {'N':>9}")
    for lo, hi in [(18, 24), (24, 25), (25, 26), (26, 28)]:
        m_ = (rmag >= lo) & (rmag < hi)
        if m_.sum() < 5000:
            continue
        print(f"  [{lo},{hi})   {sc2[m_].mean():+8.5f} {fc2[m_].mean():+8.5f} {sc2[m_].mean()-fc2[m_].mean():+8.5f} "
              f"{sc1[m_].mean():+8.5f} {fc1[m_].mean():+8.5f} {sc1[m_].mean()-fc1[m_].mean():+8.5f} {int(m_.sum()):>9,}")


if __name__ == "__main__":
    main()
