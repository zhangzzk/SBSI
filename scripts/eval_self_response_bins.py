#!/usr/bin/env python
"""Figure 5 GPU eval: flow SELF-response R_model per flux/size bin on the g=0
training/val catalogue.

R_model is the central-secant trace/2 self-response the training response-loss
targets (epoch_response in train_measurement_model.py, delta=0.02,
response_difference="central"), evaluated via the SAME validated forward path as
the certified R_flow harvest (response_ratio_diagnostic.model_mean_proj /
bundle.sample, CRN-seeded so the sampling noise cancels in the +/-delta legs):

  R_model(obj) = 0.5 * [ (proj(+d,e1) - proj(-d,e1)) + (proj(+d,e2) - proj(-d,e2)) ] / (2 d)

Binned (uniform weights, matching compute_response_target_constant.py) by
r_input_p (flux) and Re_input_p (size) using the edges of the response target
npz.  Writes a small npz consumed by plotting/plot_flow_figures.py figure5().

Bounded GPU job -- do NOT run the full pass on the login node.
"""
from __future__ import annotations

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

import torch  # noqa: E402
from sbs_shear.measurement_model import (  # noqa: E402
    load_measurement_model,
    raw_columns_for_measurement_targets,
)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa: E402

CAT = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
       "det_meas_crowd_conc_g0.0_train_full.feather")
TARGET_NPZ = "results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz"


def _seed_flow(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_g0(bundle, max_rows, seed):
    """Load the g=0 catalogue: standard source cuts + detected, keep intrinsic
    shape (rot0), conditioning raw columns, and the flux/size/crowd bin columns.
    No shear-threshold cut (this is the g=0 render)."""
    rng = np.random.default_rng(seed)
    cond = bundle.condition_preprocessor.feature_names
    tgt = bundle.target_transform.target_names
    with ipc.open_file(CAT) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(cond, available_columns=available)
        needed |= raw_columns_for_measurement_targets(tgt)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p",
                   "e1_input_rot0_p", "e2_input_rot0_p",
                   "r_input_p", "Re_input_p", "r_blend", "shear_case"}
        cols = sorted(c for c in needed if c in available)
        reservoir = None
        raw = 0
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            if "shear_case" in b.columns:
                b = b[b["shear_case"] == 0.0]
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b) == 0:
                continue
            b = b.copy()
            b["__key"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    if reservoir is None:
        raise SystemExit("No g=0 rows selected")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    reservoir["gamma1_input_p"] = 0.0
    reservoir["gamma2_input_p"] = 0.0
    print(f"  raw scanned={raw:,}  kept (g=0, detected)={len(reservoir):,}", flush=True)
    return reservoir


def proj_perobj(bundle, base, s, gh1, gh2, intrinsic, rk, n_samples, batch_size):
    """Per-object induced first-moment projection onto (gh1,gh2) after applying
    S_{s*ghat} to the intrinsic shape.  Per-object version of model_mean_proj."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * gh1, s * gh2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rk)
    draws = bundle.sample(frame, n_samples=n_samples, batch_size=batch_size)
    mean = draws.mean(axis=1)
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    return mean[:, i1] * gh1 + mean[:, i2] * gh2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model",
                    default="models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt")
    ap.add_argument("--output", default="results/fig5_selfresp_bins_s501.npz")
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    bundle = load_measurement_model(args.measurement_model, device=device)

    z = np.load(TARGET_NPZ)
    ef, es = z["edges_flux"], z["edges_size"]
    nf, ns = len(ef) - 1, len(es) - 1

    df = load_g0(bundle, args.max_rows, args.seed)
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(),
            df["e2_input_rot0_p"].to_numpy(float).copy())
    N = len(df)
    gh1 = np.ones(N); gh0 = np.zeros(N)
    d = args.delta

    # CRN: same flow seed for the +/- legs of each axis so sampling noise cancels.
    _seed_flow(args.flow_seed); pp1 = proj_perobj(bundle, df, +d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm1 = proj_perobj(bundle, df, -d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pp2 = proj_perobj(bundle, df, +d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm2 = proj_perobj(bundle, df, -d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    r_model = 0.5 * ((pp1 - pm1) + (pp2 - pm2)) / (2 * d)   # trace/2 central secant
    print(f"  <R_model> global = {np.mean(r_model):.4f}", flush=True)

    flux = df["r_input_p"].to_numpy(float)
    size = df["Re_input_p"].to_numpy(float)
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(r_model)
    flux, size, rm = flux[fin], size[fin], r_model[fin]
    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)

    Rmodel_flux = np.full(nf, np.nan); n_flux = np.zeros(nf, np.int64)
    for a in range(nf):
        m = fi == a
        n_flux[a] = int(m.sum())
        if m.any():
            Rmodel_flux[a] = float(np.mean(rm[m]))
    Rmodel_size = np.full(ns, np.nan); n_size = np.zeros(ns, np.int64)
    for b in range(ns):
        m = si == b
        n_size[b] = int(m.sum())
        if m.any():
            Rmodel_size[b] = float(np.mean(rm[m]))

    np.savez(args.output, Rmodel_flux=Rmodel_flux, Rmodel_size=Rmodel_size,
             n_flux=n_flux, n_size=n_size, edges_flux=ef, edges_size=es,
             delta=d, n_rows_used=int(fin.sum()), seed=args.flow_seed,
             model=os.path.basename(args.measurement_model))
    print(f"wrote {args.output}", flush=True)
    print("Rmodel_flux:", np.round(Rmodel_flux, 4).tolist())
    print("Rmodel_size:", np.round(Rmodel_size, 4).tolist())
    print("FIG5_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
