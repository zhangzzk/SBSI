"""Why is R_model frozen across the population even though the flow conditions on
flux/size/neighbour?  Direct check at the mean-head level (no big catalogue):

  response (per galaxy) = d mu / d(shape feature) = the mean head's Jacobian wrt e1/e2_input_p.

For a LINEAR head mu=W.c+b this Jacobian is the constant weight column W[:,shape] -- identical
for every galaxy BY CONSTRUCTION (a linear model has no shape x flux / shape x size interaction).
For an MLP head the Jacobian depends on the other inputs, so it can vary across flux/size.
We print the linear head's shape weights (the smoking gun) and the per-(flux,size)-bin Jacobian
for both a linear and an MLP model.  If the MLP varies and the linear is flat, the freeze is the
architecture, not a bug.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn as nn

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model, add_measurement_target_features  # noqa
from sbs_shear.preprocessing import (DEFAULT_SELECTION_CUTS, rescale,  # noqa
                                      raw_columns_for_selection_features, source_select_selection)

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather"


def small_sample(bundle, n_batches=20):
    feats = bundle.condition_preprocessor.feature_names
    with ipc.open_file(CAT) as r:
        avail = set(r.schema.names)
        need = raw_columns_for_selection_features(feats, available_columns=avail) | {
            "detected", "r_input_p", "Re_input_p"}
        cols = sorted(c for c in need if c in avail)
        parts = []
        for bi in range(min(n_batches, r.num_record_batches)):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            b = b[b["detected"].astype(bool)]
            parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    df = rescale(df, pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    return df


def mean_head_response(bundle, df, sidx):
    """per-galaxy standardized mean-head Jacobian 0.5*(d mu_e1/d ctx[s0] + d mu_e2/d ctx[s1])."""
    ctx = torch.as_tensor(bundle.condition_preprocessor.transform_frame(df), dtype=torch.float32)
    ctx.requires_grad_(True)
    mu = bundle.model._mu(ctx)
    g0 = torch.autograd.grad(mu[:, 0].sum(), ctx, retain_graph=True)[0][:, sidx[0]]
    g1 = torch.autograd.grad(mu[:, 1].sum(), ctx, retain_graph=True)[0][:, sidx[1]]
    return (0.5 * (g0 + g1)).detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-model", default="SBSI/models/measurement_flow_g0_shape2d_resp_lam1000_v1.pt")
    ap.add_argument("--mlp-model", default="/tmp/smoke_binned.pt")
    ap.add_argument("--n-flux", type=int, default=3)
    ap.add_argument("--n-size", type=int, default=3)
    args = ap.parse_args()

    blin = load_measurement_model(args.linear_model, device="cpu")
    feats = blin.condition_preprocessor.feature_names
    si = [feats.index("e1_input_p"), feats.index("e2_input_p")]
    print(f"shape feature indices in context: {si}  (features {feats[si[0]]}, {feats[si[1]]})")

    print("\n--- LINEAR head: shape-weight columns ARE the response (one constant for all galaxies) ---")
    W = blin.model.mean_net.weight.detach().numpy()  # (2, 34)
    print(f"  mean_net is Linear: {isinstance(blin.model.mean_net, nn.Linear)}")
    print(f"  W[:, e1_idx={si[0]}] = {W[:, si[0]]}   W[:, e2_idx={si[1]}] = {W[:, si[1]]}")
    print("  -> d mu / d(shape) does NOT depend on flux/size/neighbour: NO interaction term exists.")

    df = small_sample(blin)
    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si_ = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    print(f"\n  sample N={len(df):,}")

    for tag, path in (("LINEAR (resp-aware lam1000)", args.linear_model),
                      ("MLP    (smoke binned)", args.mlp_model)):
        if not os.path.exists(path):
            print(f"\n[{tag}] model {path} not found, skip"); continue
        b = load_measurement_model(path, device="cpu")
        if b.condition_preprocessor.feature_names != feats:
            print(f"\n[{tag}] feature set differs, skip"); continue
        resp = mean_head_response(b, df, si)
        print(f"\n[{tag}] mean-head response (standardized units) per flux x size bin:")
        print("   flux\\size " + "  ".join(f"s{j}" for j in range(args.n_size)))
        cell = np.full((args.n_flux, args.n_size), np.nan)
        for a in range(args.n_flux):
            for c in range(args.n_size):
                m = (fi == a) & (si_ == c)
                if m.sum() > 50:
                    cell[a, c] = resp[m].mean()
            print(f"    f{a}   " + "  ".join(f"{cell[a,c]:+.4f}" for c in range(args.n_size)))
        vals = cell[np.isfinite(cell)]
        print(f"   spread across bins: {vals.min():+.4f}..{vals.max():+.4f}  "
              f"(max/min = {abs(vals.max()/vals.min()):.2f})  "
              f"-> {'VARIES (resolves)' if abs(vals.max()/vals.min())>1.3 else 'FLAT (frozen)'}")


if __name__ == "__main__":
    main()
