"""Diagnose the measured-shape response matrix M = d(measured e)/d(intrinsic e).

The held-out-shear recovery under-recovers by ~0.74x, sharp and S/N-robust.  The
recovery's sensitivity to shear is M . J_Sgamma, where M is how the measured image
ellipticity responds to the intrinsic (conditioning) ellipticity.  This script measures M
two ways on the SAME g=0 sample:

  * M_data : least-squares regression of measured_e1/e2_image on e1/e2_input_p (the true
    physical response, including PSF dilution and the sky->image rotation), and
  * M_model: finite-difference of the trained flow's predicted mean measured shape w.r.t.
    e1/e2_input_p.

If M_model ~= M_data, the flow learned the shape response correctly and the recovery bias
must come from S_gamma (the Mobius Jacobian) or the omitted selection factor.  If they
differ, the flow itself under/over-responds.  |M| ~ 0.3-0.5 is the expected PSF-diluted
responsivity (cf blendemu delta_et/gamma).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features,
    load_measurement_model,
    raw_columns_for_measurement_targets,
)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)


def load_g0_sample(args, bundle):
    cond = bundle.condition_preprocessor.feature_names
    tgt = bundle.target_transform.target_names
    rng = np.random.default_rng(args.seed)
    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        needed = set(raw_columns_for_selection_features(cond, available_columns=available))
        needed |= raw_columns_for_measurement_targets(tgt)
        needed |= {"detected", "r_input_p", "Re_input_p", "distance", "neighbored",
                   "gamma1_input_p", "gamma2_input_p"}
        read_columns = sorted(c for c in needed if c in available)
        reservoir = None
        for bi in range(reader.num_record_batches):
            if args.max_read_batches is not None and bi >= args.max_read_batches:
                break
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)]
            # g=0 only: keep unsheared rows (gamma ~ 0) so intrinsic == rendered
            g = np.hypot(b["gamma1_input_p"].to_numpy(float), b["gamma2_input_p"].to_numpy(float))
            b = b[g <= 1e-6]
            if len(b) == 0:
                continue
            b = b.copy(); b["__k"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * args.max_rows:
                reservoir = reservoir.nlargest(args.max_rows, "__k").reset_index(drop=True)
    reservoir = reservoir.nlargest(args.max_rows, "__k").reset_index(drop=True).drop(columns="__k")
    return reservoir


def regress_M(frame):
    """Least-squares M, c so that measured_e ~= M @ e_input_p + c."""
    f = add_measurement_target_features(frame.copy())
    f = rescale(f)  # produces e1_input_p, e2_input_p
    keep = np.isfinite(f["measured_e1_image"]) & np.isfinite(f["measured_e2_image"]) \
        & np.isfinite(f["e1_input_p"]) & np.isfinite(f["e2_input_p"])
    f = f[keep]
    X = np.column_stack([f["e1_input_p"], f["e2_input_p"], np.ones(len(f))])
    Y = np.column_stack([f["measured_e1_image"], f["measured_e2_image"]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (3,2)
    M = coef[:2, :].T  # rows = measured comp, cols = input comp
    c = coef[2, :]
    return M, c, len(f)


def flow_M(bundle, frame, delta=0.02, n_samples=64):
    """Finite-difference d E_flow[measured e]/d(e1/e2_input_p) averaged over the sample."""
    base = rescale(frame.copy())
    e1 = base["e1_input_p"].to_numpy(float); e2 = base["e2_input_p"].to_numpy(float)
    ti = {n: i for i, n in enumerate(bundle.target_transform.target_names)}
    je1, je2 = ti["measured_e1_image"], ti["measured_e2_image"]

    def mean_meas(de1, de2):
        f = frame.copy()
        # perturb the raw intrinsic shape so rescale regenerates e1/e2_input_p consistently
        f["e1_input_rot0_p"] = f["e1_input_rot0_p"].to_numpy(float) + de1
        f["e2_input_rot0_p"] = f["e2_input_rot0_p"].to_numpy(float) + de2
        f = rescale(f)
        s = bundle.sample(f, n_samples=n_samples).mean(axis=1)  # (N,6) engineered units
        return s[:, je1].mean(), s[:, je2].mean()

    m1p = mean_meas(+delta, 0.0); m1m = mean_meas(-delta, 0.0)
    m2p = mean_meas(0.0, +delta); m2m = mean_meas(0.0, -delta)
    M = np.array([
        [(m1p[0] - m1m[0]) / (2 * delta), (m2p[0] - m2m[0]) / (2 * delta)],
        [(m1p[1] - m1m[1]) / (2 * delta), (m2p[1] - m2m[1]) / (2 * delta)],
    ])
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement-model", default="SBSI/models/measurement_flow_g0_oriented_v1.pt")
    ap.add_argument("--catalogue", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather")
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--max-read-batches", type=int, default=400)
    ap.add_argument("--flow-rows", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    print(f"Device {device}; model targets {bundle.target_transform.target_names}")

    frame = load_g0_sample(args, bundle)
    print(f"g=0 sample rows: {len(frame):,}")

    M_data, c_data, n = regress_M(frame)
    print("\n=== M_data (regression measured_e <- e_input_p), n=%d ===" % n)
    print(np.array2string(M_data, precision=4))
    print(f"  offset c = {np.round(c_data, 4)}")
    print(f"  |M_data| diag mean = {0.5*(M_data[0,0]+M_data[1,1]):.4f}  "
          f"off-diag mean = {0.5*(M_data[0,1]+M_data[1,0]):.4f}")

    sub = frame.iloc[:args.flow_rows].reset_index(drop=True)
    M_model = flow_M(bundle, sub, delta=args.delta, n_samples=args.n_samples)
    print(f"\n=== M_model (flow finite-diff), rows={len(sub):,}, delta={args.delta} ===")
    print(np.array2string(M_model, precision=4))
    print(f"  |M_model| diag mean = {0.5*(M_model[0,0]+M_model[1,1]):.4f}  "
          f"off-diag mean = {0.5*(M_model[0,1]+M_model[1,0]):.4f}")

    ratio = 0.5 * (M_model[0, 0] + M_model[1, 1]) / max(1e-6, 0.5 * (M_data[0, 0] + M_data[1, 1]))
    print(f"\n=== M_model/M_data diagonal ratio = {ratio:.3f} ===")
    print("  ~1 => flow learned the shape response correctly (bias is S_gamma/selection);")
    print("  <1 => flow under-responds; >1 => flow over-responds.")


if __name__ == "__main__":
    main()
