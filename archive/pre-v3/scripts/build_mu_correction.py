"""Measure the flow's location-vs-e misfit U(e; r-mag cell) on the g0 TRAIN sample.

WORKLOG cont.33: the mid-bright etilde residual decomposes into the flow's location
misfit U(e1) (pushes m NEGATIVE, ~-3%) and the skewed pooled shape of the e-blind
residual flow (pushes POSITIVE, ~+6%), both from mu under-fitting the e-dependent
additive (PSF-direction) structure.  This script measures the location part as a
per-magnitude-cell polynomial

    U_k(e1, e2) = < ehat_k - mu_flow(ctx(e_true)) | e, r-mag cell >,   k = 1,2

on the g0 TRAINING catalogue (true e known; ZERO shear information -> deployable with
the same legitimacy as the OLS mean-freeze).  The population mean over each cell's own
e-sample is SUBTRACTED from the fit: the e-blind residual flow already carries the
per-context mean m_f, so only the e-VARIATION of U is a genuine location correction --
adding the constant part would double-count m_f.

Output: npz with MAG_EDGES, poly design exponents, per-cell coefficients (raw target
units).  Consumed by scripts/infer_posterior_shape.py --mu-correction, which evaluates
it on the e-grid and hands (K,G,2) standardized offsets to the estimator.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SBSI_ROOT = os.path.dirname(SCRIPTS)
for p in (SBSI_ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer_posterior_shape as ips  # noqa: E402
from sbs_shear.measurement_model import (  # noqa: E402
    ConditionalMeanFlowRA, load_measurement_model)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)

# poly design: total degree <= 4 in (e1, e2) -- smooth enough for the measured
# inverted-U + asymmetry, immune to sparse-cell noise at large |e|
EXPO = [(i, j) for d in range(5) for i in range(d + 1) for j in [d - i]]


def design(e1, e2):
    return np.stack([e1 ** i * e2 ** j for i, j in EXPO], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", default=ips.G0_CAT)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--chunk", type=int, default=262_144)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", required=True)
    import inspect
    optics = {k: float(p.default)  # canonical values: rescale()'s own signature
              for k, p in inspect.signature(rescale).parameters.items()
              if p.default is not inspect.Parameter.empty}
    for k, v in optics.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={os.path.basename(args.measurement_model)}")
    bundle = load_measurement_model(args.measurement_model, device=device)
    # FENCE (realisation-aware head). U is defined as < ehat - mu_flow >, i.e. it assumes the
    # model's conditional mean IS mu(c). For a ConditionalMeanFlowRA it is mu(c) + E_u[A(c,u)],
    # so every residual below -- and the polynomial fitted to them -- would be biased by the
    # A term with nothing raising. It is also consumed by infer_posterior_shape, which itself
    # refuses RA models (PosteriorShapeEstimator fence).
    if isinstance(bundle.model, ConditionalMeanFlowRA):
        raise NotImplementedError(
            "build_mu_correction subtracts model._mu as the conditional mean; for a "
            "realisation-aware checkpoint the conditional mean is mu(c) + E_u[A(c,u)] and the "
            "fitted U would be biased. Use a 'mean_affine' checkpoint.")
    rk = {k: getattr(args, k) for k in optics}

    tnames = bundle.target_transform.target_names
    cols = ips.COND_RAW + ips.NBR_COLS + ["detected", "e1_input_rot0_p",
                                          "e2_input_rot0_p", *tnames]
    df = ips.stream_feather(args.catalogue, cols, max_rows=3 * args.max_rows)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    ehat = df[list(tnames)].to_numpy(np.float32)
    fin = np.isfinite(ehat).all(axis=1)
    df = df[fin].reset_index(drop=True).iloc[:args.max_rows].reset_index(drop=True)
    ehat = df[list(tnames)].to_numpy(np.float64)
    print(f"N={len(df):,} detected+selected, finite targets")

    fr = rescale(df, **rk)  # copies rot0 -> e1/e2_input_p: context holds the TRUE e
    pre = bundle.condition_preprocessor
    ctx = np.asarray(pre.transform_frame(fr), dtype=np.float32)
    ts = bundle.target_transform
    mu = np.empty((len(ctx), ts.dim), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(ctx), args.chunk):
            t = torch.as_tensor(ctx[s:s + args.chunk], device=device)
            mu[s:s + args.chunk] = bundle.model._mu(t).cpu().numpy()
    mu = mu * ts.scales[None, :] + ts.means[None, :]
    print(f"mu evals: {len(ctx):,} in {time.time() - t0:.0f}s")
    resid = ehat - mu  # (N,2) raw target units

    e1 = df["e1_input_rot0_p"].to_numpy(float)
    e2 = df["e2_input_rot0_p"].to_numpy(float)
    rmag = df["r_input_p"].to_numpy(float)
    nmb = len(ips.MAG_EDGES) - 1
    mb = np.clip(np.digitize(rmag, ips.MAG_EDGES) - 1, 0, nmb - 1)

    coeffs = np.zeros((nmb, 2, len(EXPO)))
    for k in range(nmb):
        s = mb == k
        X = design(e1[s], e2[s])
        sol, *_ = np.linalg.lstsq(X, resid[s], rcond=None)  # (nterms, 2)
        fit = X @ sol
        pop_mean = fit.mean(axis=0)          # remove the population mean over the
        sol[0] -= pop_mean                   # cell's own e-sample (m_f stays with f)
        coeffs[k] = sol.T
        rms_before = resid[s].std(axis=0)
        rms_after = (resid[s] - fit).std(axis=0)
        print(f"  mag [{ips.MAG_EDGES[k]:g},{ips.MAG_EDGES[k+1]:g}] N={int(s.sum()):>9,}  "
              f"U1 range [{(fit[:,0]-pop_mean[0]).min():+.4f},{(fit[:,0]-pop_mean[0]).max():+.4f}]  "
              f"resid rms {rms_before[0]:.4f}->{rms_after[0]:.4f} (e1), "
              f"{rms_before[1]:.4f}->{rms_after[1]:.4f} (e2)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, mag_edges=ips.MAG_EDGES, expo=np.array(EXPO),
             coeffs=coeffs, model=os.path.basename(args.measurement_model),
             n_rows=len(df))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
