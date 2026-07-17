"""Per-magnitude flow-vs-sim mean-response ratio on the antithetic constant gold render.

WORKLOG cont.32/33: the sheared-prior etilde residual (job 15064545 stage B, 4M rows)
is a sign-changing pattern in true r-mag (+2.2..+2.9% bright, -1.1% faint, crossing
zero near r~25.7) that does NOT track blending and is NOT fixed by conditioning the
prior on magnitude (rmag priors: global m moves +0.75% -> +0.88%).  In the
linear-Gaussian projection limit a likelihood WIDTH error cannot bias etilde under
the exact sheared prior (the posterior mean is affine in the measurement, so the
tower rule survives a wrong sigma); a conditional-MEAN response error can:

    m_she(bin)  ~= w(bin) * (R_sim/R_flow - 1)(bin)     w = posterior data weight
    K_intr(bin)  = w(bin) * (R_sim/R_flow)(bin)         (intrinsic-prior shrinkage)
 => m_she(bin) ~= K_intr(bin) * (1 - R_flow/R_sim)(bin)

This measures, on the SAME catalogue/selection/lookup as the gold etilde runs:

    R_sim(bin)  = <(ehat_+ - ehat_-) . ghat> / (2g)     pure columns, no model
    R_flow(bin) = <(mu(ctx(e_+)) - mu(ctx(e_-))) . ghat> / (2g)

with e_+/- the intrinsic shape Mobius-sheared by the per-case +/-g and mu the flow's
conditional mean (2 mean-net evals per object, NO grid, NO flow.log_prob -> ~1000x
cheaper than an etilde run; CPU is fine).  If (R_flow/R_sim - 1)(bin) mirrors the
measured m_she(bin) pattern, the etilde residual is the flow's per-magnitude
mean-response miscalibration -- a flow-training problem (the other session's
territory), not an inference-pipeline problem; the pipeline itself is then exact at
the closure level and the deployment question becomes purely how well the flow's
response can be calibrated per magnitude.

Compare the printed table against the stage-B "m_etilde by true r-mag" table in
logs/infer_etilde_blend_15064545.out.
"""
import argparse
import os
import sys
import time

import numpy as np
import pyarrow.feather as pf
import torch

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SBSI_ROOT = os.path.dirname(SCRIPTS)
for p in (SBSI_ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer_posterior_shape as ips  # noqa: E402  (loader helpers + shared constants)
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbs_shear.posterior_shape import shape_target_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", default=None, help=f"default: {ips.GOLD_CAT}")
    ap.add_argument("--crowd-flux-lookup", default=ips.CROWD_LOOKUP)
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c40-139.feather")
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=262_144)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/flow_response_by_mag.csv")
    ap.add_argument("--dump-objects", default=None,
                    help="feather of per-object (case,input_index,r_input_p,r_blend,"
                         "r_sim,r_flow) for joint mag x blending cells vs the etilde dump")
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
    rk = {k: getattr(args, k) for k in optics}

    # --- load: identical path to infer_posterior_shape.run_gold (uncached branch) ---
    cat = args.catalogue or ips.GOLD_CAT
    meas = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus"]
    need = [*meas, "applied_g1", "applied_g2", "case", "input_index",
            "axis_ratio_input_p", "position_angle_input_p", *ips.COND_RAW]
    stride = ips.auto_stride(cat, args.max_rows * 2)
    print(f"batch stride={stride} for max_rows={args.max_rows:,}")
    df = ips.stream_feather(cat, need, max_rows=0, stride=stride)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i
    df["e2_input_rot0_p"] = e2i
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    fin = np.isfinite(df[meas].to_numpy(float)).all(axis=1)
    df = df[fin].reset_index(drop=True)
    if len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    cf = pf.read_table(args.crowd_flux_lookup).to_pandas()[["case", "input_index", *ips.NBR_COLS]]
    cm = df[["case", "input_index"]].merge(cf, on=["case", "input_index"], how="left")
    for c in ips.NBR_COLS:
        df[c] = cm[c].fillna(0.0).to_numpy(float)
    print(f"crowd-flux lookup: matched {np.mean(cm['nbr_flux_near'].notna()):.1%} of rows")
    df["r_blend"] = 0.0
    if args.blend_lookup and os.path.exists(args.blend_lookup):
        rb = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
        bm = df[["case", "input_index"]].merge(rb, on=["case", "input_index"], how="left")
        df["r_blend"] = bm["R_blend"].fillna(0.0).to_numpy(float)
        print(f"blend lookup (binning only): matched {np.mean(bm['R_blend'].notna()):.1%}")

    gmag = np.hypot(df["applied_g1"], df["applied_g2"]).to_numpy(float)
    g = float(np.median(gmag))
    gh1 = df["applied_g1"].to_numpy(float) / gmag
    gh2 = df["applied_g2"].to_numpy(float) / gmag
    print(f"N={len(df):,}  cases={df['case'].nunique()}  |g|={g:.4f}")

    # --- sim response per object (pure columns) ---
    ep = df[["measured_e1_plus", "measured_e2_plus"]].to_numpy(float)
    em = df[["measured_e1_minus", "measured_e2_minus"]].to_numpy(float)
    r_sim = ((ep[:, 0] - em[:, 0]) * gh1 + (ep[:, 1] - em[:, 1]) * gh2) / (2 * g)

    # --- flow mean response per object: mu at the two lensed contexts ---
    fr = rescale(df, **rk)
    pre = bundle.condition_preprocessor
    ctx = np.asarray(pre.transform_frame(fr), dtype=np.float32)  # (N,D), intrinsic e inside
    i1 = pre.feature_names.index("e1_input_p")
    i2 = pre.feature_names.index("e2_input_p")
    n_feat = len(pre.feature_names)
    ts = bundle.target_transform
    j1, j2 = shape_target_indices(ts.target_names)
    e1s = df["e1_input_rot0_p"].to_numpy(float)
    e2s = df["e2_input_rot0_p"].to_numpy(float)
    g1 = df["applied_g1"].to_numpy(float)
    g2 = df["applied_g2"].to_numpy(float)

    @torch.no_grad()
    def mu_proj(sign):
        """(mu . ghat, e_lensed . ghat) at the sign*g lensed context."""
        l1, l2 = apply_shear_to_ellipticity(e1s, e2s, sign * g1, sign * g2)
        c = ctx.copy()
        c[:, i1] = (l1 - pre.means[i1]) / pre.scales[i1]
        c[:, i2] = (l2 - pre.means[i2]) / pre.scales[i2]
        if pre.add_missing_indicators:
            c[:, n_feat + i1] = 0.0
            c[:, n_feat + i2] = 0.0
        out = np.empty((len(c), ts.dim), np.float32)
        for s in range(0, len(c), args.chunk):
            t = torch.as_tensor(c[s:s + args.chunk], device=device)
            out[s:s + args.chunk] = bundle.model._mu(t).cpu().numpy()
        raw = out * ts.scales[None, :] + ts.means[None, :]
        return raw[:, j1] * gh1 + raw[:, j2] * gh2, l1 * gh1 + l2 * gh2

    t0 = time.time()
    mu_p, elens_p = mu_proj(+1.0)
    mu_m, elens_m = mu_proj(-1.0)
    r_flow = (mu_p - mu_m) / (2 * g)
    print(f"flow mean-response evals: {2 * len(df):,} in {time.time() - t0:.0f}s")

    # --- binned tables: r-mag (MAG_EDGES) and R_blend quantiles, + GLOBAL ---
    import pandas as pd
    rows = []

    def emit(label, sel, val):
        n = int(sel.sum())
        rs, rf = r_sim[sel], r_flow[sel]
        Rs, Rf = rs.mean(), rf.mean()
        Rs_sem = rs.std() / np.sqrt(n)
        Rf_sem = rf.std() / np.sqrt(n)
        d = Rf / Rs - 1.0
        d_sem = abs(Rf / Rs) * np.hypot(Rs_sem / abs(Rs), Rf_sem / abs(Rf))
        m_pred = Rs / Rf - 1.0  # unattenuated (w=1) prediction for m_she
        rows.append(dict(bin=label, n=n, val=val, R_sim=Rs, R_sim_sem=Rs_sem,
                         R_flow=Rf, R_flow_sem=Rf_sem, dratio=d, dratio_sem=d_sem,
                         m_pred_unatten=m_pred))
        print(f"  {label:>10}  N={n:>9,}  R_sim={Rs:+.4f}+/-{Rs_sem:.4f}  "
              f"R_flow={Rf:+.4f}+/-{Rf_sem:.4f}  R_flow/R_sim-1={100 * d:+.2f}%+/-"
              f"{100 * d_sem:.2f}%  ->m_pred(w=1)={100 * m_pred:+.2f}%")

    rmag = df["r_input_p"].to_numpy(float)
    nmb = len(ips.MAG_EDGES) - 1
    magbin = np.clip(np.digitize(rmag, ips.MAG_EDGES) - 1, 0, nmb - 1)
    print("\n--- by true r-mag (compare stage-B 'm_etilde by true r-mag') ---")
    for k in range(nmb):
        sel = magbin == k
        emit(f"{ips.MAG_EDGES[k]:g}-{ips.MAG_EDGES[k + 1]:g}", sel, float(rmag[sel].mean()))

    rb = df["r_blend"].to_numpy(float)
    iso = np.abs(rb) < 0.02
    print("\n--- by emulator R_blend quantile ---")
    emit("ISO(~0)", iso, float(rb[iso].mean()))
    if (~iso).sum():
        qs = np.quantile(rb[~iso], [0.25, 0.5, 0.75])
        qbin = np.digitize(rb, qs)
        for k in range(4):
            sel = (~iso) & (qbin == k)
            emit(f"q{k + 1}", sel, float(rb[sel].mean()))

    print("\n--- GLOBAL ---")
    emit("GLOBAL", np.ones(len(df), bool), float("nan"))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nwrote {args.output}")
    if args.dump_objects:
        # ghat-projected per-object pieces: residual-profile analysis (conditional mean
        # residual vs projected lensed e -- probes mu's e-profile beyond the local slope)
        pd.DataFrame(dict(case=df["case"].to_numpy(np.int64),
                          input_index=df["input_index"].to_numpy(np.int64),
                          r_input_p=rmag, r_blend=rb,
                          r_sim=r_sim.astype(np.float32),
                          r_flow=r_flow.astype(np.float32),
                          mu_p=mu_p.astype(np.float32), mu_m=mu_m.astype(np.float32),
                          ehat_p=(ep[:, 0] * gh1 + ep[:, 1] * gh2).astype(np.float32),
                          ehat_m=(em[:, 0] * gh1 + em[:, 1] * gh2).astype(np.float32),
                          elens_p=elens_p.astype(np.float32),
                          elens_m=elens_m.astype(np.float32))).to_feather(args.dump_objects)
        print(f"per-object dump -> {args.dump_objects} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
