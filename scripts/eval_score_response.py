"""INFERENCE.md §5B in code: the shear response and the shear itself, from DATA.

The certified Gold-v1 number is a TRANSPORT measurement (§5A): shear the true shapes by
+-g, push both legs through the flow, difference.  It needs the truth, so it only runs in
a simulation.  §5B evaluates the SAME derivative the other way round -- per-object score
`s_i = E_post[u]`, information `I_i`, and `ghat = sum s / sum I` -- from the measured
shapes alone.  This script runs both and compares them.

Three modes, in increasing order of exposure to reality:

  unit        node bank only.  `E_0[u] = 0` and `E_0[du] + Var_0(u) = 0` must hold for any
              normalised sheared prior (Bartlett with a flat likelihood), and the
              finite-difference generator must reproduce the closed form.  No flow, no data.

  closure     synthetic data DRAWN FROM THE FLOW at a known shear.  Model and data agree
              by construction, so `ghat` must return the injected shear and
              `Cov(ehat, s)` must return the flow's own transport response.  This is the
              end-to-end unit test of Fisher's identity (2.2) + the estimator (2.6).

  constgold   the certified constant-shear catalogue.  Here model and data DISAGREE, by
              exactly the amount Gold-v1 already quantified: the flow carries the self
              response only, so the score route should over-estimate the shear by
              `R_sim / R_flow`, and adding the external `R_blend` (§5C.3) should collapse
              that to the certified `m`.

The predicted numbers, from `Gold-v1.md`:  R_sim = 0.4534, R_flow = 0.2930,
R_blend = 0.1593, so ghat/g = 0.4534/0.2930 = 1.547 bare and 0.4534/0.4523 = 1.0025 with
the blend injection -- i.e. m = +54.7% and +0.245%.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.posterior_shape import PosteriorShapeEstimator, make_e_grid  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.response import flow_response  # noqa: E402
from sbs_shear.score_inference import (  # noqa: E402
    ShapeScoreNodes,
    SmoothRadialPrior,
    blend_injection_term,
    bootstrap_by_case,
    project,
    response_from_score,
    scores_from_loglike,
    shear_estimate,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
CATBASE = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
G0_CAT = CATBASE + "det_meas_crowd_conc_g0.0_train_full.feather"
GOLD_CAT = CBASE + "constant_response_catalogue_train.feather"
PRIOR_CACHE = "results/etilde_prior_e_samples.feather"

# Gold-v1.md §5, the transport reference this script is checked against.
R_SIM_CERT, R_FLOW_CERT, R_BLEND_CERT = 0.4534, 0.2930, 0.1593


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------

def stream(path, columns, max_rows, stride=1, start=0):
    with ipc.open_file(path) as r:
        avail = set(r.schema.names)
        cols = [c for c in columns if c in avail]
        parts, n = [], 0
        for bi in range(start, r.num_record_batches, max(1, stride)):
            parts.append(pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas())
            n += len(parts[-1])
            if n >= max_rows:
                break
    return pd.concat(parts, ignore_index=True).iloc[:max_rows].reset_index(drop=True)


def case_batch_plan(path, min_case, max_rows, oversample=1.6):
    """First record batch with `case >= min_case`, and a stride that spans the rest.

    Constgold is written in case order (~4.5 batches per case), so reading the first N
    rows and then cutting on `case` would return an empty held-out set.  Striding across
    the remaining batches instead gives a sample spread over ~all held-out cases, which
    the per-case bootstrap needs anyway.
    """
    with ipc.open_file(path) as r:
        nb = r.num_record_batches
        if min_case is None:
            return 0, max(1, nb // max(1, int(np.ceil(max_rows * oversample / 65536))))
        lo, hi = 0, nb - 1
        while lo < hi:                                   # first batch reaching min_case
            mid = (lo + hi) // 2
            c = pa.Table.from_batches([r.get_batch(mid)]).select(["case"]).to_pandas()
            if int(c["case"].min()) >= min_case:   # ALL rows of this batch qualify
                hi = mid
            else:
                lo = mid + 1
        per = pa.Table.from_batches([r.get_batch(lo)]).num_rows
    need = max(1, int(np.ceil(max_rows * oversample / per)))
    return lo, max(1, (nb - lo) // need)


def load_g0(path, max_rows):
    """Detected + source-selected rows of the g=0 training catalogue."""
    cols = ["e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
            "sersic_n_input_p", "sersic_n_input_s", "measured_mag_auto",
            "measured_flux_radius", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
            "Re_input_p", "Re_input_s", "r_input_p", "r_input_s", "distance",
            "neighbored", "detected", "polarization_angle", "case", "input_index",
            "measured_ngmix_g1", "measured_ngmix_g2"]
    df = stream(path, cols, int(max_rows * 1.6) + 10_000)
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    df = df[df["detected"].astype(bool)].reset_index(drop=True)
    df["gamma1_input_p"] = 0.0
    df["gamma2_input_p"] = 0.0
    return df.iloc[:max_rows].reset_index(drop=True)


def load_constgold(args):
    """Constgold rows + the lookups the measured-conditioned flow needs.

    Deliberately a copy of `validate_constant_with_blend.load()` plus its lookup merges,
    so the sample this script scores is row-for-row the sample Gold-v1 certified.
    """
    need = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "distance", "input_index", "case",
            "polarization_angle", "Re_input_p", "Re_input_s", "axis_ratio_input_p",
            "axis_ratio_input_s", "position_angle_input_p", "position_angle_input_s",
            "r_input_p", "r_input_s", "sersic_n_input_p", "sersic_n_input_s"]
    start, stride = case_batch_plan(args.catalogue, args.min_case, args.max_rows)
    print(f"constgold: reading from batch {start} with stride {stride}")
    df = stream(args.catalogue, need, int(args.max_rows * 1.6), stride=stride, start=start)
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i
    df["e2_input_rot0_p"] = e2i
    e1s, e2s = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_s"].to_numpy(float),
                                                 df["position_angle_input_s"].to_numpy(float))
    df["e1_input_rot0_s"] = e1s
    df["e2_input_rot0_s"] = e2s
    df["gamma1_input_p"] = 0.0
    df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    if args.min_case is not None:
        df = df[df["case"] >= args.min_case].reset_index(drop=True)
        print(f"held-out split: case >= {args.min_case} -> N={len(df):,}")
    if len(df) < 100:
        raise RuntimeError(f"only {len(df)} rows survived the case cut -- raise --max-rows")
    df = df.iloc[:args.max_rows].reset_index(drop=True)
    print(f"scoring N={len(df):,} rows over {df['case'].nunique()} cases "
          f"({df['case'].min()}..{df['case'].max()})")

    def merge(path, cols, fill):
        t = pf.read_table(path).to_pandas()
        have = [c for c in cols if c in t.columns]
        m = df[["case", "input_index"]].merge(t[["case", "input_index", *have]],
                                              on=["case", "input_index"], how="left")
        for c in have:
            df[c] = m[c].fillna(fill).to_numpy(float) if fill is not None else m[c].to_numpy(float)
        return float(np.mean(m[have[0]].notna())), have

    frac, cols = merge(args.crowd_flux_lookup, ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"], 0.0)
    print(f"crowd-flux lookup: matched {frac:.1%}; cols={cols}")
    # measured mag/size: keep NaN where unmatched so the trained missing-indicator fires
    frac, cols = merge(args.meas_prim_lookup, ["measured_mag_auto", "measured_flux_radius",
                                               "measured_class_star"], None)
    print(f"meas-prim lookup: matched {frac:.1%}; cols={cols}")
    rbt = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
    m = df[["case", "input_index"]].merge(rbt, on=["case", "input_index"], how="left")
    df["R_blend"] = m["R_blend"].fillna(0.0).to_numpy(float)
    print(f"R_blend lookup: matched {float(np.mean(m['R_blend'].notna())):.1%}, "
          f"<R_blend>={df['R_blend'].mean():.4f}")
    return df


def build_prior(args):
    """Smooth isotropic prior over the intrinsic ellipticity (the g=0 population)."""
    if not os.path.exists(args.prior_sample):
        print(f"prior cache missing -> building from {os.path.basename(args.prior_catalogue)}",
              flush=True)
        s = load_g0(args.prior_catalogue, args.prior_rows)
        os.makedirs(os.path.dirname(args.prior_sample) or ".", exist_ok=True)
        s[["e1_input_rot0_p", "e2_input_rot0_p", "r_input_p"]].astype(np.float32) \
            .reset_index(drop=True).to_feather(args.prior_sample)
    s = pf.read_table(args.prior_sample).to_pandas()
    prior = SmoothRadialPrior(s["e1_input_rot0_p"].to_numpy(), s["e2_input_rot0_p"].to_numpy(),
                              n_bins=args.prior_bins, n_knots=args.prior_knots)
    print(f"prior: {prior.n_samples:,} shapes, r_max={prior.r_max:.3f}, "
          f"per-comp std={s[['e1_input_rot0_p','e2_input_rot0_p']].std().mean():.4f}, "
          f"{prior.n_knots} knots, chi2/dof={prior.fit_chi2_dof:.2f}, "
          f"norm err={prior.norm_error:+.2e}", flush=True)
    return prior


# ------------------------------------------------------------------------------------
# the §5B pass
# ------------------------------------------------------------------------------------

"""(the old whole-catalogue log-likelihood buffer is gone -- see `score_pass`)"""


def grad_ehat_on_grid(est, frame, ehat_raw, chunk=256):
    """`grad_ehat log p_flow(ehat | e_k, rest)` per (galaxy, node) -- §5C.3's injection.

    In RAW target units, so the injected `R_blend * delta_e` shift is in the same units
    as the measured shape.  The residual flow is blind to `e`, so the density is a
    location family and the gradient is just the residual flow's own score at
    `ehat - mu(e_k)`; autograd handles it without any structural assumption.
    """
    model = est.bundle.model
    tstd = est.bundle.target_transform
    scales = torch.as_tensor(tstd.scales, dtype=torch.float32, device=est.device)
    ehat_std = tstd.transform_array(np.asarray(ehat_raw, dtype=np.float32))
    n = len(frame)
    out = np.empty((n, est.G, 2), dtype=np.float16)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rep = est._grid_tiled_context(frame.iloc[start:stop])
        b = rep.shape[0]
        flat = rep.view(b * est.G, -1)
        with torch.no_grad():
            mu = model._mu(flat)
            fctx = model._flow_ctx(flat)
        xh = torch.as_tensor(ehat_std[start:stop], dtype=torch.float32, device=est.device)
        x = xh[:, None, :].expand(b, est.G, 2).reshape(b * est.G, 2) - mu
        x = x.detach().requires_grad_(True)
        ll = model.flow.log_prob(x, fctx).sum()
        (gx,) = torch.autograd.grad(ll, x)
        # d/d ehat_raw = (d/d ehat_std) / scale
        out[start:stop] = (gx / scales).view(b, est.G, 2).detach().cpu().numpy().astype(np.float16)
    return out


def score_pass(est, nodes, frame, ehat_raw, chunk, tag, r_blend=None, grad_chunk=256,
               slab_mult=64, analytic_info=False):
    """One leg: log-likelihood over the node bank -> `(s_i, I_i)`, slab by slab.

    Nothing of size `N x G` is ever held: each slab's likelihood (and, when injecting,
    its `grad_ehat log p_flow`) is consumed into the per-object `(s, I)` and dropped.
    At a million rows and a 61-grid the whole-catalogue buffer alone would be 5.5 GB per
    leg, and the injection's would be 22 GB.
    """
    n = len(frame)
    s_out = np.empty((n, 2))
    i_out = np.empty((n, 2, 2))
    slab = chunk * slab_mult
    t0 = time.time()
    for s0 in range(0, n, slab):
        s1 = min(s0 + slab, n)
        ll = est.log_likelihood(frame.iloc[s0:s1], ehat_raw[s0:s1], chunk=chunk,
                                row_offset=s0)
        extra = None
        if r_blend is not None:
            g = grad_ehat_on_grid(est, frame.iloc[s0:s1], ehat_raw[s0:s1], chunk=grad_chunk)
            extra = blend_injection_term(g.astype(np.float32), nodes.grid, r_blend[s0:s1])
        s_out[s0:s1], i_out[s0:s1], _ = scores_from_loglike(
            ll, nodes, device=est.device, extra=extra, analytic_info=analytic_info)
        el = time.time() - t0
        print(f"  [{tag}] {s1:,}/{n:,}  {el:.0f}s  ETA {el / s1 * (n - s1):.0f}s", flush=True)
    return s_out, i_out


def report_leg(tag, s, info, ehat, gh1, gh2, g_true, cases, n_boot=200):
    s_p, i_p = project(s, info, gh1, gh2)
    e_p = ehat[:, 0] * gh1 + ehat[:, 1] * gh2
    ghat, den = shear_estimate(s_p, i_p)
    R = response_from_score(e_p, s_p)
    per = s_p / (den / len(s_p))          # per-object ghat contribution, for bootstrap
    err = bootstrap_by_case(per, cases, n_boot=n_boot) if cases is not None else float("nan")
    print(f"  {tag:>10}:  <s>={np.mean(s_p):+.4f}  <I>={np.mean(i_p):.3f}  "
          f"ghat={ghat:+.5f} +/- {err:.5f}   (g_true={g_true:+.4f})   "
          f"Cov(ehat,s)={R:+.4f}", flush=True)
    return dict(ghat=ghat, ghat_err=err, R=R, s_proj=s_p, i_proj=i_p, e_proj=e_p, per=per)


# ------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------

def mode_unit(args, prior, grid):
    print("\n=== MODE unit: prior-only Bartlett identities (no flow, no data) ===")
    for n in sorted({args.grid_n, 41, 61}):
        gr, area = make_e_grid(n=n, emax=args.grid_emax, rmax=args.grid_rmax)
        nodes = ShapeScoreNodes(gr, prior, delta=args.fd_delta)
        b = nodes.bartlett()
        print(f"  grid n={n:>3} (G={len(gr):>5}, cell={area:.2e}):")
        print(f"     E_0[u]                  = [{b['mean_u'][0]:+.4e}, {b['mean_u'][1]:+.4e}]"
              f"   (rms u = {b['scale']:.3f})")
        print(f"     E_0[du] + Var_0(u)      = [[{b['curvature'][0,0]:+.4e}, "
              f"{b['curvature'][0,1]:+.4e}], [{b['curvature'][1,0]:+.4e}, "
              f"{b['curvature'][1,1]:+.4e}]]")
        print(f"     |u_fd - u_closed|/rms(u)= {nodes.closed_form_residual():.3e}")
    # analytic sanity: for a Gaussian-like prior, u ~ e * (4 + 1/sigma^2) near the origin
    gr, _ = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    nodes = ShapeScoreNodes(gr, prior, delta=args.fd_delta)
    r = np.hypot(gr[:, 0], gr[:, 1])
    inner = (r > 0.02) & (r < 0.15)
    amp = np.hypot(nodes.u[inner, 0], nodes.u[inner, 1]) / r[inner]
    b = nodes.bartlett()
    print(f"  small-|e| amplitude |u|/|e| = {amp.mean():.2f} +/- {amp.std():.2f}   "
          f"(closed form: 4 - 2 psi'(0))")
    print(f"  Var_0(u) = {b['scale'] ** 2:.3f};  Bartlett curvature is "
          f"{abs(b['curvature'][0, 0]) / b['scale'] ** 2:.3%} of it "
          f"-> the spurious per-object information floor")
    return nodes


def mode_closure(args, bundle, prior, grid, rk):
    """Synthetic data from the flow at a KNOWN shear -> ghat must return it."""
    print("\n=== MODE closure: data drawn FROM the flow at known shear ===")
    df = load_g0(args.g0_catalogue, args.max_rows)
    print(f"rows: {len(df):,} detected+selected from {os.path.basename(args.g0_catalogue)}")
    rng = np.random.default_rng(args.seed)
    e1i, e2i = prior.sample(len(df), rng)         # generative prior == estimation prior
    gh1 = np.ones(len(df))
    gh2 = np.zeros(len(df))
    g = args.closure_g

    est = PosteriorShapeEstimator(bundle, grid, device=args.device)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"node bank: G={len(grid)}, supported={int(nodes.support.sum())}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}")

    # transport reference on THESE rows, same flow, same conditioning (§5A)
    def reseed():
        torch.manual_seed(args.flow_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.flow_seed)

    R_transport = flow_response(bundle, df, g, gh1, gh2, (e1i, e2i), rk,
                                args.n_samples, args.batch_size, reseed=reseed)
    print(f"transport R_flow on these rows = {R_transport:.4f}  "
          f"(certified {R_FLOW_CERT:.4f})", flush=True)

    out = {}
    for sign, tag in ((+1, "leg +g"), (-1, "leg -g")):
        e1l, e2l = apply_shear_to_ellipticity(e1i, e2i, sign * g * gh1, sign * g * gh2)
        fr = df.copy()
        fr["e1_input_rot0_p"] = e1l
        fr["e2_input_rot0_p"] = e2l
        fr = rescale(fr, **rk)
        reseed()                                    # CRN: same latents in both legs
        ehat = bundle.sample(fr, n_samples=1, batch_size=args.batch_size)[:, 0, :]
        s, info = score_pass(est, nodes, fr, ehat, args.chunk, tag,
                             slab_mult=args.slab_mult)
        out[sign] = report_leg(tag, s, info, ehat, gh1, gh2, sign * g, None)
        if sign > 0:
            # the two information estimators must agree on the plain model; only then
            # is the finite-difference one trustworthy for the injected model (§5C.3)
            _, info_a = score_pass(est, nodes, fr.iloc[:args.info_check_rows],
                                   ehat[:args.info_check_rows], args.chunk,
                                   "I-analytic", slab_mult=args.slab_mult,
                                   analytic_info=True)
            k = args.info_check_rows
            fd = info[:k, 0, 0].mean()
            an = info_a[:, 0, 0].mean()
            print(f"      information cross-check on {k:,} rows: "
                  f"<I11> finite-difference {fd:.4f}  vs  Louis analytic {an:.4f}  "
                  f"(ratio {fd / an:.4f})", flush=True)

    sp = 0.5 * (out[+1]["s_proj"] - out[-1]["s_proj"])
    ip = 0.5 * (out[+1]["i_proj"] + out[-1]["i_proj"])
    ghat_anti = float(np.sum(sp) / np.sum(ip))
    err = float(np.std(sp / np.mean(ip)) / np.sqrt(len(sp)))
    R_score = 0.5 * (out[+1]["R"] + out[-1]["R"])
    print(f"\n  ANTITHETIC ghat = {ghat_anti:+.5f} +/- {err:.5f}   "
          f"(injected {g:+.4f};  ratio {ghat_anti / g:.4f}, m = {ghat_anti / g - 1:+.2%})")
    print(f"  Cov(ehat,s) leg-averaged = {R_score:.4f}   vs transport R_flow = "
          f"{R_transport:.4f}   ratio {R_score / R_transport:.4f}")
    print("\n  VERDICT: with model == data both lines must read 1.000 up to MC error;")
    print("           a departure is a bug in the generator, the prior or the weights.")
    return dict(ghat=ghat_anti, err=err, R_score=R_score, R_transport=R_transport)


def mode_constgold(args, bundle, prior, grid, rk):
    """The certified catalogue: score route vs the Gold-v1 transport numbers."""
    print("\n=== MODE constgold: score route on the certified catalogue ===")
    df = load_constgold(args)
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    norm = np.hypot(df["applied_g1"], df["applied_g2"]).to_numpy(float)
    gh1 = df["applied_g1"].to_numpy(float) / norm
    gh2 = df["applied_g2"].to_numpy(float) / norm
    cases = df["case"].to_numpy(np.int64)
    rb = df["R_blend"].to_numpy(float)
    print(f"N={len(df):,}  |g|={g:.4f}  n_cases={df['case'].nunique()}  "
          f"blended={df['neighbored'].astype(bool).mean():.3f}")

    e1p = df["measured_e1_plus"].to_numpy(float)
    e2p = df["measured_e2_plus"].to_numpy(float)
    e1m = df["measured_e1_minus"].to_numpy(float)
    e2m = df["measured_e2_minus"].to_numpy(float)
    r_sim = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2 * g)
    R_sim = float(np.mean(r_sim))
    print(f"R_sim (transport truth, this sample) = {R_sim:.4f}  "
          f"(certified {R_SIM_CERT:.4f})", flush=True)

    est = PosteriorShapeEstimator(bundle, grid, device=args.device)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    print(f"node bank: G={len(grid)}, supported={int(nodes.support.sum())}, "
          f"|u_fd-u_closed|/rms={nodes.closed_form_residual():.2e}")

    # transport R_flow on exactly these rows (the Gold-v1 harvest, for reference)
    def reseed():
        torch.manual_seed(args.flow_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.flow_seed)

    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(),
            df["e2_input_rot0_p"].to_numpy(float).copy())
    R_flow = flow_response(bundle, df, g, gh1, gh2, intr, rk, args.n_samples,
                           args.batch_size, reseed=reseed)
    R_blend = float(np.mean(rb))
    print(f"transport R_flow = {R_flow:.4f} (certified {R_FLOW_CERT:.4f}); "
          f"R_blend = {R_blend:.4f} (certified {R_BLEND_CERT:.4f})")
    print(f"transport m = R_sim/(R_flow+R_blend) - 1 = "
          f"{R_sim / (R_flow + R_blend) - 1:+.2%}", flush=True)

    fr = rescale(df.copy(), **rk)
    results = {}
    for inject in ([False, True] if args.inject_blend else [False]):
        legs = {}
        for sign, ehat in ((+1, np.stack([e1p, e2p], 1)), (-1, np.stack([e1m, e2m], 1))):
            tag = f"{'inj' if inject else 'bare'} {'+g' if sign > 0 else '-g'}"
            s, info = score_pass(est, nodes, fr, ehat, args.chunk, tag,
                                 r_blend=(rb if inject else None),
                                 grad_chunk=args.grad_chunk,
                                 slab_mult=args.slab_mult)
            legs[sign] = report_leg(tag, s, info, ehat, gh1, gh2, sign * g, cases,
                                    n_boot=args.n_boot)
        sp = 0.5 * (legs[+1]["s_proj"] - legs[-1]["s_proj"])
        ip = 0.5 * (legs[+1]["i_proj"] + legs[-1]["i_proj"])
        ghat = float(np.sum(sp) / np.sum(ip))
        per = sp / np.mean(ip)
        err = bootstrap_by_case(per, cases, n_boot=args.n_boot)
        R_score = 0.5 * (legs[+1]["R"] + legs[-1]["R"])
        label = "WITH R_blend injection (§5C.3)" if inject else "BARE flow (§5B)"
        print(f"\n  --- {label} ---")
        print(f"  ANTITHETIC ghat = {ghat:+.5f} +/- {err:.5f}   g_true = {g:.4f}")
        print(f"    ghat/g          = {ghat / g:.4f} +/- {err / g:.4f}")
        print(f"    m_5B            = {ghat / g - 1:+.2%} +/- {err / g:.2%}")
        R_model = R_flow + (R_blend if inject else 0.0)
        print(f"    predicted ghat/g = R_sim/R_model = {R_sim / R_model:.4f}  "
              f"(R_model = {R_model:.4f})")
        print(f"    inferred response R_model * ghat/g = {R_model * ghat / g:.4f}  "
              f"vs R_sim = {R_sim:.4f}")
        print(f"    Cov(ehat,s) leg-averaged = {R_score:.4f}")
        results[inject] = dict(ghat=ghat, err=err, R_score=R_score)
    print(f"\n  transport reference:   R_sim={R_sim:.4f}  R_flow={R_flow:.4f}  "
          f"R_blend={R_blend:.4f}")
    return dict(R_sim=R_sim, R_flow=R_flow, R_blend=R_blend, results=results)


# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["unit", "closure", "constgold"])
    ap.add_argument("--measurement-model",
                    default="models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt")
    ap.add_argument("--catalogue", default=GOLD_CAT)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_conc_c0-199.feather")
    ap.add_argument("--meas-prim-lookup", default="results/meas_prim_lookup_c0-139.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c40-139.feather")
    ap.add_argument("--inject-blend", action="store_true",
                    help="also run the §5C.3 external-R_blend injection")
    # node bank
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--analytic-info", action="store_true",
                    help="use the Louis form for I instead of d_gamma s (plain model only)")
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    # execution
    ap.add_argument("--closure-g", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--grad-chunk", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--slab-mult", type=int, default=64)
    ap.add_argument("--info-check-rows", type=int, default=20_000)
    ap.add_argument("--info-delta", type=float, default=0.0025)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", default=None)
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0,
                     psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={args.device}  torch={torch.__version__}")

    prior = build_prior(args)
    grid, cell = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    print(f"grid: n={args.grid_n} -> G={len(grid)} nodes, cell area {cell:.3e}")

    if args.mode == "unit":
        mode_unit(args, prior, grid)
        return
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    print(f"model: {os.path.basename(args.measurement_model)}")
    if args.mode == "closure":
        res = mode_closure(args, bundle, prior, grid, rk)
    else:
        res = mode_constgold(args, bundle, prior, grid, rk)
    if args.dump:
        np.savez(args.dump, **{k: v for k, v in res.items() if not isinstance(v, dict)})
        print(f"dumped -> {args.dump}")


if __name__ == "__main__":
    main()
