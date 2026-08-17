"""Where the wall-clock actually goes in `eval_score_select.py`, and what it would cost
to get it back.

The definitive §5B runs take 8+ hours, and the profile is not mysterious: both halves are
dominated by the SAME 10-layer affine coupling stack evaluated once per `(row, node)`.
At `G = 2765` nodes and 2M rows a leg that is `2.2e10` network evaluations, `2.7 MFLOP`
each, so `60 PFLOP` per leg -- and we currently spend them in fp32 on a *fractional* A40.
Two things follow, and this script measures both rather than asserting them:

  (1) PRECISION.  The whole stack is 256x256 matmuls, exactly what Ampere's tensor cores
      exist for, and PyTorch leaves TF32 OFF by default for matmul.  The relevant question
      is not "is TF32 faster" (it is) but "does it move the answer", and the honest metric
      is not `max |dlogL|` -- it is the shift in `ghat` itself, since the log-likelihood is
      already stored to fp16 and everything downstream is a softmax over the node bank,
      which is invariant to a per-row constant and forgiving of the rest.  So each mode is
      carried all the way through `score_pass` to `(sum s, sum I)` and the modes are
      compared as a shear.

  (2) THE `Pi` LOOP.  `pass_fraction_by_node` rebuilds the ENTIRE conditioning frame inside
      its node loop -- `sub.copy()` then `rescale()` then the preprocessor's
      `transform_frame` inside `bundle.sample` -- 2765 times per replicate, when exactly two
      columns change.  The scoring side already avoids this (`_grid_tiled_context` writes
      the two standardized columns into a tiled tensor), which is why `Pi` runs at a small
      fraction of the score pass's throughput despite doing the same kind of arithmetic.
      This script times the per-node body against the flow sample alone; the gap is the
      hoisting prize.

Nothing here changes a result.  It is a measurement of the cost of producing them, run at
a size where every mode finishes in a couple of minutes.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbsi.measurement_model import load_measurement_model  # noqa: E402
from sbsi.posterior_shape import PosteriorShapeEstimator, make_e_grid  # noqa: E402
from sbsi.precision import precision_region  # noqa: E402
from sbsi.preprocessing import rescale  # noqa: E402
from sbsi.score_inference import ShapeScoreNodes  # noqa: E402
from sbsi.shear_map import apply_shear_to_ellipticity  # noqa: E402

from eval_score_response import (  # noqa: E402
    G0_CAT, LL_DTYPE, PRIOR_CACHE, build_prior, load_g0, score_pass,
)
from eval_score_select import MODEL  # noqa: E402

# MACs in one pass of the residual flow, from the certified checkpoint's shapes: 10 affine
# coupling layers, each Linear(14,256) -> Linear(256,256) -> Linear(256,256) -> Linear(256,4),
# plus the 16->128->2 mean head.  Used only to quote throughput in TFLOP/s, which is what
# makes "are we near the hardware's peak" answerable.
MACS_PER_EVAL = 10 * (14 * 256 + 256 * 256 + 256 * 256 + 256 * 4) + (16 * 128 + 128 * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default=MODEL)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--prior-sample", default=PRIOR_CACHE)
    ap.add_argument("--prior-catalogue", default=G0_CAT)
    ap.add_argument("--prior-rows", type=int, default=2_000_000)
    ap.add_argument("--prior-bins", type=int, default=120)
    ap.add_argument("--prior-knots", type=int, default=8)
    ap.add_argument("--prior-knot-margin", type=float, default=0.10)
    ap.add_argument("--rows", type=int, default=40_000,
                    help="rows for the score-pass timing; big enough that the fixed "
                         "start-up cost is not what is being measured")
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--grid-emax", type=float, default=0.96)
    ap.add_argument("--grid-rmax", type=float, default=0.95)
    ap.add_argument("--fd-delta", type=float, default=0.01)
    ap.add_argument("--info-delta", type=float, default=0.02)
    ap.add_argument("--grad-delta", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--slab-mult", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--ll-dtype", default="float16", choices=sorted(LL_DTYPE))
    ap.add_argument("--modes", default="fp32,tf32,bf16,fp16")
    ap.add_argument("--pi-rows", type=int, default=8192,
                    help="rows per node for the Pi-loop timing")
    ap.add_argument("--pi-nodes", type=int, default=8,
                    help="nodes to time; the full sweep is 2765 of these")
    ap.add_argument("--pi-samples", type=int, default=8)
    ap.add_argument("--closure-g", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=6.0, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.8,
                     moffat_beta=3.5).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    print(f"device={args.device}  torch={torch.__version__}", flush=True)
    if args.device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu={p.name}  {p.total_memory / 2**30:.1f} GiB  SMs={p.multi_processor_count}",
              flush=True)

    prior = build_prior(args)
    grid, cell = make_e_grid(n=args.grid_n, emax=args.grid_emax, rmax=args.grid_rmax)
    nodes = ShapeScoreNodes(grid, prior, delta=args.fd_delta, info_delta=args.info_delta)
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    df = load_g0(args.g0_catalogue, max(args.rows, args.pi_rows))
    print(f"grid: G={len(grid)}   rows: {len(df):,}", flush=True)

    # ---- one leg's worth of data, drawn exactly the way the driver draws it -------------
    rng = np.random.default_rng(args.seed)
    sub = df.iloc[:args.rows].copy()
    e1, e2 = prior.sample(len(sub), rng)
    sub["e1_input_rot0_p"], sub["e2_input_rot0_p"] = apply_shear_to_ellipticity(
        e1, e2, args.closure_g * np.ones(len(sub)), np.zeros(len(sub)))
    frame = rescale(sub, **rk)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ehat = bundle.sample(frame, n_samples=1, batch_size=args.batch_size)[:, 0, :]

    # ---- (1) precision: same rows, same nodes, four arithmetic settings -----------------
    est = PosteriorShapeEstimator(bundle, grid, device=args.device)
    evals = len(frame) * len(grid)
    ref = None
    print(f"\n=== score pass: {len(frame):,} rows x {len(grid):,} nodes = "
          f"{evals:,} flow evaluations ===", flush=True)
    print(f"{'mode':>6} {'seconds':>9} {'TFLOP/s':>9} {'speedup':>8} "
          f"{'d ghat1':>11} {'d ghat2':>11}", flush=True)
    for mode in args.modes.split(","):
        if mode != "fp32" and args.device != "cuda":
            continue
        torch.cuda.synchronize() if args.device == "cuda" else None
        t0 = time.time()
        with precision_region(mode, args.device):
            s, info = score_pass(est, nodes, frame, ehat, args.chunk, f"bench:{mode}",
                                 slab_mult=args.slab_mult, grad_delta=args.grad_delta,
                                 ll_dtype=LL_DTYPE[args.ll_dtype])
        torch.cuda.synchronize() if args.device == "cuda" else None
        dt = time.time() - t0
        gh = np.linalg.solve(info.sum(axis=0), s.sum(axis=0))
        if ref is None:
            ref, t_ref = gh, dt
            d = "     ref     ref"
            print(f"{mode:>6} {dt:9.1f} {2 * evals * MACS_PER_EVAL / dt / 1e12:9.2f} "
                  f"{1.0:8.2f} {d}", flush=True)
        else:
            print(f"{mode:>6} {dt:9.1f} {2 * evals * MACS_PER_EVAL / dt / 1e12:9.2f} "
                  f"{t_ref / dt:8.2f} {gh[0] - ref[0]:+11.2e} {gh[1] - ref[1]:+11.2e}",
                  flush=True)
    print(f"\nreference ghat = ({ref[0]:+.6f}, {ref[1]:+.6f})   truth g1 = "
          f"{args.closure_g:+.6f}\n  A mode is ACCEPTABLE if |d ghat| is far below the run's "
          f"statistical error, which for\n  the definitive 8M-object runs is 2.5e-4 in ghat "
          f"(0.25% of g=0.05).  Anything at 1e-5 or\n  below is irrelevant to the answer; "
          f"anything near 1e-4 is not.", flush=True)

    # ---- (2) the Pi loop: how much of it is not the flow --------------------------------
    print(f"\n=== Pi loop: {args.pi_rows:,} rows x {args.pi_samples} draws, "
          f"{args.pi_nodes} of {len(grid):,} nodes ===", flush=True)
    psub = df.iloc[:args.pi_rows].copy()
    ksel = np.linspace(0, len(grid) - 1, args.pi_nodes).astype(int)

    # (a) exactly the body of `pass_fraction_by_node`, per node
    t0 = time.time()
    for k in ksel:
        f = psub.copy()
        f["e1_input_rot0_p"] = grid[k, 0]
        f["e2_input_rot0_p"] = grid[k, 1]
        fr = rescale(f, **rk)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        bundle.sample(fr, n_samples=args.pi_samples, batch_size=args.batch_size)
    torch.cuda.synchronize() if args.device == "cuda" else None
    t_full = (time.time() - t0) / len(ksel)

    # (b) the frame work alone, no flow
    t0 = time.time()
    for k in ksel:
        f = psub.copy()
        f["e1_input_rot0_p"] = grid[k, 0]
        f["e2_input_rot0_p"] = grid[k, 1]
        rescale(f, **rk)
    t_frame = (time.time() - t0) / len(ksel)

    # (c) the flow alone, on a frame built once -- what the loop COULD cost
    fr1 = rescale(psub.copy(), **rk)
    bundle.sample(fr1, n_samples=args.pi_samples, batch_size=args.batch_size)  # warm
    torch.cuda.synchronize() if args.device == "cuda" else None
    t0 = time.time()
    for _ in ksel:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        bundle.sample(fr1, n_samples=args.pi_samples, batch_size=args.batch_size)
    torch.cuda.synchronize() if args.device == "cuda" else None
    t_flow = (time.time() - t0) / len(ksel)

    # (d) THE FAST PATH, and a proof that it is the same numbers.
    #
    # `ConditionalMeanFlow.sample` is `flow.sample(_flow_ctx(c)) + _mu(c)`, and `_flow_ctx`
    # index-selects away exactly the shape columns (`flow_drop_indices = [0,1,8,9]` = e1, e2
    # and their missing indicators).  So the residual flow -- the whole 10-layer coupling
    # stack, 590x the mean head's arithmetic -- is BLIND to the node.  Under the common
    # random numbers the loop already imposes, its output at node `k` is therefore not
    # merely similar to its output at node `k'`, it is bit-identical.  The 2765-node sweep
    # is one flow sample pass plus 2765 evaluations of a 16->128->2 MLP.
    #
    # This is the same location-family structure `PosteriorShapeEstimator` already asserts
    # for the scoring side; it just was never exploited on the sampling side.  Verified
    # here against the per-node reference rather than argued, because "exact" is a strong
    # claim and it is one array comparison to check.
    model, pre, tstd = bundle.model, bundle.condition_preprocessor, bundle.target_transform
    i1, i2 = pre.feature_names.index("e1_input_p"), pre.feature_names.index("e2_input_p")
    n_feat = len(pre.feature_names)
    fr1 = rescale(psub.copy(), **rk)
    ctx0 = torch.as_tensor(pre.transform_frame(fr1), dtype=torch.float32,
                           device=est.device)
    fctx = model._flow_ctx(ctx0)
    gs = np.stack([(grid[:, 0] - pre.means[i1]) / pre.scales[i1],
                   (grid[:, 1] - pre.means[i2]) / pre.scales[i2]], axis=1)

    torch.cuda.synchronize() if args.device == "cuda" else None
    t0 = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        resid = model.flow.sample(fctx, n_samples=args.pi_samples)     # (m, ns, 2), ONCE
        fast = []
        for k in ksel:
            c = ctx0.clone()
            c[:, i1], c[:, i2] = float(gs[k, 0]), float(gs[k, 1])
            if pre.add_missing_indicators:
                c[:, n_feat + i1] = c[:, n_feat + i2] = 0.0
            x = resid + model._mu(c)[:, None, :]
            fast.append(tstd.inverse_transform_array(
                x.reshape(-1, tstd.dim).cpu().numpy()).reshape(len(psub), args.pi_samples,
                                                               tstd.dim))
    torch.cuda.synchronize() if args.device == "cuda" else None
    t_fast = (time.time() - t0) / len(ksel)

    # exactness check against the loop as written, node by node
    worst = 0.0
    for j, k in enumerate(ksel):
        f = psub.copy()
        f["e1_input_rot0_p"], f["e2_input_rot0_p"] = grid[k, 0], grid[k, 1]
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        want = bundle.sample(rescale(f, **rk), n_samples=args.pi_samples,
                             batch_size=args.batch_size)
        worst = max(worst, float(np.abs(want - fast[j]).max()))

    sweep = lambda t: t * len(grid) / 60.0
    print(f"  per node, as written        {t_full:7.3f}s   -> {sweep(t_full):6.1f} min/replicate",
          flush=True)
    print(f"    of which frame rebuild    {t_frame:7.3f}s   ({t_frame / t_full:.0%})", flush=True)
    print(f"    of which flow sampling    {t_flow:7.3f}s   ({t_flow / t_full:.0%})", flush=True)
    print(f"  hoist the frame only        {t_flow:7.3f}s   -> {sweep(t_flow):6.1f} min "
          f"({t_full / t_flow:.1f}x)", flush=True)
    print(f"  + reuse the e-blind residual{t_fast:7.3f}s   -> {sweep(t_fast):6.1f} min "
          f"({t_full / max(t_fast, 1e-9):.0f}x)", flush=True)
    print(f"  max |fast - as written| over {len(ksel)} nodes x {args.pi_rows:,} rows x "
          f"{args.pi_samples} draws = {worst:.3e}", flush=True)
    print("  (this must be 0, or float-noise small: the residual flow cannot see the node, "
          "so\n   under common random numbers its draws at every node are the same draws.)",
          flush=True)
    print("\n### DONE ###", flush=True)


if __name__ == "__main__":
    main()
