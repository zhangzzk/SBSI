"""GALAXY-LEVEL anatomy of the 5C information-equality failure (completeness probe).

Every probe so far reported the failure as ONE population number, Var(s-<s>_sel) vs
<I>-I_sel, and dissected it at NODE level.  Two questions were never asked:

  (1) is the excess UNIFORM across the galaxy population, or concentrated in a
      sub-population (faint / small / low-Pdet / low-ESS / close-blend)?  A concentrated
      failure and a uniform failure are different diagnoses.
  (2) is Var(s) carried by a few OUTLIER GALAXIES, the way `tail_diagnostics` asked at
      node level?  If a handful of galaxies carry the second moment, the Bartlett
      denominator sum (s_i-<s>_sel)^2 is a few-object statistic.

CAVEAT, stated up front.  The identity Var(s) = E[I] is exact only over the FULL joint
(scene ~ p0, data ~ the estimator's own model) at gamma = 0.  Conditioning on the
galaxy's own true properties is conditioning on the latent, which the identity does not
license, so a per-bin ratio is a LOCALISER, not a test.  The trimmed global numbers ARE
a fair test of "does a small set of galaxies drive the global ratio".

Also runs the whole thing on a SECOND CHECKPOINT (another training seed, and optionally
another architecture): no probe in the dossier used more than one.

The xhat draw IS seeded here.  `closure_v2_lagrangian.py` never calls torch.manual_seed
before `mean_flow.sample`, so its data realisation is fresh on every invocation.
"""

import argparse
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import (  # noqa: E402
    CAT, CKPT, load_rows, phi_block, rebuild, scene_context,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency, posterior_weights, score_and_information,
    selection_terms, shear_estimate_bartlett, shear_estimate_louis,
)
from train_joint_forward import intrinsic_shape  # noqa: E402


def intr_of(df):
    return dict(zip(("e1p", "e2p"), intrinsic_shape(df, "p"))) | \
           dict(zip(("e1s", "e2s"), intrinsic_shape(df, "s")))


def summarize(c, info, i_sel, tag):
    """c = s - <s>_sel already centred."""
    var = float(np.mean(c ** 2))
    lou = float(np.mean(info) - i_sel)
    return (f"{tag:<26} N={c.size:6d}  Var(s-sel)={var:10.3f}  <I>-I_sel={lou:10.3f}  "
            f"ratio={var / lou if lou != 0 else np.nan:+10.3f}  "
            f"ghat5.9={c.sum() / np.sum(c ** 2):+.5f}")


def run_one(ck_path, args, dev):
    print(f"\n{'=' * 100}\nCHECKPOINT {os.path.basename(ck_path)}\n{'=' * 100}")
    model, pre, nbr_std, target_std, meta = rebuild(ck_path, dev)
    print(f"  metadata: {meta}")

    need = args.gal_offset + args.n_gal
    rows = load_rows(args.catalogue, need * 6)
    tc = meta.get("true_cut")
    if tc is not None:
        keep_true = ((rows["Re_input_p"].to_numpy(float) > float(tc[0]))
                     & (rows["r_input_p"].to_numpy(float) < float(tc[1])))
        rows = rows[keep_true].reset_index(drop=True)
    node_df = rows.iloc[:args.n_node].reset_index(drop=True)
    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    print(f"  rows: {len(node_df):,} node + {len(gal_df):,} galaxy scenes  "
          f"(distinct bank primaries {node_df['input_index'].nunique():,})")

    node_intr, gal_intr = intr_of(node_df), intr_of(gal_df)
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    kept = keep.cpu().numpy().astype(bool)
    xhat = xhat[keep]
    print(f"  detection: kept {int(kept.sum()):,}/{len(gal_df):,}  "
          f"(<Pdet>={float(pdet_g.mean()):.4f})   gamma_true={args.gamma}")

    d = args.delta
    phis, pdets = {}, {}
    for t in (0.0, +d, -d):
        c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
        phis[t] = phi_block(model, xhat, c, torch.log(pd_), args.chunk)
        pdets[t] = float(pd_.mean())
    p0, pp, pm = phis[0.0], phis[+d], phis[-d]
    d1 = (pp - pm) / (2 * d)
    d2 = (pp - 2 * p0 + pm) / d ** 2
    s, info = score_and_information(p0, d1, d2)
    lp = {t: np.log(v) for t, v in pdets.items()}
    s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                   (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
    w = posterior_weights(p0)
    ess = 1.0 / np.sum(w ** 2, axis=1)
    c = s - s_sel

    print(f"\n  s_sel={s_sel:+.5f}  I_sel={i_sel:+.5f}  ESS mean={ess.mean():.1f} "
          f"median={np.median(ess):.1f}")
    print("  " + summarize(c, info, i_sel, "GLOBAL"))
    print(f"  ghat(5.8)={shear_estimate_louis(s, info, s_sel, i_sel):+.5f}   "
          f"ghat(5.9)={shear_estimate_bartlett(s, s_sel):+.5f}")

    # ---- (2) galaxy-level concentration ------------------------------------------------
    print("\n  --- GALAXY-LEVEL CONCENTRATION of the Bartlett denominator sum (s_i-s_sel)^2")
    q = np.percentile(c, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    print("   quantiles of (s_i - s_sel): " + "  ".join(f"{v:+.3f}" for v in q)
          + f"   min={c.min():+.3f} max={c.max():+.3f}")
    o = np.argsort(-np.abs(c))
    tot = float(np.sum(c ** 2))
    n = c.size
    for k in (1, 5, 10, max(1, n // 100), max(1, n // 20), max(1, n // 10)):
        print(f"   top {k:5d} galaxies by |s-s_sel| ({k / n:6.2%} of N): "
              f"{float(np.sum(c[o[:k]] ** 2)) / tot:7.2%} of sum (s-s_sel)^2, "
              f"{float(np.sum(c[o[:k]])) / float(np.sum(c)):+8.2%} of sum (s-s_sel)")
    oi = np.argsort(-np.abs(info))
    ti = float(np.sum(np.abs(info)))
    for k in (1, 10, max(1, n // 100)):
        print(f"   top {k:5d} galaxies by |I_i| ({k / n:6.2%}): "
              f"{float(np.sum(np.abs(info[oi[:k]]))) / ti:7.2%} of sum|I_i|")

    print("\n  --- TRIMMED (drop the largest |s-s_sel| galaxies; a fair global test)")
    for frac in (0.001, 0.005, 0.01, 0.05, 0.10):
        m = np.ones(n, bool)
        m[o[:int(np.ceil(frac * n))]] = False
        print("   " + summarize(c[m], info[m], i_sel, f"trim {frac:.1%}"))

    # ---- (1) binned localisation --------------------------------------------------------
    print("\n  --- BINNED (localiser, NOT an exact test -- see the module docstring)")
    gd = gal_df.iloc[np.flatnonzero(kept)].reset_index(drop=True)
    e1p, e2p = intrinsic_shape(gd, "p")
    axes = {
        "mag r_input_p": gd["r_input_p"].to_numpy(float),
        "size Re_input_p": gd["Re_input_p"].to_numpy(float),
        "Pdet": pdet_g.cpu().numpy()[kept],
        "ESS": ess,
        "nbr distance": gd["distance"].to_numpy(float),
        "|e| primary": np.hypot(e1p, e2p),
        "dmag nbr-prim": gd["r_input_s"].to_numpy(float) - gd["r_input_p"].to_numpy(float),
    }
    for name, v in axes.items():
        finite = np.isfinite(v)
        edges = np.nanpercentile(v[finite], [0, 20, 40, 60, 80, 100])
        edges = np.unique(edges)
        if len(edges) < 3:
            print(f"   {name}: degenerate (constant), skipped")
            continue
        print(f"   axis {name}:")
        for j in range(len(edges) - 1):
            m = finite & (v >= edges[j]) & (v <= edges[j + 1] if j == len(edges) - 2
                                            else v < edges[j + 1])
            if m.sum() < 20:
                continue
            print("     " + summarize(c[m], info[m], i_sel,
                                      f"[{edges[j]:.3g},{edges[j + 1]:.3g})"))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default=CKPT,
                    help="comma-separated; the dossier used exactly one")
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=4000)
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--gal-offset", type=int, default=20000)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    print(f"device={args.device}  torch={torch.__version__}")
    if args.gal_offset < args.n_node:
        raise SystemExit("--gal-offset must be >= --n-node")
    for ck in args.checkpoints.split(","):
        run_one(ck.strip(), args, dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
