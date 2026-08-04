"""INDEPENDENT audit of ConditionalMeanFlowRA. Written from scratch; does not import or
reuse tests/test_measurement_model.py.  Run:  python .scratch/audit_ra/audit_ra_head.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot")
from sbs_shear.measurement_model import (  # noqa: E402
    ConditionalMeanFlow, ConditionalMeanFlowRA, build_flow)

torch.set_default_dtype(torch.float64)   # audit in double where possible
RESULTS = []


def rep(name, ok, msg=""):
    RESULTS.append((name, ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {msg}")


# fiducial-like geometry: 4D target, linear mean head (mean_hidden=0), flow blind to shape ctx cols
KW = dict(target_dim=4, context_dim=6, base_flow="affine", mean_hidden=0,
          hidden_dim=16, n_layers=2, n_flows=6, flow_drop_indices=(0, 1))


def make_pair(seed=7, ra_hidden=12, mean_hidden=None):
    kw = dict(KW)
    if mean_hidden is not None:
        kw["mean_hidden"] = mean_hidden
    torch.manual_seed(seed)
    ra = ConditionalMeanFlowRA(ra_hidden=ra_hidden, ra_indices=(2, 3), ra_targets=(0, 1), **kw)
    torch.manual_seed(seed)
    plain = ConditionalMeanFlow(**kw)
    # force EXACT weight sharing: copy every non-RA tensor from ra into plain and back-check
    sd = {k: v.clone() for k, v in ra.state_dict().items() if not k.startswith("ra_")}
    missing, unexpected = plain.load_state_dict(sd, strict=True), None
    ra.eval(); plain.eval()
    return ra, plain


def randomise_A(ra, scale=0.5, seed=99):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ra.ra_net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)


# ---------------------------------------------------------------- 1. exact reduction
def t1_exact_reduction():
    for mh in (0, 8):
        ra, plain = make_pair(mean_hidden=mh)
        x = torch.randn(9, 4)
        c = torch.randn(9, 6)
        lp_ra, lp_pl = ra.log_prob(x, c), plain.log_prob(x, c)
        same_lp = torch.equal(lp_ra, lp_pl)
        torch.manual_seed(1234); s_ra = ra.sample(c, n_samples=7)
        torch.manual_seed(1234); s_pl = plain.sample(c, n_samples=7)
        same_s = torch.equal(s_ra, s_pl)
        # also qmc path
        torch.manual_seed(5); q_ra = ra.sample(c, n_samples=4, qmc=True)
        torch.manual_seed(5); q_pl = plain.sample(c, n_samples=4, qmc=True)
        same_q = torch.equal(q_ra, q_pl)
        rep(f"1.exact_reduction(mean_hidden={mh}) log_prob bitwise", same_lp,
            f"max|d|={float((lp_ra - lp_pl).abs().max()):.3e}")
        rep(f"1.exact_reduction(mean_hidden={mh}) sample bitwise", same_s,
            f"max|d|={float((s_ra - s_pl).abs().max()):.3e}")
        rep(f"1.exact_reduction(mean_hidden={mh}) sample(qmc) bitwise", same_q, "")
    # float32 (the real training dtype)
    torch.set_default_dtype(torch.float32)
    ra, plain = make_pair()
    x = torch.randn(9, 4); c = torch.randn(9, 6)
    rep("1.exact_reduction(float32) log_prob bitwise", torch.equal(ra.log_prob(x, c), plain.log_prob(x, c)))
    torch.manual_seed(2); a = ra.sample(c, 5)
    torch.manual_seed(2); b = plain.sample(c, 5)
    rep("1.exact_reduction(float32) sample bitwise", torch.equal(a, b))
    torch.set_default_dtype(torch.float64)
    # warm start realism: load a plain state dict into a fresh RA model with strict=False
    torch.manual_seed(31)
    donor = ConditionalMeanFlow(**KW)
    torch.manual_seed(77)
    fresh = ConditionalMeanFlowRA(ra_hidden=12, ra_indices=(2, 3), ra_targets=(0, 1), **KW)
    miss, unexp = fresh.load_state_dict(donor.state_dict(), strict=False)
    fresh.eval(); donor.eval()
    x = torch.randn(11, 4); c = torch.randn(11, 6)
    ok = torch.equal(fresh.log_prob(x, c), donor.log_prob(x, c)) and len(unexp) == 0 \
        and all(k.startswith("ra_") for k in miss)
    rep("1.warm_start strict=False is exact", ok, f"missing={list(miss)} unexpected={list(unexp)}")


# ---------------------------------------------------------------- 2. Jacobian / normalisation
def t2_jacobian_and_norm():
    ra, _ = make_pair()
    randomise_A(ra, scale=0.8)
    c = torch.randn(1, 6)

    def fwd(x):                       # the map used inside log_prob: x -> z
        r = x - ra._mu(c)
        u = r.index_select(-1, ra.ra_indices)
        return (r - ra._A(c, u)).squeeze(0)

    dets, offdiag_bad, diag_bad = [], 0.0, 0.0
    for k in range(6):
        x = torch.randn(1, 4) * 1.5
        J = torch.autograd.functional.jacobian(fwd, x).reshape(4, 4)
        dets.append(float(torch.det(J)))
        diag_bad = max(diag_bad, float((torch.diagonal(J) - 1.0).abs().max()))
        # triangular claim: A writes 0,1 and reads 2,3 -> dz_{0,1}/dx_{2,3} may be nonzero,
        # everything else off-diagonal must be exactly 0
        M = J.clone()
        M[0, 2] = M[0, 3] = M[1, 2] = M[1, 3] = 0.0
        M = M - torch.eye(4)
        offdiag_bad = max(offdiag_bad, float(M.abs().max()))
    rep("2.jacobian det==1 (autograd, full 4x4)", max(abs(d - 1) for d in dets) < 1e-10,
        f"max|det-1|={max(abs(d - 1) for d in dets):.2e}")
    rep("2.jacobian structure is exactly triangular", offdiag_bad < 1e-12 and diag_bad < 1e-12,
        f"max stray offdiag={offdiag_bad:.2e}, max|diag-1|={diag_bad:.2e}")
    # is the coupling actually non-trivial? (otherwise det==1 is vacuous)
    x = torch.randn(1, 4)
    J = torch.autograd.functional.jacobian(fwd, x).reshape(4, 4)
    rep("2.coupling block is NON-zero (test not vacuous)",
        float(J[:2, 2:].abs().max()) > 1e-3, f"max|dz01/dx23|={float(J[:2, 2:].abs().max()):.3f}")

    # change-of-variables consistency: log_prob(T(z)) must equal flow.log_prob(z)
    torch.manual_seed(404)
    c8 = torch.randn(8, 6)
    x = ra.sample(c8, n_samples=6)
    B, N, D = x.shape
    cflat = c8[:, None, :].expand(B, N, 6).reshape(-1, 6)
    lp_model = ra.log_prob(x.reshape(-1, D), cflat)
    # recover z through the forward map and score it under the base flow directly
    r = x.reshape(-1, D) - ra._mu(cflat)
    z = r - ra._A(cflat, r.index_select(-1, ra.ra_indices))
    lp_base = ra.flow.log_prob(z, ra._flow_ctx(cflat))
    rep("2.change-of-variables: log_prob(x) == flow.log_prob(z) (no log-det needed)",
        float((lp_model - lp_base).abs().max()) < 1e-12,
        f"max|d|={float((lp_model - lp_base).abs().max()):.2e}")

    # brute-force normalisation by importance sampling in the full 4D
    torch.manual_seed(7)
    c1 = torch.randn(1, 6)
    with torch.no_grad():
        s = ra.sample(c1.expand(1, 6), n_samples=200000).reshape(-1, 4)
        m, sd = s.mean(0), s.std(0)
        n = 2000000
        eps = torch.randn(n, 4)
        q = m + 2.0 * sd * eps                                  # inflated Gaussian proposal
        logq = (-0.5 * (eps ** 2).sum(1) - 0.5 * 4 * np.log(2 * np.pi)
                - torch.log(2.0 * sd).sum())
        lp = ra.log_prob(q, c1.expand(n, 6))
        w = torch.exp(lp - logq)
        Z = float(w.mean()); Zerr = float(w.std() / np.sqrt(n))
    rep("2.density normalises to 1 (IS, N=2e6, 4D)", abs(Z - 1.0) < 5 * Zerr + 5e-3,
        f"Z = {Z:.5f} +- {Zerr:.5f}")

    # sanity: if the SAME model is scored WITH a log-det term it would NOT normalise -- i.e. show
    # that the check has teeth by breaking triangularity on purpose.
    bad = ConditionalMeanFlowRA(ra_hidden=12, ra_indices=(2, 3), ra_targets=(0, 1), **KW)
    randomise_A(bad, scale=0.8)
    with torch.no_grad():                # forcibly make A read+write channel 0 (illegal config)
        bad.ra_indices.copy_(torch.tensor([0, 3]))
    def fwd_bad(x):
        r = x - bad._mu(c)
        return (r - bad._A(c, r.index_select(-1, bad.ra_indices))).squeeze(0)
    Jb = torch.autograd.functional.jacobian(fwd_bad, torch.randn(1, 4)).reshape(4, 4)
    rep("2.control: an overlapping (illegal) config really does break det==1",
        abs(float(torch.det(Jb)) - 1.0) > 1e-3, f"det={float(torch.det(Jb)):.4f}")


# ---------------------------------------------------------------- 4. swallow trap
def t4_swallow():
    good = dict(flow_type="mean_affine_ra", target_dim=4, context_dim=6, mean_hidden=0,
                hidden_dim=16, n_layers=2, n_flows=6, ra_hidden=12,
                ra_indices=[2, 3], ra_targets=[0, 1])
    m = build_flow(dict(good))
    rep("4.build_flow dispatches mean_affine_ra", isinstance(m, ConditionalMeanFlowRA))
    rep("4.A is zero at construction", float(m.ra_net[-1].weight.abs().sum()) == 0.0
        and float(m.ra_net[-1].bias.abs().sum()) == 0.0)

    for typo, val in (("ra_hiden", 12), ("ra_target", [0, 1]), ("ra_indicies", [2, 3]),
                      ("ra_activaton", "gelu"), ("nonsense_key", 3)):
        cfg = dict(good); cfg.pop({"ra_hiden": "ra_hidden", "ra_target": "ra_targets",
                                   "ra_indicies": "ra_indices"}.get(typo, "__none__"), None)
        cfg[typo] = val
        try:
            mm = build_flow(cfg)
            raised = False
        except Exception as e:                                    # noqa: BLE001
            raised = True; err = type(e).__name__
        if raised:
            rep(f"4.typo {typo!r} raises", True, err)
        else:
            rep(f"4.typo {typo!r} raises", False,
                f"SWALLOWED -> built {type(mm).__name__} "
                f"ra_hidden={getattr(mm, 'ra_hidden', None)} "
                f"ra_indices={mm.ra_indices.tolist()} ra_targets={mm.ra_targets.tolist()}")

    # older code copy: does a HEAD-vintage build_flow reject the new flow_type loudly?
    rep("4.unknown flow_type still raises", _raises(lambda: build_flow(dict(good, flow_type="junk"))))


def _raises(fn):
    try:
        fn()
    except Exception:                                             # noqa: BLE001
        return True
    return False


# ---------------------------------------------------------------- 5. SWA safety
def t5_swa():
    m = build_flow(dict(flow_type="mean_affine_ra", target_dim=4, context_dim=6, mean_hidden=0,
                        hidden_dim=16, n_layers=2, n_flows=6, ra_hidden=12,
                        ra_indices=[2, 3], ra_targets=[0, 1]))
    bn = [type(x).__name__ for x in m.modules()
          if isinstance(x, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm,
                            torch.nn.GroupNorm, torch.nn.Dropout))]
    rep("5.no BatchNorm/LayerNorm/Dropout anywhere in the RA model", not bn, f"found={bn}")
    ra_bufs = {k: v.dtype for k, v in m.named_buffers() if k.startswith("ra_")}
    rep("5.RA buffers are integer (copied, not averaged, by the SWA code)",
        all(v == torch.int64 for v in ra_bufs.values()), f"{ra_bufs}")
    ra_pars = {k: tuple(v.shape) for k, v in m.named_parameters() if k.startswith("ra_")}
    rep("5.RA parameters are plain float Linear weights", len(ra_pars) == 4, f"{ra_pars}")

    # emulate the trainer's SWA average over 3 snapshots and check it loads + is the mean
    snaps = []
    for i in range(3):
        with torch.no_grad():
            for p in m.ra_net.parameters():
                p.copy_(torch.randn(p.shape))
        snaps.append({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    newest = snaps[-1]; swa = {}
    for k, ref in newest.items():
        if torch.is_floating_point(ref):
            swa[k] = torch.stack([s[k].to(torch.float64) for s in snaps], 0).mean(0).to(ref.dtype)
        else:
            swa[k] = ref.clone()
    m.load_state_dict(swa)
    want = torch.stack([s["ra_net.2.weight"].to(torch.float64) for s in snaps], 0).mean(0)
    rep("5.SWA average of ra_net is the elementwise mean and loads cleanly",
        float((m.ra_net[2].weight.double() - want).abs().max()) < 1e-12)
    rep("5.SWA keeps the long RA buffers intact",
        m.ra_indices.tolist() == [2, 3] and m.ra_targets.tolist() == [0, 1])


# ---------------------------------------------------------------- extra: does A move the response?
def t_extra_response():
    ra, plain = make_pair()
    randomise_A(ra, scale=0.4)
    c0 = torch.randn(5, 6)
    c1 = c0.clone(); c1[:, 0] += 0.01     # shear-shifted shape context columns (flow-blind)
    for mdl, tag in ((plain, "plain"), (ra, "RA")):
        torch.manual_seed(55); a = mdl.sample(c0, n_samples=256)
        torch.manual_seed(55); b = mdl.sample(c1, n_samples=256)
        d = (b - a)[:, :, 0] / 0.01
        print(f"    {tag}: per-object per-draw response spread "
              f"std={float(d.std(dim=1).mean()):.4f}  mean={float(d.mean()):.4f}")
    torch.manual_seed(55); a = ra.sample(c0, n_samples=256)
    torch.manual_seed(55); b = ra.sample(c1, n_samples=256)
    d = (b - a)[:, :, 0] / 0.01
    rep("X.RA response varies draw-to-draw within an object", float(d.std(dim=1).min()) > 1e-6)
    # and the mean-head-only estimate (what _mu would give) is NOT the same number
    mu_resp = ((ra._mu(c1) - ra._mu(c0))[:, 0] / 0.01)
    full = d.mean(dim=1)
    rep("X.mu-only response != true RA response (so every _mu call site matters)",
        float((full - mu_resp).abs().max()) > 1e-3,
        f"max|R_full - R_mu| = {float((full - mu_resp).abs().max()):.4f}")


if __name__ == "__main__":
    t1_exact_reduction()
    t2_jacobian_and_norm()
    t4_swallow()
    t5_swa()
    t_extra_response()
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n==== {len(RESULTS) - nfail}/{len(RESULTS)} PASS, {nfail} FAIL ====")
    for n, ok, msg in RESULTS:
        if not ok:
            print(f"  FAIL: {n}  {msg}")
