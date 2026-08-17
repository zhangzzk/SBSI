"""Call-site classification, executed rather than read: does each hand-rolled `x - mu(c)` /
`_mu` site FAIL LOUDLY under a ConditionalMeanFlowRA, or does it silently return a mu-only number?
Also: checkpoint save/load round-trip for the RA head.
"""
import os
import sys
import tempfile
import numpy as np
import torch

ROOT = "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot"
sys.path.insert(0, ROOT)
from sbs_shear.measurement_model import (  # noqa: E402
    ConditionalMeanFlow, ConditionalMeanFlowRA, build_flow,
    save_measurement_model, load_measurement_model, TargetStandardizer)

RESULTS = []


def rep(name, ok, msg=""):
    RESULTS.append((name, ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {msg}")


KW = dict(target_dim=4, context_dim=6, base_flow="affine", mean_hidden=0,
          hidden_dim=16, n_layers=2, n_flows=4, flow_drop_indices=(0, 1))
CFG = dict(flow_type="mean_affine_ra", target_dim=4, context_dim=6, mean_hidden=0,
           hidden_dim=16, n_layers=2, n_flows=4, ra_hidden=12,
           ra_indices=[2, 3], ra_targets=[0, 1], flow_drop_indices=[0, 1])


def ra_model(seed=3, scale=0.5):
    torch.manual_seed(seed)
    m = build_flow(dict(CFG))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.ra_net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    m.eval()
    return m


# ---------------------------------------------------------------- posterior_shape fence
def t_posterior_fence():
    from sbs_shear.posterior_shape import PosteriorShapeEstimator

    class Stub:
        def __init__(self, m):
            self.model = m
            self.device = torch.device("cpu")
            self.condition_preprocessor = None
            self.target_transform = None
    try:
        PosteriorShapeEstimator(Stub(ra_model()), grid=None)
        ok, why = False, "constructed WITHOUT raising"
    except NotImplementedError as e:
        ok, why = True, "NotImplementedError"
    except Exception as e:                                             # noqa: BLE001
        ok, why = False, f"raised the WRONG error first: {type(e).__name__}: {e}"
    rep("3.posterior_shape.PosteriorShapeEstimator refuses an RA model", ok, why)
    # and the plain model still constructs far enough to reach the real checks
    try:
        PosteriorShapeEstimator(Stub(ConditionalMeanFlow(**KW)), grid=None)
        why = "constructed"
    except NotImplementedError as e:
        why = f"WRONGLY refused a plain mean_affine: {e}"
    except Exception as e:                                             # noqa: BLE001
        why = f"reached a later check ({type(e).__name__}) -- fence did not fire early"
    rep("3.posterior_shape fence does NOT fire for the plain head",
        "WRONGLY" not in why, why)


# ---------------------------------------------------------------- forward_model fence
def t_forward_fence():
    import inspect
    from sbs_shear import forward_model as fm
    src = inspect.getsource(fm.SetConditionedForwardModel.__init__)
    builds_plain = "self.mean_flow = ConditionalMeanFlow(" in src
    rep("3.forward_model fence is unreachable-by-construction (defensive only)", builds_plain,
        "constructor hard-codes ConditionalMeanFlow, so the isinstance guard can never fire; "
        "harmless, but it is NOT what protects mu()/log_prob_obs()")


# ---------------------------------------------------------------- unfenced script sites
def t_unfenced_sites():
    m = ra_model()
    c0 = torch.randn(64, 6)
    d = 0.01
    cp = c0.clone(); cp[:, 0] += d
    cm = c0.clone(); cm[:, 0] -= d
    r_mu = float(((m._mu(cp) - m._mu(cm))[:, 0] / (2 * d)).mean())
    # the response the model ACTUALLY has (CRN sampling = the scoring path)
    with torch.no_grad():
        torch.manual_seed(4); sp = m.sample(cp, n_samples=512)
        torch.manual_seed(4); sm = m.sample(cm, n_samples=512)
    r_true = float(((sp - sm)[:, :, 0] / (2 * d)).mean())
    rep("3.mu-only shape response is NOT the model's response (quantifies the silent-failure size)",
        abs(r_true - r_mu) > 1e-3,
        f"R via _mu = {r_mu:+.4f}   R via sample (scoring path) = {r_true:+.4f}   "
        f"error = {100 * (r_mu / r_true - 1):+.1f}%")
    # dims 2,3 really are untouched by A (so the coupling pin and R_mag/R_size stay exact)
    with torch.no_grad():
        u = torch.randn(64, 2)
        A = m._A(c0, u)
    rep("3.A is EXACTLY zero on dims 2,3 (coupling pin + R_mag/R_size via _mu remain exact)",
        float(A[:, 2:].abs().max()) == 0.0 and float(A[:, :2].abs().max()) > 0,
        f"max|A[:,2:]|={float(A[:, 2:].abs().max()):.1e}, max|A[:,:2]|={float(A[:, :2].abs().max()):.3f}")

    for path in ("scripts/eval_selfresp_gap.py", "scripts/build_mu_correction.py",
                 "scripts/eval_fluxsize_response.py"):
        src = open(os.path.join(ROOT, path)).read()
        fenced = "ConditionalMeanFlowRA" in src or "NotImplementedError" in src
        print(f"    {path}: uses model._mu -> "
              f"{'has some RA guard' if fenced else 'NO RA GUARD AT ALL'}")


# ---------------------------------------------------------------- checkpoint round trip
def t_roundtrip():
    m = ra_model()
    ts = TargetStandardizer(["a", "b", "c", "d"], np.zeros(4), np.ones(4))

    class PP:
        feature_names = [f"f{i}" for i in range(6)]
        output_dim = 6
        def to_state(self):
            return {"feature_names": self.feature_names}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ra.pt")
        try:
            save_measurement_model(p, m, PP(), ts, dict(CFG), metadata={"tag": "audit"})
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m2 = build_flow(dict(ck["model_config"]))
            missing, unexpected = m2.load_state_dict(ck["state_dict"], strict=True), None
            m2.eval()
            x = torch.randn(9, 4); c = torch.randn(9, 6)
            ok = torch.equal(m.log_prob(x, c), m2.log_prob(x, c))
            rep("3.RA checkpoint save -> build_flow(model_config) -> load_state_dict is exact", ok)
        except Exception as e:                                        # noqa: BLE001
            rep("3.RA checkpoint save/load round trip", False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    t_posterior_fence()
    t_forward_fence()
    t_unfenced_sites()
    t_roundtrip()
    nf = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n==== {len(RESULTS) - nf}/{len(RESULTS)} PASS, {nf} FAIL ====")
