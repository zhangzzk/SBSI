import numpy as np, torch, sys
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot")
from sbs_shear.measurement_model import load_measurement_model
p = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt"
b = load_measurement_model(p, device="cpu")
m = b.model
print("model class:", type(m).__name__)
print("keep_indices:", m.keep_indices.tolist())
print("context_dim:", m.context_dim, "target_dim:", m.target_dim)
print("mean_net:", m.mean_net)
print("n coupling layers:", len(m.flow.layers), "flow context_dim:", m.flow.context_dim)
nbuf = [(n, tuple(v.shape), str(v.dtype)) for n, v in m.named_buffers()]
print("buffers:", nbuf[:3], "... total", len(nbuf))
print("any BatchNorm/running stats:", any("running_" in n for n,_ in m.named_buffers()))

torch.manual_seed(0)
N = 2000
ctx = torch.randn(N, 16) * 0.5
ctx[:, 8:] = 0.0            # missing indicators off
d = 0.02
ctxp = ctx.clone(); ctxp[:, 0] += d
ctxm = ctx.clone(); ctxm[:, 0] -= d
def draw(c, seed, n=64):
    torch.manual_seed(seed)
    with torch.no_grad():
        return m.sample(c, n_samples=n).mean(dim=1)
sp = draw(ctxp, 123); sm = draw(ctxm, 123)
r_sample = ((sp - sm) / (2*d))[:, 0]
with torch.no_grad():
    r_mu = ((m._mu(ctxp) - m._mu(ctxm)) / (2*d))[:, 0]
print("\nCRN sample-based dR/dc0 vs mu-based: max|diff| =", float((r_sample - r_mu).abs().max()))
# different seeds per leg
sp2 = draw(ctxp, 1); sm2 = draw(ctxm, 2)
r_sample2 = ((sp2 - sm2) / (2*d))[:, 0]
print("independent-seed version: max|diff| =", float((r_sample2 - r_mu).abs().max()),
      " mean diff =", float((r_sample2 - r_mu).mean()))
# does the residual flow mean depend on shape columns at all?
with torch.no_grad():
    torch.manual_seed(7); f0 = m.flow.sample(m._flow_ctx(ctx), n_samples=32).mean(1)
    torch.manual_seed(7); f1 = m.flow.sample(m._flow_ctx(ctxp), n_samples=32).mean(1)
print("flow residual mean identical under shape shift:", bool(torch.equal(f0, f1)))
