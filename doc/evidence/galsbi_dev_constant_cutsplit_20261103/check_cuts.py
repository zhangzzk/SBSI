import importlib.util
import numpy as np
import torch
import scripts.evaluate_smooth_classifier_joint as ev

spec = importlib.util.spec_from_file_location(
    'egc', '/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/scripts/evaluate_galsbi_constant.py')
egc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(egc)

with np.load(egc.OUT/'parent/case1000.npz') as z:
    ids = z['input_index']
values, usable, _ = ev.measured_leg(egc.SIMS, 1000, egc.H, ids)
mine, frozen = egc.output_cuts(values) & usable[:, None], ev.output_cuts(values) & usable[:, None]
print('measured: first three rows identical:', np.array_equal(mine[:, :3], frozen), ' kept per cut:',
      dict(zip(egc.CUT_NAMES, mine.sum(0).tolist())))

g = torch.Generator().manual_seed(1)
v = torch.rand(512, 64, 4, generator=g, dtype=torch.float64)
v[..., 2] = 2+3*v[..., 2]
v[..., 3] = 10**(-0.4*(24.5+2*v[..., 3]-30))
shift = 0.01*torch.randn(512, 64, 2, generator=g, dtype=torch.float64)
k = egc.output_cuts(v)
print('generated: first three rows identical:', torch.equal(k[..., :3], ev.output_cuts(v)))
a = [x.numpy() for x in egc.draw_moments(k, v, shift)]
b = [x.numpy() for x in ev.selected_draw_moments(v, shift)]
print('draw moments, max |diff| over the frozen rows:', max(np.abs(x[:, :3]-y).max() for x, y in zip(a, b)))
