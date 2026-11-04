"""Synthetic checks of the grid-cut arm of scripts/run_galsbi_guard_strength.py.

Needs the frozen F8 checkout's sbsi (flow_guard_response); run with its interpreter:
  cd /tmp && <frozen venv>/bin/python -I -m pytest <SBSI>/tests/test_grid_cut_arm.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
guard = pytest.importorskip('sbsi.flow_guard_response')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_galsbi_guard_strength as g  # noqa: E402


def test_grid_cut_arm(monkeypatch):
    monkeypatch.setattr(g, 'GRID_MIN_MASS', 5.0)
    monkeypatch.setattr(g, 'GRID_MIN_CASES', 2)
    rng = np.random.default_rng(1)
    n, cases = 60_000, 10
    ctx = np.zeros((2 * n, 8))
    ctx[:n, [3, 4, 6]] = np.column_stack([rng.uniform(22, 26, n), rng.uniform(0.1, 0.6, n), rng.exponential(0.1, n)])
    ctx[n:] = ctx[:n]
    phi = rng.uniform(0, np.pi, n)
    gamma = 0.05 * np.column_stack([np.cos(2 * phi), np.sin(2 * phi)])
    zero = np.column_stack([rng.normal(0, 0.3, (n, 2)), rng.uniform(2.0, 4.5, n), 10 ** (0.4 * (30 - rng.uniform(23.5, 26.5, n)))])
    shear = zero.copy()
    shear[:, :2] += 0.8 * gamma + rng.normal(0, 0.01, (n, 2))
    shear[:, 2] *= 1 + 2 * (gamma * zero[:, :2]).sum(1)          # size responds to shear, so objects cross the cut
    shear[:, 3] *= 1 + 3 * (gamma * zero[:, :2]).sum(1)
    data = {'pairs': np.column_stack([np.arange(n), np.arange(n, 2 * n)]), 'pair_zero_target': zero,
            'pair_shear_target': shear, 'gamma': gamma, 'context_raw': ctx, 'pair_case': rng.integers(0, cases, n)}
    kw = dict(radius_softness=0.05, magnitude_softness=0.05)
    w0 = guard.numpy_guard_weights(zero, (g.GRID_CUT,), **kw)[:, 0]
    edges = g.grid_edges(ctx[:n][:, [3, 4, 6]], w0)
    assert edges.shape == (3, g.GRID_BINS + 1)

    # 1. sums-based cell response == frozen fit_guard_response on the cell's pairs alone
    sums, cell = g.grid_sums(data, edges, kw)
    resp, mass, mean, _ = g.grid_response(sums.sum(0))
    worst = 0.0
    for c in np.unique(cell)[:40]:
        m = cell == c
        ref = guard.fit_guard_response(zero[m], shear[m], gamma[m], (g.GRID_CUT,), **kw)
        worst = max(worst, np.abs(ref['target'][0] - resp[c]).max(), abs(ref['baseline_mass'][0] - mass[c]))
    print('max |sums - fit_guard_response| over 40 cells:', worst)
    assert worst < 1e-10

    # 2. a "flow" returning the simulated measurements reproduces every kept cell's target exactly
    table = torch.as_tensor(np.concatenate([zero, shear]))
    class Lookup(torch.nn.Module):
        target_dim, context_dim = 4, 1
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(()))
        def forward_physical(self, base, context):
            v = table[context[:, 0].long()]
            return torch.cat([v[:, :2] * self.p + 0 * base[:, :2], v[:, 2:]], dim=1), None
    context = torch.arange(2 * n, dtype=torch.float64)[:, None]
    pop = g.GridCutPopulation(context, data, edges, kw)
    print('cells kept', len(pop.cells), 'of', g.GRID_BINS ** 3, 'dropped', {k: len(v) for k, v in pop.statistics['cells_dropped'].items()})
    model = Lookup()
    with torch.no_grad():
        idx = torch.arange(n)
        keep = pop.pair_slot[idx] >= 0
        total = torch.zeros(len(pop.cells), 4, dtype=torch.float64).index_add_(
            0, pop.pair_slot[idx[keep]], pop.contributions(model, idx[keep], 2, 7))
        count = torch.bincount(pop.pair_slot[idx[keep]], minlength=len(pop.cells))
        err = (total / count[:, None] - pop.target).abs().max().item()
    print('max |model cell mean - target| for an exact model:', err)
    assert err < 1e-10
    assert np.isfinite(pop.statistics['scale']).all() and (np.asarray(pop.statistics['scale']) > 0).all()
    print('median target R11, R22:', np.median(np.asarray(pop.statistics['target'])[:, [0, 3]], 0),
          ' median jackknife SE:', np.median(np.asarray(pop.statistics['scale']), 0))

    # 3. training gradient and validation score run and are finite; an exact model scores ~0 apart from sampling
    monkeypatch.setattr(g, 'GRID_CELLS_PER_STEP', min(12, len(pop.cells)))
    score = g.grid_backward(model, pop, 123, 1.0)
    print('train score', score, 'grad', model.p.grad.item())
    assert np.isfinite(score) and np.isfinite(model.p.grad.item())
    val = g.GridCutPopulation(context, data, edges, kw, scale=pop.scale_all, allowed=pop.cells)
    res = g.grid_evaluate(model, val, np.arange(n), draws=2, groups=2, batch_size=4096, seed=5)
    print('validation score on all pairs with the exact model:', res['loss'], 'cells', res['cells_scored'])
    assert abs(res['loss']) < 1e-12

