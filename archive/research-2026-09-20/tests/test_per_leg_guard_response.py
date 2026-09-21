from itertools import product

import numpy as np
import pytest
import torch

from sbsi.flow_guard_response import fit_guard_response, independent_guard_score, make_guard_cuts, sampled_guard_response_backward
from scripts.per_leg_guard_response import PerLegGuardResponsePopulation, fit_per_leg_guard_response

CUTS = make_guard_cuts([3.], [25.8])
OPTIONS = {'radius_softness': .05, 'magnitude_softness': .05}


def oracle_data():
    zero, shear, gamma, u0, ug = [], [], [], [], []
    for axis, sign, zx, zy, w in product((0, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)):
        g = np.zeros(2)
        g[axis] = .05*sign
        e = .01*np.array([zx, zy])
        zero.append([*e, 4., 100.])
        shear.append([*(-e+.5*g), 4., 100.])
        gamma.append(g)
        u0.append(w == 1)
        ug.append(np.sign(e[axis])*sign == -w)
    return tuple(map(np.asarray, (zero, shear, gamma, u0, ug)))


class LinearFlow(torch.nn.Module):
    context_dim, target_dim = 2, 4

    def __init__(self, response):
        super().__init__()
        self.response = torch.nn.Parameter(torch.tensor(response, dtype=torch.float64))

    def forward_physical(self, z, context):
        e = self.response*context+.01*z[:, :2]
        photometry = torch.tensor([4., 100.], dtype=e.dtype, device=e.device).expand(len(e), -1)
        return torch.cat((e, photometry), -1), torch.zeros(len(e), dtype=e.dtype, device=e.device)


def population():
    zero, shear, gamma, u0, ug = oracle_data()
    return PerLegGuardResponsePopulation(torch.zeros_like(torch.tensor(gamma)), torch.tensor(gamma),
        zero, shear, gamma, u0, ug, CUTS, response_scale=1., **OPTIONS)


def test_all_usable_case_reduces_to_established_guard_estimator():
    zero, shear, gamma, _, _ = oracle_data()
    usable = np.ones(len(gamma), dtype=bool)
    old = fit_guard_response(zero, shear, gamma, CUTS, **OPTIONS)
    new = fit_per_leg_guard_response(zero, shear, gamma, usable, usable, CUTS, chunk_size=7, **OPTIONS)
    for field in ('target', 'baseline_mass', 'baseline_mean_shape', 'shape_normal'):
        np.testing.assert_allclose(new[field], old[field], atol=2e-15)


def test_population_consistency_and_exact_loss_gradient():
    pop = population()
    assert len(pop.pairs) == 32 and pop.statistics['usable_counts']['both'] == 8
    expected = np.broadcast_to([.5, 0., 0., .5], (len(CUTS), 4))
    np.testing.assert_allclose(pop.target.numpy(), expected, atol=2e-15)
    model = LinearFlow(.5)
    predicted = pop.contributions(model, torch.arange(32), 4, 41).mean(0)
    torch.testing.assert_close(predicted, pop.target, rtol=0, atol=2e-15)
    model = LinearFlow(.6)
    predicted = pop.contributions(model, torch.arange(32), 4, 41).mean(0)
    loss = .5*((predicted-pop.target)/pop.response_scale).square().mean()
    loss.backward()
    np.testing.assert_allclose(float(loss.detach()), .0025, atol=2e-15)
    np.testing.assert_allclose(float(model.response.grad), .05, atol=2e-15)


def test_independent_row_gradient_replay_matches_direct_autograd():
    pop, model = population(), LinearFlow(.6)
    count, chunk, seed = 17, 5, 461
    result = sampled_guard_response_backward(model, pop, count, 4, chunk_size=chunk, seed=seed, independent_pairs=True)
    replay_gradient = model.response.grad.clone()
    model.zero_grad()
    rng = np.random.default_rng(seed)
    indices = [torch.tensor(rng.integers(32, size=count)) for _ in range(2)]
    means = []
    for stream in (0, 1):
        pieces = [pop.contributions(model, indices[stream][start:start+chunk], 4, seed+100000000*stream+block)
                  for block, start in enumerate(range(0, count, chunk))]
        means.append(torch.cat(pieces).mean(0))
    direct = independent_guard_score(*means, pop.target, pop.response_scale)
    direct.backward()
    np.testing.assert_allclose(result['loss'], float(direct.detach()), atol=1e-15)
    torch.testing.assert_close(replay_gradient, model.response.grad, rtol=0, atol=1e-15)


def test_missing_measurements_ignored_only_when_own_leg_is_unusable():
    zero, shear, gamma, u0, ug = oracle_data()
    clean = fit_per_leg_guard_response(zero, shear, gamma, u0, ug, CUTS, **OPTIONS)
    zero[~u0], shear[~ug] = np.nan, np.nan
    missing = fit_per_leg_guard_response(zero, shear, gamma, u0, ug, CUTS, **OPTIONS)
    np.testing.assert_array_equal(clean['target'], missing['target'])
    zero[np.flatnonzero(u0)[0], 0] = np.nan
    with pytest.raises(ValueError, match='usable targets'):
        fit_per_leg_guard_response(zero, shear, gamma, u0, ug, CUTS, **OPTIONS)
