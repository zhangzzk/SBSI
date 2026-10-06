from types import SimpleNamespace

import numpy as np
import torch

from scripts.guard_selection_mass import estimate_primary_mass


class DeterministicPhotometry:
    context_dim, target_dim = 1, 4

    def forward_physical(self, latent, context):
        values = torch.zeros((len(context), 4), dtype=context.dtype, device=context.device)
        values[:, 2] = context[:, 0]
        values[:, 3] = 100.
        return values, torch.zeros(len(context), dtype=context.dtype, device=context.device)


def test_primary_mass_uses_own_baseline_usability_and_full_parent_denominator():
    population = SimpleNamespace(context=torch.tensor([[4.], [1.], [4.], [2.], [5.], [6.]], dtype=torch.float64),
        pairs=torch.tensor([[0, 3], [1, 4], [2, 5]]),
        usable=torch.tensor([[True, False], [True, True], [False, True]]),
        cuts=({'name': 'primary', 'radius_min': 3., 'magnitude_max': 25.8},),
        radius_softness=.05, magnitude_softness=.05, zero_point=30.)
    result = estimate_primary_mass(DeterministicPhotometry(), population, seed=91, batch_size=2)
    mag_weight = 1/(1+np.exp(-(25.8-25.)/.05))
    expected_soft = mag_weight*(1/(1+np.exp(-20.))+1/(1+np.exp(40.)))/3
    np.testing.assert_allclose(result['estimate'], [expected_soft, 1/3], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result['integration_se'], [0., 0.])


def test_exact_radius_boundary_is_excluded_from_hard_cut():
    population = SimpleNamespace(context=torch.tensor([[3.]], dtype=torch.float64), pairs=torch.tensor([[0, 0]]),
        usable=torch.tensor([[True, True]]), cuts=({'name': 'primary', 'radius_min': 3., 'magnitude_max': 25.8},),
        radius_softness=.05, magnitude_softness=.05, zero_point=30.)
    result = estimate_primary_mass(DeterministicPhotometry(), population, seed=91)
    assert result['estimate'][1] == 0
    assert 0 < result['estimate'][0] < 1
