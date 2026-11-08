import numpy as np
import pytest
import torch
import pandas as pd

from scripts.audit_v36_domain_response import group_masks, moments, paired_draws, response, selected
from scripts.audit_v36_domain_response import close_crowding_context


def test_close_crowding_uses_all_neighbours_and_removes_self():
    scene = pd.DataFrame(dict(RA=[0., 1/3600., 20/3600.], DEC=[0., 0., 0.], r=[25., 25., 25.]))
    result = close_crowding_context(scene)
    expected = np.log1p(np.exp(-.5/np.square([.5, 1., 2.])))
    np.testing.assert_allclose(result[:2], np.tile(expected, (2, 1)), rtol=1e-6)
    np.testing.assert_array_equal(result[2], 0.)


def test_nested_domain_partition_rejects_missing_retained_rows():
    masks = group_masks(np.arange(4), [25., 26., 24., 27.], [0])
    np.testing.assert_array_equal(masks['all_bright'], [True, False, True, False])
    np.testing.assert_array_equal(masks['retained'].astype(int)+masks['added_bright']+masks['faint'], 1)
    with pytest.raises(ValueError, match='absent'):
        group_masks(np.arange(4), [25., 26., 24., 27.], [7])
    with pytest.raises(ValueError, match='magnitude support'):
        group_masks(np.arange(4), [25., 26., 24., 27.], [1])


def test_independent_leg_denominators_and_invalid_observed_rows():
    # Different selected populations: averaging object responses is incorrect.
    plus = np.array([[.2, .1, 4., 100.], [.4, -.1, 4., 100.], [np.nan]*4])
    minus = np.array([[.1, .1, 4., 100.], [.3, -.1, 2., 100.], [np.nan]*4])
    mask = {'all': np.ones(3, bool)}
    p = moments(plus, np.array([1., 1., 0.]), mask)['all']
    m = moments(minus, np.array([1., 1., 0.]), mask)['all']
    np.testing.assert_allclose(response(p, m), [5., -2.5])
    np.testing.assert_array_equal(selected(np.array([[0., 0., 3., 100.]])), [False])


def test_weighted_draw_sums_preserve_population_mass():
    x = np.array([[[.2, 0., 4., 100.], [.6, 0., 2., 100.]],
                  [[.4, 0., 4., 100.], [.8, 0., 4., 100.]]])
    sums = moments(x, np.array([.4, .8]), {'all': np.ones(2, bool)})['all']
    np.testing.assert_allclose(sums, [1., .52, 0.])


def test_antithetic_draws_share_random_numbers_across_shear_legs():
    class Model:
        target_dim = 4
        context_dim = 4

        def forward_physical(self, z, context):
            return z+context, None

    c = torch.zeros(3, 4, dtype=torch.float64)
    a, b = paired_draws(Model(), [c, c+2], 8, 7301)
    torch.testing.assert_close(b-a, torch.full_like(a, 2.))
    torch.testing.assert_close(a[:, :4], -a[:, 4:])


def test_underflow_excluded_by_measured_cut_keeps_original_denominator():
    values = np.array([[[.4, .2, 4., 100.], [np.nan, np.nan, 0., 0.]]])
    result = moments(values, np.ones(1), {'all': np.ones(1, bool)})['all']
    np.testing.assert_allclose(result, [.5, .2, .1])
