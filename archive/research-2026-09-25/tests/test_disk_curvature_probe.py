import numpy as np

from scripts.probe_disk_curvature import exact_observation, stencil_moments


def test_stencil_reproduces_quadratic_score_and_full_information():
    center = np.array([.014, .0003])
    h = .001
    gradient = np.array([2., -3.])
    info = np.array([[5., 1.5], [1.5, 7.]])
    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]
    points = [center+h*np.array(x) for x in offsets]
    values = [4 + gradient @ p - .5*p @ info @ p for p in points]
    score, information = stencil_moments(values, h)
    np.testing.assert_allclose(score, gradient-info @ center, atol=1e-10)
    np.testing.assert_allclose(information, info, atol=2e-9)


def test_exact_sum_keeps_global_masses_across_atom_blocks():
    from types import SimpleNamespace
    import torch
    weights = np.array([.1, .6, .3])
    h = .001
    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]
    points = h*np.array(offsets)

    def terms(observed, g1, g2, *, atom_indices, **kwargs):
        return torch.as_tensor((np.log(weights[atom_indices])+2*g1-3*g2-.5*(5*g1*g1+7*g2*g2))[None, :])

    likelihood = SimpleNamespace(disk_response=SimpleNamespace(n_atoms=3),
        log_importance_weights_tensor=terms,
        population_normalization=SimpleNamespace(log_mass=lambda *p: 0.))
    report = exact_observation(likelihood, None, points, h, atom_block=2)
    np.testing.assert_allclose(report['score'], [2., -3.], atol=1e-11)
    np.testing.assert_allclose(report['information'], np.diag([5., 7.]), atol=1e-9)
    np.testing.assert_allclose(report['log_numerator'][0], 0., atol=1e-15)
    np.testing.assert_allclose(report['top_posterior_weights'][0], [.6, .3, .1])
    assert report['top_atoms'][0] == [1, 2, 0]
