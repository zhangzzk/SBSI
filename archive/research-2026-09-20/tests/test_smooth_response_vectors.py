import numpy as np

from scripts.summarize_smooth_response_vectors import response_vectors


def test_paired_vector_residual_with_unequal_leg_masses():
    sufficient = {}
    for source, offset in [('measured', [0.0, 0.0]), ('model', [0.01, -0.02])]:
        sufficient[source] = {}
        for leg, sign, mass in [('plus', 1, [1, 3]), ('minus', -1, [2, 1])]:
            mass = np.asarray(mass, float)[:, None]
            means = sign*.02*(np.asarray([[.7, .03], [.8, -.01]])+offset)
            sufficient[source][leg] = {'denominator': mass, 'numerator': mass[..., None]*means[:, None, :]}
    weights = np.asarray([[1, 1], [2, 0], [0, 2]])
    values = response_vectors(sufficient, weights, .02, 1)
    np.testing.assert_allclose(values['measured'][0], [(3.1/4+2.2/3)/2, (.00/4+.05/3)/2])
    np.testing.assert_allclose(values['measured_minus_model'], np.tile([-.01, .02], (3, 1)), atol=1e-15)
    np.testing.assert_allclose(values['residual_over_model_diagonal_percent'],
                               100*values['measured_minus_model']/values['model'][:, 0, None])
    swapped_axis = response_vectors(sufficient, weights, .02, 2)
    np.testing.assert_allclose(swapped_axis['residual_over_model_diagonal_percent'],
                               100*values['measured_minus_model']/values['model'][:, 1, None])
