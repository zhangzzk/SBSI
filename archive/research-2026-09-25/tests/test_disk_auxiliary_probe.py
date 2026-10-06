import numpy as np
import pytest

from scripts.probe_disk_auxiliary import StudentUnionProposal, union_candidates


def test_union_preserves_original_and_auxiliary_rank_without_duplicates():
    got = union_candidates([[4, 2], [7, 8]], [[2, 1, 4, 0], [0, 8, 1, 3]], 4)
    np.testing.assert_array_equal(got, [[4, 2, 1, 0], [7, 8, 0, 1]])


def test_union_rejects_duplicate_or_insufficient_support():
    with pytest.raises(ValueError, match='unique'):
        union_candidates([[1, 1]], [[1, 2, 3]], 3)
    with pytest.raises(ValueError, match='insufficient'):
        union_candidates([[1, 2]], [[1, 2]], 3)
    with pytest.raises(ValueError, match='budget'):
        union_candidates([[1, 2]], [[1, 2]], 1)


@pytest.mark.parametrize('nu', [1., 3.])
def test_student_proxy_matches_independent_log_density_ranking(nu):
    from types import SimpleNamespace
    from scipy.stats import t
    values = np.array([[0., 1., 2., 3.], [1., -1., 4., 2.], [0., 0., 0., 0.]])
    dispersion = np.array([[1., 2., 3., 1.], [.5, 1., .5, 2.], [1., 1., 1., 1.]])
    weights = np.array([.8, .5, 0.])
    observed = np.array([[.2, .5, 5., 3.], [2., 1., 4., 8.]])
    base = SimpleNamespace(active_indices=np.arange(3),
        coordinates=SimpleNamespace(values=values, dispersion=dispersion),
        local_base_weights=weights, _observed_values=lambda x: x)
    proxy = StudentUnionProposal(base, device='cpu', degrees_of_freedom=nu)
    with np.errstate(divide='ignore'):
        expected = np.log(weights) + t.logpdf(observed[:, None, :], df=nu,
                                             loc=values, scale=dispersion).sum(-1)
    actual = proxy.auxiliary_score(observed).numpy()
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(actual[:, 1]-actual[:, 0], expected[:, 1]-expected[:, 0], atol=1e-12)
    with pytest.raises(ValueError, match='degrees'):
        StudentUnionProposal(base, device='cpu', degrees_of_freedom=0)
