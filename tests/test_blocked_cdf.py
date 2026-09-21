import numpy as np
import pytest
import torch

from sbsi.catalogue_sampling import _blocked_cumulative_sum


@pytest.mark.parametrize("rows,width,block", [(1, 1, 16), (3, 16, 4), (2, 19, 4), (3, 8193, 1024)])
def test_tiled_probability_scan_matches_full_scan(rows, width, block):
    rng = np.random.default_rng(82)
    values = rng.lognormal(0, 3, (rows, width))
    values /= values.sum(axis=1, keepdims=True)
    tensor = torch.from_numpy(values)
    output = _blocked_cumulative_sum(tensor, block)
    np.testing.assert_allclose(output, np.cumsum(values, axis=1), rtol=3e-14, atol=1e-15)
    np.testing.assert_array_equal(tensor, values)
    assert bool((torch.diff(output, dim=1) >= 0).all())
    uniforms = torch.from_numpy(rng.random((rows, 1000)))
    reference = torch.cumsum(tensor, dim=1)
    output[:, -1] = reference[:, -1] = 1.
    torch.testing.assert_close(torch.searchsorted(output, uniforms, right=True),
                               torch.searchsorted(reference, uniforms, right=True), rtol=0, atol=0)
