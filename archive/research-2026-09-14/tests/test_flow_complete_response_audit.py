from types import SimpleNamespace

import numpy as np
import torch

from _script_loader import load_script_module


AUDIT = load_script_module("audit_flow_derivative_losses.py")


class _ToySampleModel(torch.nn.Module):
    def sample(self, context, n_samples=1, qmc=False):
        noise = torch.randn(
            len(context), n_samples, 2, dtype=context.dtype, device=context.device
        )
        return noise + context[:, None, :2]


def test_sample_pair_delta_uses_common_random_numbers():
    bundle = SimpleNamespace(
        model=_ToySampleModel(),
        device=torch.device("cpu"),
        target_transform=SimpleNamespace(scales=np.array([2.0, 3.0])),
    )
    first = np.zeros((5, 2), dtype=np.float32)
    second = np.tile(np.array([0.1, -0.2], dtype=np.float32), (5, 1))

    delta = AUDIT._sample_pair_delta(
        bundle, first, second, n_samples=8, batch_size=3, seed=17
    )

    np.testing.assert_allclose(delta, np.tile([0.2, -0.6], (5, 1)), atol=1.0e-6)


def test_response_matrix_outputs_recovers_known_linear_map():
    applied = np.array(
        [[0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05]]
    )
    response = np.array([[0.8, 0.1], [-0.2, 0.7], [0.03, -0.04]])
    delta = applied @ response.T

    recovered = AUDIT._response_matrix_outputs(delta, applied)

    np.testing.assert_allclose(recovered, response, atol=1.0e-12)
