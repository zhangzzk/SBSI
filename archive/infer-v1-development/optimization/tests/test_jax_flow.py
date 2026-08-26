"""Archived tests for the unadopted JAX flow kernel."""

import numpy as np
import pytest
import torch

from sbsi.jax_flow import build_jax_log_prob_from_torch, freeze_mean_affine_flow
from test_catalogue_null import _torch_two_shape_likelihood


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_jax_mean_affine_log_prob_matches_torch_with_view_axis():
    likelihood = _torch_two_shape_likelihood(sigma=0.12)
    model = likelihood.flow_model.model
    frozen = freeze_mean_affine_flow(model)
    evaluate = build_jax_log_prob_from_torch(model)

    rng = np.random.default_rng(1201)
    target = rng.normal(size=(3, 17, frozen.target_dim)).astype(np.float32)
    context = rng.normal(size=(3, 17, frozen.context_dim)).astype(np.float32)
    with torch.no_grad():
        expected = model.log_prob(
            torch.from_numpy(target.reshape(-1, frozen.target_dim)),
            torch.from_numpy(context.reshape(-1, frozen.context_dim)),
        ).reshape(3, 17)
    actual = np.asarray(evaluate(jnp.asarray(target), jnp.asarray(context)))

    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_jax_converter_rejects_non_flow_module():
    with pytest.raises(TypeError, match="ConditionalMeanFlow"):
        freeze_mean_affine_flow(torch.nn.Linear(2, 2))
