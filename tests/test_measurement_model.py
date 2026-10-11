import math

import torch

from sbsi.flow_physical_disk import ConditionalPhysicalDiskFlow
from sbsi.measurement_model import ConditionalAffineFlow


def test_conditional_affine_flow_inverse_undoes_forward():
    torch.manual_seed(5)
    model = ConditionalAffineFlow(target_dim=4, context_dim=3, hidden_dim=8, n_layers=1, n_flows=2)
    for layer in model.layers:  # move off the identity initialization
        torch.nn.init.normal_(layer.net[-1].weight, std=.3)
    x, context = torch.randn(6, 4), torch.randn(6, 3)
    z, inverse_logdet = model.inverse(x, context)
    recovered, forward_logdet = model.forward(z, context)
    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(inverse_logdet + forward_logdet, torch.zeros_like(inverse_logdet))


def test_physical_flow_draws_and_density_agree():
    torch.manual_seed(3)
    flow = ConditionalPhysicalDiskFlow(
        target_dim=4, context_dim=2, hidden_dim=8, n_layers=1, n_flows=2,
        disk_map="radial_tanh", disk_shape_means=[.1, -.2], disk_shape_scales=[.3, .4],
        disk_coordinate_means=[.1, .2], disk_coordinate_scales=[.7, 1.2], physical_map="softplus",
        physical_means=[2, 100], physical_scales=[1, 100], physical_units=[1, 100],
        physical_coordinate_means=[0, 0], physical_coordinate_scales=[1, 1]).eval()
    for layer in flow.base.layers:
        torch.nn.init.normal_(layer.net[-1].weight, std=.3)
    context = torch.randn(5, 2)
    draws = flow.sample_physical(context, n_samples=3)
    assert draws.shape == (5, 3, 4)
    assert (draws[..., :2].square().sum(-1) < 1).all() and (draws[..., 2:] > 0).all()

    z = torch.randn(64, 4)
    context = torch.randn(64, 2)
    with torch.no_grad():
        physical, logdet = flow.forward_physical(z, context)
        means = torch.cat((flow.disk_shape_means, flow.physical_means))
        scales = torch.cat((flow.disk_shape_scales, flow.physical_scales))
        log_prob = flow.log_prob((physical - means) / scales, context)
    # Change of variables: the standardized density equals the base density over the Jacobian.
    log_base = -.5 * (z.double().square() + math.log(2 * math.pi)).sum(-1)
    torch.testing.assert_close(log_prob, log_base - logdet + scales.log().sum(), rtol=0, atol=1e-4)
