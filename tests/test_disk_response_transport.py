import numpy as np
import torch

from sbsi.disk_response_transport import (
    directional_response_gain, disk_response_transport, infinitesimal_response_matrix,
    transported_physical_log_prob,
)


def test_exact_inverse_and_density_jacobian():
    shape = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
    velocity = torch.tensor([.12, .07], dtype=torch.float64)
    shifted, logdet = disk_response_transport(shape, velocity)
    restored, inverse_logdet = disk_response_transport(shifted, velocity, inverse=True)
    torch.testing.assert_close(restored, shape, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(logdet+inverse_logdet, torch.zeros_like(logdet), rtol=0, atol=1e-13)
    jacobian = torch.autograd.functional.jacobian(lambda e: disk_response_transport(e, velocity)[0], shape)
    torch.testing.assert_close(logdet, torch.linalg.slogdet(jacobian).logabsdet, rtol=1e-13, atol=1e-13)


def test_transport_stays_inside_disk_without_clipping():
    generator = torch.Generator().manual_seed(7301)
    shapes = torch.randn((1024, 2), generator=generator, dtype=torch.float64)
    shapes = .999*shapes/(1+torch.linalg.vector_norm(shapes, dim=-1, keepdim=True))
    velocity = torch.randn((1024, 2), generator=generator, dtype=torch.float64)*.8
    transformed, logdet = disk_response_transport(shapes, velocity)
    assert bool((transformed.square().sum(-1) < 1).all())
    assert bool(torch.isfinite(logdet).all())
    identity, zero_logdet = disk_response_transport(shapes, torch.zeros_like(velocity))
    torch.testing.assert_close(identity, shapes, rtol=0, atol=0)
    torch.testing.assert_close(zero_logdet, torch.zeros_like(zero_logdet), rtol=0, atol=0)


def test_physical_response_gain_matches_autograd_and_is_positive():
    shape = torch.tensor([.6, .7], dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda v: disk_response_transport(shape, v)[0], zero)
    matrix = infinitesimal_response_matrix(shape)
    torch.testing.assert_close(jacobian, matrix, rtol=1e-13, atol=1e-13)
    direction = torch.tensor([.6, .8], dtype=torch.float64)
    gain = directional_response_gain(shape, direction)
    assert 1-shape.square().sum() <= gain <= 1+shape.square().sum()
    assert gain > 0
    torch.testing.assert_close(gain, direction @ jacobian @ direction)


def test_joint_jacobian_when_velocity_depends_on_auxiliary_outputs():
    value = torch.tensor([.3, -.4, 2., 5.], dtype=torch.float64, requires_grad=True)

    def transform(y):
        velocity = torch.stack((.03*y[2]+.01*y[3], -.02*y[2]))
        shifted, logdet = disk_response_transport(y[:2], velocity)
        return torch.cat((shifted, y[2:])), logdet

    _, logdet = transform(value)
    jacobian = torch.autograd.functional.jacobian(lambda y: transform(y)[0], value)
    torch.testing.assert_close(logdet, torch.linalg.slogdet(jacobian).logabsdet, rtol=1e-13, atol=1e-13)


def test_transported_density_normalizes_on_disk_and_preserves_auxiliary_marginal():
    # Integrate over the observed disk, not latent samples: this catches the
    # sign or omission of the inverse Jacobian. Auxiliary density is Exp(1)^2.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    radius = torch.tensor(np.sqrt((nodes+1)/2), dtype=torch.float64)
    angle = torch.arange(128, dtype=torch.float64)*(2*np.pi/128)
    shape = torch.stack((radius[:, None]*angle.cos(), radius[:, None]*angle.sin()), -1)
    auxiliary = torch.tensor([2., 3.], dtype=torch.float64).expand(32, 128, 2)
    physical = torch.cat((shape, auxiliary), -1)
    velocity = torch.stack((.1*auxiliary[..., 0], -.1*auxiliary[..., 1]), -1)

    def base_log_prob(values):
        return -np.log(np.pi)-values[..., 2:].sum(-1)

    density = transported_physical_log_prob(physical, velocity, base_log_prob).exp()
    integral = (density.mean(-1)*torch.tensor(weights)*(np.pi/2)).sum()
    torch.testing.assert_close(integral, torch.tensor(np.exp(-5)), rtol=1e-12, atol=1e-14)
    identity = transported_physical_log_prob(physical, torch.zeros_like(velocity), base_log_prob)
    torch.testing.assert_close(identity, base_log_prob(physical), rtol=0, atol=0)


def test_transported_density_rejects_invalid_physical_outputs_without_clipping():
    physical = torch.tensor([[1., 0., 2., 3.], [.1, .2, -1., 2.],
                             [float('nan'), 0., 1., 2.], [.1, .2, 2., 3.]], dtype=torch.float64)
    result = transported_physical_log_prob(
        physical, torch.tensor([.1, .2]), lambda values: -values.square().sum(-1))
    assert bool(torch.isneginf(result[:3]).all())
    assert bool(torch.isfinite(result[3]))
