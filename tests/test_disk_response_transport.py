import torch

from sbsi.disk_response_transport import disk_response_transport


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


def test_joint_jacobian_when_velocity_depends_on_auxiliary_outputs():
    value = torch.tensor([.3, -.4, 2., 5.], dtype=torch.float64, requires_grad=True)

    def transform(y):
        velocity = torch.stack((.03*y[2]+.01*y[3], -.02*y[2]))
        shifted, logdet = disk_response_transport(y[:2], velocity)
        return torch.cat((shifted, y[2:])), logdet

    _, logdet = transform(value)
    jacobian = torch.autograd.functional.jacobian(lambda y: transform(y)[0], value)
    torch.testing.assert_close(logdet, torch.linalg.slogdet(jacobian).logabsdet, rtol=1e-13, atol=1e-13)
