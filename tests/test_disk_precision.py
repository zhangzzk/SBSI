from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
import torch

from sbsi.disk_precision import (physical_log_prob_with_gap, radial_log_gap,
                                transport_with_log_gap, validate_shape_encoding)
from sbsi.flow_disk import radial_tanh
from sbsi.flow_physical_disk import ConditionalPhysicalDiskFlow


def model():
    torch.manual_seed(912)
    return ConditionalPhysicalDiskFlow(
        target_dim=4, context_dim=2, hidden_dim=8, n_layers=1, n_flows=2,
        disk_map='radial_tanh', disk_shape_means=[.1, -.2],
        disk_shape_scales=[.3, .4], disk_coordinate_means=[.1, .2],
        disk_coordinate_scales=[.7, 1.2], physical_map='softplus',
        physical_means=[2, 100], physical_scales=[1, 100],
        physical_units=[1, 100], physical_coordinate_means=[0, 0],
        physical_coordinate_scales=[1, 1]).eval()


@pytest.mark.parametrize('radius', [0., 1.e-9, .5, 8., 25., 80.])
def test_radial_gap_and_transport_against_120_digit_arithmetic(radius):
    u = torch.tensor([[radius*.6, radius*.8]], dtype=torch.float64)
    shape, _ = radial_tanh(u)
    gap = radial_log_gap(u)
    velocity = torch.tensor([[.02, -.01]], dtype=torch.float64)
    moved, moved_gap, ld = transport_with_log_gap(shape, gap, velocity)
    with localcontext() as ctx:
        ctx.prec = 120
        x, y = map(lambda a: Decimal.from_float(float(a)), u[0])
        r = (x*x+y*y).sqrt()
        ex = (2*r).exp()
        t = (ex-1)/(ex+1)
        e1, e2 = (x*t/r, y*t/r) if r else (Decimal(0), Decimal(0))
        vx, vy = map(lambda a: Decimal.from_float(float(a)), velocity[0])
        vr = (vx*vx+vy*vy).sqrt()
        ev = (2*vr).exp()
        ratio = (ev-1)/(ev+1)/vr
        a, b = vx*ratio, vy*ratio
        dr, di = 1+a*e1+b*e2, a*e2-b*e1
        den = dr*dr+di*di
        transformed = [((e1+a)*dr+(e2+b)*di)/den, ((e2+b)*dr-(e1+a)*di)/den]
        exact_gap = (1-e1*e1-e2*e2).ln()
        exact_moved_gap = (1-transformed[0]**2-transformed[1]**2).ln()
        exact_ld = 2*((1-a*a-b*b).ln()-den.ln())
    np.testing.assert_allclose(gap.numpy(), [float(exact_gap)], rtol=1.e-13, atol=1.e-30)
    np.testing.assert_allclose(moved.numpy()[0], list(map(float, transformed)), rtol=1.e-13, atol=1.e-15)
    np.testing.assert_allclose(moved_gap.numpy(), [float(exact_moved_gap)], rtol=1.e-13, atol=1.e-15)
    np.testing.assert_allclose(ld.numpy(), [float(exact_ld)], rtol=1.e-13, atol=1.e-15)
    restored, restored_gap, inverse_ld = transport_with_log_gap(moved, moved_gap, velocity, inverse=True)
    torch.testing.assert_close(restored, shape, rtol=1.e-13, atol=1.e-15)
    torch.testing.assert_close(restored_gap, gap, rtol=1.e-13, atol=1.e-15)
    torch.testing.assert_close(ld+inverse_ld, torch.zeros_like(ld), rtol=0, atol=1.e-14)


def test_sampling_metadata_preserves_every_draw_and_random_stream():
    flow = model()
    context = torch.randn(20, 2)
    for qmc in (False, True):
        torch.manual_seed(734)
        before = flow.sample_physical(context, n_samples=3, qmc=qmc)
        state = torch.random.get_rng_state()
        torch.manual_seed(734)
        after, gap = flow.sample_physical(context, n_samples=3, qmc=qmc, return_shape_log_gap=True)
        assert torch.equal(before, after)
        assert torch.equal(state, torch.random.get_rng_state())
        validate_shape_encoding(after[..., :2].detach().numpy(), gap.detach().numpy())


def test_density_matches_forward_jacobian_including_rounded_boundary():
    flow = model()
    context = torch.zeros(4, 2)
    z = torch.tensor([[0., 0., 0., 0.], [.3, -.6, .5, -.2], [30., 40., 1., 2.], [80., -60., 1., 2.]])
    physical, forward_ld, gap = flow.forward_physical(z, context, return_shape_log_gap=True)
    expected = -.5*(z.double().square()+math.log(2*math.pi)).sum(-1)-forward_ld
    actual = physical_log_prob_with_gap(flow, physical, gap, context)
    torch.testing.assert_close(actual, expected, rtol=2.e-5, atol=2.e-5)
    assert (physical[:, :2].square().sum(-1)[2:] >= 1-1.e-15).all()
    assert torch.isfinite(actual).all()
    means = torch.cat((flow.disk_shape_means, flow.physical_means))
    scales = torch.cat((flow.disk_shape_scales, flow.physical_scales))
    old = flow.log_prob((physical[:2]-means)/scales, context[:2])-scales.log().sum()
    torch.testing.assert_close(actual[:2], old, rtol=2.e-6, atol=2.e-6)


def test_inconsistent_precision_metadata_is_rejected():
    with pytest.raises(ValueError, match='encoding'):
        validate_shape_encoding([[1.1, 0]], [-50.])
    with pytest.raises(ValueError, match='encoding'):
        validate_shape_encoding([[.5, 0]], [-50.])
    with pytest.raises(ValueError, match='encoding'):
        validate_shape_encoding([[1., 0]], [float('-inf')])
    validate_shape_encoding([[1., 0]], [-1000.])
