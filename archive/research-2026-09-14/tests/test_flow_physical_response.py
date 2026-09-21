import numpy as np
import pytest
import torch
from sbsi.flow_physical_response import (physical_outputs, baseline_frame_gamma,
                                         marginal_physical_response_statistics)

def test_physical_before_mean_and_gradient():
    x=torch.tensor([[.1,.2,30.,0.],[.1,.2,25.,np.log(3)]],dtype=torch.float64,requires_grad=True)
    y=physical_outputs(x);np.testing.assert_allclose(y.detach()[:,2:],[[1,1],[3,100]])
    np.testing.assert_allclose(y.mean(0).detach()[2:],[2,50.5])
    assert not torch.allclose(y.mean(0)[2:],physical_outputs(x.mean(0))[2:])
    y[:,2:].sum().backward()
    np.testing.assert_allclose(x.grad[:,3],[1,3]);np.testing.assert_allclose(x.grad[:,2],-.4*np.log(10)*np.array([1,100]))

def test_no_repair():
    with pytest.raises(ValueError,match='nonfinite'):physical_outputs(np.array([[0,0,-1000,1000.]]))

def test_fixed_frame_and_circular():
    g=np.array([[.02,.01],[.02,.01],[.02,.01]])
    e=np.array([[.3,0],[0,.3],[0,0]])
    np.testing.assert_allclose(baseline_frame_gamma(g,e),[[.02,.01],[.01,-.02],[.02,.01]])

def test_joint_covariance_units_and_shape_block():
    from sbsi.flow_physical_response import fit_physical_response
    from sbsi.flow_grid_response import fit_grid_response
    rng=np.random.default_rng(14);n=800
    g=rng.normal(size=(n,2))*.03;d=rng.normal(size=(n,4))
    cases=np.repeat(np.arange(40),20);bins=np.zeros(n,dtype=int)
    a=fit_physical_response(d,g,bins,cases,1)
    shape=fit_grid_response(d[:,:2],g,bins,cases,1)
    np.testing.assert_allclose(a['target'][:,:4],shape['target'])
    np.testing.assert_allclose(a['covariance'][:,:4,:4],shape['covariance'])
    units=np.array([1,1,1e-2,1e5]);b=fit_physical_response(d*units,g,bins,cases,1)
    np.testing.assert_allclose(b['target'],a['target']*np.repeat(units,2),rtol=1e-11)
    q=lambda x:np.einsum('bi,bij,bj->b',x['target'],x['precision'],x['target'])
    np.testing.assert_allclose(q(a),q(b),rtol=1e-10)


def test_shape_radius_statistics_use_marginal_covariance_not_conditional_precision():
    rng = np.random.default_rng(91)
    mixing = rng.normal(size=(8, 8))
    covariance = mixing @ mixing.T + np.eye(8)
    covariance[4, 6] = covariance[6, 4] = .85*np.sqrt(covariance[4, 4]*covariance[6, 6])
    # Restore positive definiteness after imposing a strong radius--flux coupling.
    covariance += np.eye(8)*max(0., 1e-6-np.linalg.eigvalsh(covariance).min())
    statistics = dict(target=rng.normal(size=(2, 8)),
                      covariance=np.tile(covariance, (2, 1, 1)),
                      precision=np.tile(np.linalg.inv(covariance), (2, 1, 1)),
                      jackknife=rng.normal(size=(9, 2, 8)),
                      counts=np.array([10, 11]),
                      shape_design=rng.normal(size=(21, 2)),
                      scalar_design=rng.normal(size=(21, 2)))
    result = marginal_physical_response_statistics(
        statistics, component_weights=[1, 1, 1, 1, .2, .2])
    np.testing.assert_array_equal(result['response_component_indices'], np.arange(6))
    np.testing.assert_allclose(result['target'], statistics['target'][:, :6])
    np.testing.assert_allclose(result['jackknife'], statistics['jackknife'][:, :, :6])
    np.testing.assert_allclose(result['covariance'], statistics['covariance'][:, :6, :6])
    for cell in range(2):
        np.testing.assert_allclose(result['precision'][cell]@result['covariance'][cell],
                                   np.eye(6), atol=1e-11)
    assert not np.allclose(result['precision'], statistics['precision'][:, :6, :6])


def test_shape_only_precision_is_independent_of_scalar_cross_covariance():
    rng = np.random.default_rng(79)
    mixing = rng.normal(size=(8, 8))
    covariance = mixing@mixing.T+np.eye(8)
    statistics = dict(target=np.ones((2, 8)), covariance=np.tile(covariance, (2, 1, 1)),
                      jackknife=rng.normal(size=(12, 2, 8)))
    result = marginal_physical_response_statistics(statistics, ('shape',))
    np.testing.assert_array_equal(result['response_component_indices'], np.arange(4))
    np.testing.assert_allclose(result['precision'], np.tile(np.linalg.inv(covariance[:4, :4]), (2, 1, 1)))
    assert not np.allclose(result['precision'][0], np.linalg.inv(covariance)[:4, :4])
    statistics['covariance'][:, 4:, :] = 0.
    statistics['covariance'][:, :, 4:] = 0.
    statistics['target'][:, 4:] = 1e100
    other = marginal_physical_response_statistics(statistics, ('shape',))
    np.testing.assert_array_equal(other['precision'], result['precision'])
    np.testing.assert_array_equal(other['target'], result['target'])
