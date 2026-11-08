"""Periodic product rule, actual Möbius convention and full-population accounting."""
import numpy as np
import pytest
from scipy.integrate import quad_vec

from sbsi.analytic_phase_control import (EPSILON,analytic_phase_control,
    phase_coefficient,summarize_score_comparison)
from sbsi.shear_map import apply_shear_to_ellipticity,shear_jacobian_at_zero


@pytest.mark.parametrize('radius',[.001,.1,.7])
@pytest.mark.parametrize('mu',[0.,2*np.pi-1e-5])
def test_product_rule_integrates_to_zero_for_nonuniform_periodic_targets(radius,mu):
    kappa,harmonic=3.,.4
    def integrand(psi):
        e=radius*np.array([np.cos(psi),np.sin(psi)])
        t=-kappa*np.sin(psi-mu)+2*harmonic*np.cos(2*psi)
        kernel=np.exp(kappa*np.cos(psi-mu)+harmonic*np.sin(2*psi)-kappa)
        return np.r_[analytic_phase_control(e,np.asarray(t))*kernel,kernel]
    value,error=quad_vec(integrand,0.,2*np.pi,epsabs=1e-11,epsrel=1e-11)
    np.testing.assert_allclose(value[:2]/value[2],0.,atol=2e-11,rtol=0)
    assert error<1e-9


def test_omitting_coefficient_derivative_is_not_a_zero_mean_control():
    radius,kappa=.4,2.
    def integrand(psi):
        e=radius*np.array([np.cos(psi),np.sin(psi)])
        a,prime=phase_coefficient(e);t=-kappa*np.sin(psi)
        kernel=np.exp(kappa*np.cos(psi))
        return np.r_[a*t*kernel,prime*kernel,kernel]
    value,_=quad_vec(integrand,0.,2*np.pi,epsabs=1e-11)
    missing=value[:2]/value[4];prime=value[2:4]/value[4]
    assert missing[0]>.5
    np.testing.assert_allclose(missing+prime,0.,atol=1e-11)


def test_origin_finite_near_origin_continuous_and_phase_wrap():
    assert EPSILON==.1
    e=np.array([[0.,0.],[1e-15,0.],[0.,-1e-15]])
    c=analytic_phase_control(e,np.array([100.,1.,-1.]))
    np.testing.assert_array_equal(c[0],[0.,0.])
    assert np.isfinite(c).all() and np.max(np.abs(c[1:]))<1e-12
    phase=np.array([0.,np.pi,2*np.pi-1e-10]);r=.3
    a=r*np.column_stack((np.cos(phase),np.sin(phase)))
    b=r*np.column_stack((np.cos(phase+2*np.pi),np.sin(phase+2*np.pi)))
    np.testing.assert_allclose(analytic_phase_control(a,np.sin(phase)),
        analytic_phase_control(b,np.sin(phase+2*np.pi)),atol=1e-13,rtol=0)


def test_coefficient_phase_derivative_and_component_signs():
    phase=np.array([.2,1.1,4.9]);r=np.array([.05,.3,.8]);h=1e-6
    e=r[:,None]*np.column_stack((np.cos(phase),np.sin(phase)))
    upper=r[:,None]*np.column_stack((np.cos(phase+h),np.sin(phase+h)))
    lower=r[:,None]*np.column_stack((np.cos(phase-h),np.sin(phase-h)))
    a,prime=phase_coefficient(e)
    np.testing.assert_allclose((phase_coefficient(upper)[0]-phase_coefficient(lower)[0])/(2*h),prime,atol=1e-9,rtol=1e-9)
    t=np.array([2.,-3.,.4]);f=(1+r*r)/(r*r+.01)
    expected=np.column_stack((-f*(e[:,1]*t+e[:,0]),f*(e[:,0]*t-e[:,1])))
    np.testing.assert_allclose(analytic_phase_control(e,t),expected,atol=1e-14,rtol=0)


def test_g0_intrinsic_mobius_phase_tangent_and_radial_score_decomposition():
    phase=np.array([0.,.4,2.1,5.9]);r=np.array([.02,.1,.5,.9])
    e=r[:,None]*np.column_stack((np.cos(phase),np.sin(phase)))
    jac=shear_jacobian_at_zero(e[:,0],e[:,1])
    angular=np.column_stack((-e[:,1],e[:,0]))/r[:,None]**2
    unregularized=np.einsum('ni,nij->nj',angular,jac)
    expected=(1+r*r)[:,None]*np.column_stack((-e[:,1],e[:,0]))/r[:,None]**2
    np.testing.assert_allclose(unregularized,expected,rtol=1e-14,atol=1e-14)
    h=1e-7
    for j in range(2):
        g=np.zeros(2);g[j]=h
        plus=np.stack(apply_shear_to_ellipticity(e[:,0],e[:,1],*g),axis=-1)
        minus=np.stack(apply_shear_to_ellipticity(e[:,0],e[:,1],*(-g)),axis=-1)
        np.testing.assert_allclose((plus-minus)/(2*h),jac[:,:,j],atol=1e-9,rtol=1e-8)
        zplus=plus[:,0]+1j*plus[:,1];zminus=minus[:,0]+1j*minus[:,1]
        np.testing.assert_allclose(np.angle(zplus*np.conj(zminus))/(2*h),expected[:,j],atol=2e-8,rtol=1e-8)
    gradient=np.array([[1.,2.],[-.3,1.],[2.,-.5],[.4,.7]])
    t=-e[:,1]*gradient[:,0]+e[:,0]*gradient[:,1]
    radial_derivative=(gradient*e).sum(1)/r
    radial_shear=(1-r*r)[:,None]*e/r[:,None]
    direct=np.einsum('ni,nij->nj',gradient,jac)
    np.testing.assert_allclose(direct,radial_shear*radial_derivative[:,None]+expected*t[:,None],atol=1e-13)
    a,prime=phase_coefficient(e)
    np.testing.assert_allclose(a,expected*(r*r/(r*r+.01))[:,None],atol=1e-13)
    np.testing.assert_allclose(direct-analytic_phase_control(e,t),
        radial_shear*radial_derivative[:,None]+(expected-a)*t[:,None]-prime,atol=1e-13)


def test_full_N_and_Z_weighting_and_explicit_raw_measure_qualification():
    logz=np.log([[1.],[3.]])
    raw=[np.array([[[1.,2.],[3.,4.]]]),np.array([[[5.,6.],[7.,8.],[9.,10.],[11.,12.]]])]
    controls=[v*.2 for v in raw];constant=[v-.5 for v in raw]
    phase=[np.ones(v.shape[:2]) for v in raw]
    delta=[np.log(np.array([[1.,3.]])),np.log(np.array([[2.,2.,2.,2.]]))]
    row=summarize_score_comparison(logz,raw,constant,controls,phase,delta)[0]
    a=row['raw_float32_sampled'];b=row['precision_reweighted_float64']
    np.testing.assert_allclose(a['mean'][:2],.25*np.array([2.,3.])+.75*np.array([8.,9.]))
    np.testing.assert_allclose(a['mean'][4:6],.8*a['mean'][:2])
    # Both full-run mean(exp(delta)) equal two, despite different N.
    np.testing.assert_allclose(b['weights'],[.25,.75])
    np.testing.assert_allclose(b['mean'][:2],.25*np.array([2.5,3.5])+.75*np.array([8.,9.]))
    assert a['analytic_control_known_expectation'] is None
    assert a['phase_control_known_expectation'] is None
    assert not a['zero_mean_identity_established_for_this_numerical_measure']
    assert b['analytic_control_known_expectation']==[0.,0.]


def test_invalid_shapes_domain_and_intermediate_beta_rejected():
    with pytest.raises(ValueError,match='beta'):analytic_phase_control([.1,.2],np.array(1.),beta=.5)
    with pytest.raises(ValueError,match='unit disk'):analytic_phase_control([1.,0.],np.array(1.))
    with pytest.raises(ValueError,match='aligned'):analytic_phase_control([[.1,.2]],np.array([1.,2.]))
    with pytest.raises(ValueError,match='finite'):phase_coefficient([np.nan,0.])
