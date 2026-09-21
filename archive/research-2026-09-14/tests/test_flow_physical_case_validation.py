import numpy as np
import pytest
from sbsi.flow_physical_case_validation import paired_physical_case_statistics,fixed_cell_average_statistics


def fixture():
    rng=np.random.default_rng(25)
    a=rng.normal(size=(10,3,2,2,2))
    normal=a@a.swapaxes(-1,-2)+np.eye(2)*2
    measured=rng.normal(size=normal.shape)
    groups=measured[None]+rng.normal(scale=.1,size=(4,*normal.shape))
    return normal,measured,groups


def test_joint_deleted_case_fits_and_cross_block_covariance():
    n,y,g=fixture();s=paired_physical_case_statistics(n,y,g)
    expected=[]
    for c in range(len(n)):
        keep=np.arange(len(n))!=c
        fitted=(g.mean(0)[keep]-y[keep]).sum(0)@np.linalg.inv(n[keep].sum(0))
        expected.append(fitted.reshape(3,8))
    np.testing.assert_allclose(s['residual_jackknife'],expected,atol=1e-16)
    centered=np.array(expected)-np.mean(expected,axis=0)
    cov=9/10*np.einsum('cbi,cbj->bij',centered,centered)
    np.testing.assert_allclose(s['residual_covariance'],cov,atol=1e-16)
    assert np.max(np.abs(cov[:,:4,4:]))>1e-8


def test_exact_paired_cancellation_and_unbiased_eight_component_cross():
    n,y,g=fixture()
    exact=paired_physical_case_statistics(n,y,np.repeat(y[None],4,axis=0))
    np.testing.assert_array_equal(exact['residual_covariance'],0)
    p=np.tile(np.eye(8),(3,1,1));s=paired_physical_case_statistics(n,y,g,p)
    r=s['group_model_responses']-s['measured_response']
    expected=np.mean([np.einsum('bi,bij,bj->',r[a],p,r[b])/(2*3*8) for a in range(4) for b in range(4) if a!=b])
    np.testing.assert_allclose(s['objective']['loss'],expected)


def test_fixed_average_retains_correlated_cell_and_block_uncertainty():
    n,y,g=fixture();s=paired_physical_case_statistics(n,y,g)
    weights=np.array([.2,.3,.5]);r=fixed_cell_average_statistics(s,weights)
    deleted=(s['residual_jackknife']*weights[None,:,None]).sum(1)
    centered=deleted-deleted.mean(0)
    np.testing.assert_allclose(r['residual_covariance'],9/10*centered.T@centered)
    # Cross-cell correlations must not be discarded by adding cell covariances.
    independent_cells=np.einsum('b,bij->ij',weights**2,s['residual_covariance'])
    assert not np.allclose(r['residual_covariance'],independent_cells,rtol=1e-6,atol=1e-12)
    with pytest.raises(ValueError):fixed_cell_average_statistics(s,[1,1,1])


def test_reject_malformed_group_alignment():
    n,y,g=fixture()
    with pytest.raises(ValueError):paired_physical_case_statistics(n,y,g[:3])
    with pytest.raises(ValueError):paired_physical_case_statistics(n,y,g[:,:,:2])


def test_chunked_raw_accumulation_reproduces_original_joint_targets_and_covariance():
    from sbsi.flow_physical_case_validation import physical_case_sufficient_statistics
    from sbsi.flow_physical_response import fit_physical_response,baseline_frame_gamma
    rng=np.random.default_rng(42);labels=np.arange(16)*3+5
    cases=np.repeat(labels,64);bins=np.tile(np.repeat([0,1],32),16)
    gamma=rng.normal(size=(len(cases),2))*.03
    rotated=baseline_frame_gamma(gamma,rng.normal(size=gamma.shape))
    delta=rng.normal(size=(len(cases),4))*.02;delta[:,2]*=10;delta[:,3]*=1000
    raw=physical_case_sufficient_statistics(gamma,rotated,bins,cases,delta,n_cells=2,case_labels=labels)
    pieces=[physical_case_sufficient_statistics(gamma[a:a+37],rotated[a:a+37],bins[a:a+37],cases[a:a+37],delta[a:a+37],n_cells=2,case_labels=labels) for a in range(0,len(cases),37)]
    for key in ['normal','cross','counts']:
        np.testing.assert_allclose(sum(p[key] for p in pieces),raw[key],atol=1e-12)
    fits=paired_physical_case_statistics(raw['normal'],raw['cross'],np.repeat(raw['cross'][None],4,axis=0))
    original=fit_physical_response(delta,gamma,bins,cases,2,scalar_gamma=rotated)
    np.testing.assert_allclose(fits['measured_response'],original['target'],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(fits['measurement_covariance'],original['covariance'],rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError,match='absent'):
        physical_case_sufficient_statistics(gamma,rotated,bins,cases,delta,n_cells=2,case_labels=labels+1)
