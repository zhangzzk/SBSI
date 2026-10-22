"""Paired population math and diagnostic provenance, with no scientific inputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from sbsi.prior_sensitivity import population_measure, paired_sensitivity, type_marginal_ratio
from sbsi.particle_diagnostics import combine_populations


@pytest.fixture
def populations():
    rng=np.random.default_rng(13)
    sizes=[5,7,9,6,8,5,4,7]
    values=[rng.normal(size=(n,3)) for n in sizes]
    measure=[rng.normal(0,.1,n) for n in sizes]
    ratios=[rng.normal(0,.3,n) for n in sizes]
    return rng.normal(0,.4,8),values,measure,ratios


def test_identical_ratio_has_exactly_zero_paired_contrast(populations):
    z,f,b,_=populations
    r=paired_sensitivity(z,f,b,[np.zeros(len(x)) for x in f])
    np.testing.assert_array_equal(r['numerical_minus_analytic'],0.)
    np.testing.assert_array_equal(r['paired_contrast_mcse'],0.)
    assert r['intensity_mass_ratio']==1.


def test_constant_ratio_and_intensity_scaling_invariance(populations):
    z,f,b,lr=populations
    constant=paired_sensitivity(z,f,b,[np.full(len(x),1.2) for x in f])
    np.testing.assert_allclose(constant['numerical_minus_analytic'],0.,atol=5e-16)
    assert constant['intensity_mass_ratio']==pytest.approx(np.exp(1.2))
    a=paired_sensitivity(z,f,b,lr)
    c=paired_sensitivity(z+100.,f,b,[v+4. for v in lr])
    for key in ('numerical_minus_analytic','paired_contrast_mcse','paired_contrast_covariance'):
        np.testing.assert_allclose(a[key],c[key],rtol=1e-12,atol=1e-13)
    assert c['intensity_mass_log_ratio']==pytest.approx(a['intensity_mass_log_ratio']+4.)


def test_paired_influence_equals_ratio_derivative_and_includes_covariance(populations):
    z,f,b,lr=populations
    r=paired_sensitivity(z,f,b,lr);n=len(z)
    def contrast(multipliers):
        def weighted(weights):
            mass=np.array([np.exp(v).mean() for v in weights])*np.exp(z)*multipliers
            moment=np.array([(np.exp(v)[:,None]*x).mean(0) for v,x in zip(weights,f)])*np.exp(z)[:,None]*multipliers[:,None]
            return moment.sum(0)/mass.sum()
        return weighted([x+y for x,y in zip(b,lr)])-weighted(b)
    step=1e-5
    derivative=[]
    for i in range(n):
        upper,lower=np.ones(n),np.ones(n);upper[i]+=step;lower[i]-=step
        derivative.append((contrast(upper)-contrast(lower))/(2*step)*n)
    np.testing.assert_allclose(r['paired_contrast_influence'],derivative,atol=1e-10)
    a,c=r['analytic']['summary']['covariance'],r['numerical']['summary']['covariance']
    cross=r['analytic_numerical_cross_covariance']
    np.testing.assert_allclose(r['paired_contrast_covariance'],a+c-cross-cross.T,atol=1e-17)
    assert np.any(np.abs(r['paired_contrast_mcse']-np.sqrt(np.diag(a+c)))>1e-3)


def test_original_particle_and_population_denominators_preserved(populations):
    z,f,b,lr=populations
    lr[0][:]=-np.inf;lr[1][0]=-np.inf
    r=paired_sensitivity(z,f,b,lr)
    tilted=r['numerical']
    assert tilted['summary']['independent_populations']==8
    assert tilted['summary']['extinct_populations']==1
    assert tilted['per_run_original_particles']==[len(x) for x in f]
    expected=np.array([logsumexp(x+y)-np.log(len(x)) for x,y in zip(b,lr)])
    np.testing.assert_array_equal(tilted['per_run_log_mass'],expected)
    assert tilted['summary']['weights'][0]==0.
    # Dropping the zero population changes the finite-R influence covariance.
    dropped=population_measure(z[1:],f[1:],[x+y for x,y in zip(b[1:],lr[1:])])
    assert not np.allclose(tilted['summary']['covariance'],dropped['summary']['covariance'])


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_unknown_density_failure_is_not_extinction(populations,bad):
    z,f,b,lr=populations;lr[0][0]=bad
    with pytest.raises(ValueError,match='unknown/nonfinite'):paired_sensitivity(z,f,b,lr)


def test_all_zero_observation_is_explicitly_undefined(populations):
    z,f,b,_=populations
    with pytest.raises(ValueError,match='all numerical mass'):
        paired_sensitivity(z,f,b,[np.full(len(x),-np.inf) for x in f])


def test_type_sum_includes_numerical_only_type_mass_and_common_jacobian():
    p=np.array([[np.log(2),-np.inf],[np.log(3),np.log(5)]])
    q=np.array([[np.log(4),np.log(7)],[np.log(6),np.log(10)]])
    h=np.array([[np.log(2),-np.inf],[np.log(2),np.log(2)]])
    r=type_marginal_ratio(p,q,h)
    np.testing.assert_allclose(np.exp(r['log_ratio']),[5.5,2.])
    np.testing.assert_allclose(np.exp(r['restricted_log_ratio']),[2.,2.])
    assert r['q_only_type'].tolist()==[[False,True],[False,False]]
    shifted=type_marginal_ratio(p+np.array([[100.],[-7.]]),q+np.array([[100.],[-7.]]),h)
    np.testing.assert_allclose(r['log_ratio'],shifted['log_ratio'],atol=1e-14)


def test_type_zero_and_helper_disagreement_fail_closed():
    p=np.zeros((2,2));q=np.array([[-np.inf,-np.inf],[0.,0.]])
    r=type_marginal_ratio(p,q,q)
    assert r['full_numerical_zero'].tolist()==[True,False]
    with pytest.raises(ValueError,match='support'):type_marginal_ratio(p,q,np.zeros_like(q))
    with pytest.raises(ValueError,match='reproduce'):type_marginal_ratio(p,np.zeros_like(q),np.full_like(q,.1))


def load_script():
    p=Path(__file__).resolve().parents[1]/'scripts/audit_corrected_prior_sensitivity.py'
    spec=importlib.util.spec_from_file_location('audit_prior_sensitivity',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_audited_manifest_and_data_hashes_required(tmp_path,monkeypatch):
    m=load_script();root=tmp_path/'root';source=root/'source_support_fix_20261022';source.mkdir(parents=True)
    code=source/'heldout_named_source.py';code.write_text('# pinned source, no outcomes')
    (source/'SHA256SUMS').write_text(f'{m.digest(code)}  {code.name}\n')
    declaration=root/'experiment_v10_corrected_three_way.json'
    declaration.write_text(json.dumps(dict(source=str(source),source_manifest_sha256=m.digest(source/'SHA256SUMS'))))
    data=root/'dev_v10_small_0.json';data.write_text('{}')
    audit=root/'v10_corrected_comparison_20261022_v3.json'
    record=dict(schema='sbsi-n23-corrected-comparison-v1',development_checks_pass=True,
        sources={str(p):m.digest(p) for p in (code,source/'SHA256SUMS',declaration,data)})
    audit.write_text(json.dumps(record));monkeypatch.setattr(m,'AUDIT_SHA256',m.digest(audit))
    assert m.verify_audit(root)[2]==source
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='input changed'):m.verify_audit(root)
    with pytest.raises(ValueError,match='heldout inputs'):
        m.safe_audited_path(root/'heldout_corrected'/'result.json',root,source)
    assert m.safe_audited_path(code,root,source)==code


def test_summary_replay_fails_before_any_interpretation():
    m=load_script()
    with pytest.raises(ValueError,match='baseline/coordinate replay'):m.close([1.],[2.],'mean')
    assert m.serial({'log_mass':-np.inf})=={'log_mass':None}
    with pytest.raises(ValueError):m.serial({'bad':np.nan})
