"""Exact weighted accounting and preserved-population sensitivity checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec=importlib.util.spec_from_file_location('localization',
    Path(__file__).resolve().parents[1]/'scripts/localize_particle_phase_score.py')
local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)


def test_fractional_absolute_tail_keeps_signed_positive_negative_and_complement_mass():
    score=np.array([-10.,8.,2.]); weights=np.array([.02,.08,.90])
    one=local.tail_diagnostic(score,weights,.01)
    assert one['actual_mass']==pytest.approx(.01)
    assert one['signed_mean_contribution']==pytest.approx(-.1)
    assert one['particles_touched']==1
    five=local.tail_diagnostic(score,weights,.05,np.array([0,1,1]))
    assert five['signed_mean_contribution']==pytest.approx(-.2+.03*8)
    assert five['positive_mean_contribution']==pytest.approx(.24)
    assert five['negative_mean_contribution']==pytest.approx(-.2)
    assert five['second_moment_share']==pytest.approx((.02*100+.03*64)/(weights@score**2))
    assert five['signed_mean_contribution']+five['complement_mean_contribution']==pytest.approx(weights@score)
    assert sum(v['mass'] for v in five['by_population'])==pytest.approx(.05)


def test_zero_score_tails_and_constant_feature_correlation_are_explicit():
    weights=np.ones(3)/3
    tail=local.tail_diagnostic(np.zeros(3),weights,.05)
    assert tail['absolute_score_share'] is None and tail['second_moment_share'] is None
    feature=local.feature_diagnostic(np.ones(3),np.arange(3.),weights)
    assert feature['score_covariance']==0 and feature['score_correlation'] is None


def test_weighted_correlation_and_quantiles_do_not_pool_unequal_runs_as_equal_particles():
    x=np.array([0.,1.,2.]); s=3*x+4; w=np.array([.8,.1,.1])
    result=local.feature_diagnostic(x,s,w)
    assert result['mean']==pytest.approx(.3)
    assert result['score_covariance']==pytest.approx(3*(w@x**2-(w@x)**2))
    assert result['score_correlation']==pytest.approx(1.)
    assert result['quantiles']['0.5']==0
    with pytest.raises(ValueError,match='normalized'):
        local.distribution(x,w*2)


def test_whole_population_Z_weight_and_leave_one_sensitivity():
    result=local.population_sensitivity(np.log([1.,2.,7.]),[0.,2.,10.])
    assert result['complete']['mean'][0]==pytest.approx(7.4)
    np.testing.assert_allclose(result['complete']['weights'],[.1,.2,.7])
    loo=result['leave_one_population_out']
    assert loo[2]['mean']==pytest.approx(4/3)
    assert loo[2]['shift_from_complete']==pytest.approx(4/3-7.4)
    assert 'mcse' not in loo[2]


def test_fixed_bins_and_midpoint_accounting_include_unmatched_support():
    edges=[-np.inf,0.,1.,np.inf]
    left=local.binned_contributions([-.5,.5,2.],[4.,2.,8.],[.2,.4,.4],edges)
    right=local.binned_contributions([.5,2.],[1.,3.],[.6,.4],edges)
    diff=local.binned_difference(left,right)
    assert sum(left['score_mean_contribution'])==pytest.approx(4.8)
    assert diff['unmatched_bin_component']==pytest.approx(.8)
    assert sum(diff[k] for k in ['composition_component','within_bin_component','unmatched_bin_component'])==pytest.approx(diff['total_difference'])
    assert diff['total_difference']==pytest.approx(3.)


def test_ancestry_is_genealogy_with_exact_score_contributions():
    result=local.ancestry_diagnostic([0,0,1,2],[-10.,10.,1.,2.],np.ones(4)/4)
    assert result['distinct_initial_ancestors']==3
    assert result['mass_ess']==pytest.approx(1/(.5**2+.25**2+.25**2))
    biggest=result['top_five_by_absolute_score'][0]
    assert biggest['label']==0 and biggest['signed_mean_contribution']==0
    assert biggest['absolute_score_share']==pytest.approx(20/23)
    assert biggest['second_moment_share']==pytest.approx(200/205)


def test_phase_wrap_is_circular_not_linear():
    result=local.phase_diagnostic([-.01,2*np.pi+.01],[.5,.5])
    assert min(result['circular_mean'],2*np.pi-result['circular_mean'])<1e-12
    assert result['mean_resultant_length']==pytest.approx(np.cos(.01))


def test_native_scene_extraction_indexes_observation_and_CSR_correctly():
    x=np.zeros((2,10));x[:,1]=[20.,21.];x[:,6]=np.log([.2,.3]);x[:,9]=[.3,6.]
    summary=np.zeros((2,9));summary[:,2]=[1.,2.];summary[:,3]=x[:,1];summary[:,4]=np.exp(x[:,6])
    scene=dict(target_psi=np.r_[9.,x[:,9]],target_m_r=[0.,20.,21.],target_n=[0.,1.,2.],
        target_a=np.array([0.,.2,.4]),target_R_kpc=np.array([0.,1.,2.]),
        indptr=np.array([0,1,3,3]),offset_arcsec=np.array([[99.,0.],[.2,0.],[1.,0.]]),
        neighbour_m_r=np.array([99.,20.,22.]))
    for k in ['target_m_r','target_n']:scene[k]=np.array(scene[k])
    features,phase=local.extract_features(x,summary,scene,1,2)
    np.testing.assert_array_equal(features['neighbour_count'],[2,0])
    np.testing.assert_array_equal(features['neighbours_within_0.5_arcsec'],[1,0])
    np.testing.assert_array_equal(features['neighbours_within_1_arcsec'],[1,0])
    assert features['log1p_neighbour_target_flux_sum'][0]==pytest.approx(np.log1p(1+10**(-.8)))
    assert features['log1p_neighbour_target_flux_sum'][1]==0
    np.testing.assert_array_equal(phase,x[:,9])
    scene['target_m_r'][1]+=1
    with pytest.raises(ValueError,match='disagree'):
        local.extract_features(x,summary,scene,1,2)


def test_group_pooling_preserves_Z_over_N_weights_and_tail_accounting():
    populations=[]
    for i,(n,z,score) in enumerate([(2,1.,[0.,2.]),(4,3.,[4.,4.,4.,4.])]):
        features={k:np.full(n,.5) for k in local.BINS if k!='phase'}
        features['m_r']=np.full(n,24.)
        populations.append(dict(index=i,seed=i,particles=n,log_normalizer=np.log(z),
            score=np.array(score),raw_score=np.array(score),control=np.zeros(n),features=features,
            phase=np.zeros(n),labels=np.arange(n)))
    result=local.summarize_group(populations)
    assert result['complete']['mean'][0]==pytest.approx(.25*1+.75*4)
    assert result['score_distribution']['mean']==pytest.approx(result['complete']['mean'][0])
    for tail in result['absolute_score_tails'].values():
        assert tail['signed_mean_contribution']+tail['complement_mean_contribution']==pytest.approx(result['complete']['mean'][0])
    for bins in result['fixed_bins'].values():
        assert sum(bins['score_mean_contribution'])==pytest.approx(result['complete']['mean'][0])
