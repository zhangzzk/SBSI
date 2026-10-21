"""Pre-outcome tests for fixed gates, complete inputs, zero mass and provenance."""
import copy
import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('v10summary',Path(__file__).resolve().parents[1]/'scripts/summarize_v10_particle_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from sbsi.particle_comparison_gates import difference_gate,normalizer_gate,raw_pair_gate,raw_rows,corrected_phase_gate


def summary(mean=(0.,0.),se=(1.,1.)):
    return dict(mean=list(mean),mcse=list(se),log_mean_normalizer=0.,normalizer_relative_mcse=.01)


def raw_row():
    return dict(all_populations_extinct=False,score=summary(),region=summary(se=(.1,.1)),
                posterior_score_variance_estimate=[10000.,10000.])


def test_signed_independent_combined_error_and_exact_fixed_boundary():
    a=summary((3.5,-3.5),(1.,1.));b=summary(se=(0.,0.))
    c=difference_gate(a,b)
    assert c['combined_mcse']==[1.,1.] and c['component_pass']==[True,True]
    assert c['signed_standardized_difference']==[3.5,-3.5]
    a['mean'][1]=math.nextafter(-3.5,-math.inf)
    assert difference_gate(a,b)['component_pass']==[True,False]
    assert difference_gate(summary((1.,0.),(0.,0.)),b)['component_pass']==[False,True]


def test_raw_gate_also_enforces_absolute_score_and_region_bounds():
    a,b=raw_row(),raw_row();a['score']['mean'][0]=1.
    a['posterior_score_variance_estimate']=b['posterior_score_variance_estimate']=[1.,1.]
    assert not raw_pair_gate(a,b)['pass_all']
    a,b=raw_row(),raw_row();a['region']['mean'][1]=.051
    out=raw_pair_gate(a,b)
    assert out['region']['pass_all'] and not out['region_absolute_pass'][1] and not out['pass_all']


def test_raw_normalizer_gate_not_replaced_by_score_agreement():
    a,b=raw_row(),raw_row();a['score']['log_mean_normalizer']=.2
    assert not raw_pair_gate(a,b)['normalizer']['pass_all']
    a['score']['log_mean_normalizer']+=1000;b['score']['log_mean_normalizer']+=1000
    assert not raw_pair_gate(a,b)['pass_all']


@pytest.mark.parametrize('measure',m.MEASURES)
@pytest.mark.parametrize('component',[0,1])
def test_each_corrected_component_required_for_each_measure(measure,component):
    a={'mean':[0.,0.,0.,0.,0.],'mcse':[1.,1.,1.,1.,1.]};b=copy.deepcopy(a)
    a['mean'][3+component]=3.5*np.sqrt(2)+.001
    assert not corrected_phase_gate(a,b)['pass_all']
    # A passing other numerical measure cannot cancel this failure.
    gates={x:corrected_phase_gate(a if x==measure else b,b)['pass_all'] for x in m.MEASURES}
    assert not all(gates.values())


def test_precision_and_full_replicate_denominator_preserve_zero_mass():
    logz=np.zeros((8,1));logz[0,0]=-np.inf
    means=np.zeros((8,1,2));means[:,0,0]=np.arange(8)*.01;means[:,0,1]=np.arange(8)*.02
    variance=np.ones_like(means)*10;regions=np.ones_like(means)*.2
    means[0]=np.nan;variance[0]=np.nan;regions[0]=np.nan
    row=raw_rows(logz,means,variance,regions,256,[1])[0]
    assert row['score']['independent_populations']==8 and row['score']['extinct_populations']==1
    assert row['score']['weights'][0]==0.
    assert row['score']['normalizer_relative_mcse']==pytest.approx(1/7)
    assert not row['precision_gate'] # Relative Z MCSE exceeds .1.
    np.testing.assert_allclose(row['score']['mean'],[.04,.08])
    allzero=raw_rows(np.full((8,1),-np.inf),means,variance,regions,256,[1])[0]
    assert allzero['all_populations_extinct'] and not allzero['precision_gate']


def test_zero_score_mcse_does_not_create_infinite_precision_pass():
    row=raw_rows(np.zeros((8,1)),np.zeros((8,1,2)),np.ones((8,1,2)),np.zeros((8,1,2)),256,[1])[0]
    assert row['score_effective_count_from_replicates']==[None,None]
    assert not row['precision_gate']


def scheduler_text():
    return '\n'.join(f'{j}|COMPLETED|0:0|10|cpu=2,gres/gpu=1' for j in
        [f'{m.ARRAY_JOB}_{i}' for i in range(24)]+[m.REPORT_JOB,m.PHASE_JOB])


def test_completion_guard_requires_every_array_task_and_both_reports(tmp_path):
    assert len(m.completion_records(scheduler_text()))==26
    with pytest.raises(RuntimeError):m.completion_records(scheduler_text().replace(f'{m.ARRAY_JOB}_23|COMPLETED',f'{m.ARRAY_JOB}_23|RUNNING'))
    with pytest.raises(RuntimeError):m.completion_records(scheduler_text().replace(f'{m.PHASE_JOB}|COMPLETED|0:0',f'{m.PHASE_JOB}|FAILED|1:0'))
    with pytest.raises(RuntimeError):m.build_summary(tmp_path/'does-not-exist',{})
    assert not (tmp_path/'does-not-exist').exists()


def protocol_fixture(arm=2):
    declaration={'physical_geometry':{'sha256':'geometry'},'initial_proposal':{'sha256':'initial'}}
    a=m.ARMS[arm]
    p=dict(particles=a['particles'],stages=64,moves=64,seed=a['seed_base'],observations=m.OBS,
        fixed_shear=[0.,0.],density_batch_size=4096,blocks=m.BLOCKS.copy(),scale=1.,
        endpoint_target=m.raw_report.ENDPOINT_TARGET,betas=(np.linspace(0,1,65)**2).tolist(),
        physical_geometry={'sha256':'geometry','metadata':{'pilot_seeds':[11,12,13,14]}},
        proposals=[{'component_family':'student_t','student_df':5.}]*4)
    if arm!=2:p['initial_proposal']={'sha256':'initial','metadata':{'pilot_seeds':[11,12,13,14]}}
    return p,a,declaration


@pytest.mark.parametrize('defect',['seed','particles','moves','proposal','pilot','blocks','schedule','geometry'])
def test_protocol_drift_rejected(defect):
    p,a,d=protocol_fixture();m.check_protocol(p,a,0,d)
    if defect=='seed':p['seed']+=1
    if defect=='particles':p['particles']=256
    if defect=='moves':p['moves']=32
    if defect=='proposal':p['proposals'][0]['student_df']=3.
    if defect=='pilot':p['physical_geometry']['metadata']['pilot_seeds'].append(p['seed'])
    if defect=='blocks':p['blocks'][0]='other'
    if defect=='schedule':p['betas'][1]*=2
    if defect=='geometry':p['physical_geometry']['sha256']='changed'
    with pytest.raises(ValueError):m.check_protocol(p,a,0,d)


def phase_fixture():
    n=4;p={'particles':n,'seed':99};shape=(4,n)
    saved={'scores':np.ones(shape+(2,)),'log_kernel':np.ones(shape)}
    a=dict(raw_scores=saved['scores'].copy(),saved_raw_scores=saved['scores'].copy(),
        corrected_scores=saved['scores'].copy(),phase_control=np.zeros(shape),delta_log_kernel=np.zeros(shape),
        density32=saved['log_kernel'].copy(),density64=saved['log_kernel'].copy(),observations=np.array(m.OBS),sampled_particle_indices=np.arange(n))
    d=dict(stride=1,retained_particles=n,observations=m.OBS,seed=99,numerical_checks_pass=True,
        precision_limits=m.LIMITS.copy(),derivative_protocol='phase-fine-difference-v2',exact_zero_mean_for_float32_target_established=False,
        numerical_checks=dict(sampling_replay_max=0.,float64_replay_max=0.,phase_wrap_max=0.,saved_score_replay_max=0.,
            phase_differences=[dict(step=h,max_scaled_error=0.) for h in m.STEPS]))
    return d,a,p,saved,np.ones((4,2))


@pytest.mark.parametrize('defect',['stride','indices','array_count','corrected','delta','step','threshold','f32_claim','score'])
def test_full_phase_integrity_rejects_missing_or_altered_rows(defect):
    d,a,p,s,c=phase_fixture();m.check_phase_arrays(d,a,p,s,c)
    if defect=='stride':d['stride']=2
    if defect=='indices':a['sampled_particle_indices']=a['sampled_particle_indices'][::-1]
    if defect=='array_count':a['raw_scores']=a['raw_scores'][:,:2]
    if defect=='corrected':a['corrected_scores'][0,0,0]+=.01
    if defect=='delta':a['delta_log_kernel'][0,0]=.1
    if defect=='step':d['numerical_checks']['phase_differences'][0]['step']=1e-4
    if defect=='threshold':d['precision_limits']['phase_gradient_scaled_atol']=.01
    if defect=='f32_claim':d['exact_zero_mean_for_float32_target_established']=True
    if defect=='score':a['saved_raw_scores'][0,0,0]+=.1
    with pytest.raises(ValueError):m.check_phase_arrays(d,a,p,s,c)


def test_weight_replay_is_finite_eight_ulps_without_renormalization():
    assert m.vectors_match_ulps([.125],[math.nextafter(.125,1.)])
    assert not m.vectors_match_ulps([.125],[.125+1e-8])
    assert not m.vectors_match_ulps([.125],[float('nan')])


def test_empty_comparisons_cannot_pass_vacuously():
    with pytest.raises(ValueError,match='finite aligned'):
        difference_gate({'mean':[],'mcse':[]},{'mean':[],'mcse':[]})


@pytest.mark.parametrize('count',[0,3,4,6])
def test_corrected_gate_requires_exactly_five_complete_moments(count):
    good={'mean':[0.]*5,'mcse':[1.]*5}
    bad={'mean':[0.]*count,'mcse':[1.]*count}
    with pytest.raises(ValueError,match='exactly five'):corrected_phase_gate(good,bad)


def test_corrected_gate_rejects_nonfinite_unselected_moments():
    a={'mean':[float('nan'),0.,0.,0.,0.],'mcse':[1.]*5}
    b={'mean':[0.]*5,'mcse':[1.]*5}
    with pytest.raises(ValueError,match='exactly five'):corrected_phase_gate(a,b)
