#!/usr/bin/env python3
"""Frozen n23 audit. Refuses outcome reads before all 24 tasks/reports finish."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import report_galsbi_particles as raw_report
from sbsi.particle_comparison_gates import (raw_rows,raw_pair_gate,corrected_phase_gate,difference_gate)
from sbsi.phase_control import FrozenPhaseControl, summarize_phase_populations
from summarize_v8_particle_comparison import vectors_match_ulps, comparison_status
from summarize_v9_particle_comparison import sha256,COLUMNS,MEASURES,qualified_moments

ARRAY_JOB='17422526';REPORT_JOB='17422589';PHASE_JOB='17422590'
DECLARATION_HASH='9102eb816c20ae4d0a8dbc6f698db194f4ed4ea0eea726138a3f73fb4545d72e'
LAUNCHER_HASH='d1e468b9c9cf0abe8187b078643ab74eba8fd27aeb3f4430d44f758b6e145df3'
OBS=[58165,0,1,8]
ARMS=[dict(name='small',particles=256,proposal='fitted',seed_base=2026102900),
      dict(name='large',particles=1024,proposal='fitted',seed_base=2026103000),
      dict(name='broad',particles=1024,proposal='student_t',df=5,seed_base=2026103100)]
BLOCKS=['physical','physical','nuisance','physical','environment','physical','sed','physical',
        'positions','physical','exchange','physical','nuisance','physical','independence','physical']
LIMITS=dict(sampling_replay_atol=1e-6,float64_replay_atol=1e-10,saved_score_atol=1e-6,
            phase_gradient_scaled_atol=1e-4,sampled_float32_float64_atol=1e-5)
STEPS=[2e-6,1e-6,5e-7]


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x


def completion_records(text):
    records={}
    for line in text.splitlines():
        fields=[v.strip() for v in line.split('|')]
        if len(fields)<5:continue
        job,state,exit_code,elapsed,tres=fields[:5]
        if '.' in job:continue
        if job in records:raise ValueError('duplicate scheduler job record')
        records[job]=dict(state=state,exit_code=exit_code,elapsed_seconds=int(elapsed),allocated_tres=tres)
    required=[f'{ARRAY_JOB}_{i}' for i in range(24)]+[REPORT_JOB,PHASE_JOB]
    if any(records.get(j,{}).get('state')!='COMPLETED' or records[j]['exit_code']!='0:0' for j in required):
        raise RuntimeError('No v10 outcomes read: all 24 array tasks and raw/phase jobs must be COMPLETED/0:0')
    return {j:records[j] for j in required}


def require_completion():
    command=['sacct','-n','-P','-j',','.join([ARRAY_JOB,REPORT_JOB,PHASE_JOB]),
             '--format=JobID%64,State%30,ExitCode,ElapsedRaw,AllocTRES%200']
    return completion_records(subprocess.check_output(command,text=True))


def assert_close(actual,expected,label,*,atol=1e-11):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=1e-11,atol=atol):
        raise ValueError(f'{label}: finite aligned replay mismatch')


def check_summary(actual,expected,label):
    for key in ('mean','mcse','covariance','run_weight_ess','log_mean_normalizer','normalizer_relative_mcse'):
        assert_close(actual[key],expected[key],f'{label}/{key}')
    if not vectors_match_ulps(actual['weights'],expected['weights']):
        raise ValueError(f'{label}: weights exceed eight-ULP replay tolerance')
    for key in ('independent_populations','extinct_populations'):
        if actual[key]!=expected[key]:raise ValueError(f'{label}: population counts differ')


def check_protocol(p,arm,index,declaration):
    expected=dict(particles=arm['particles'],stages=64,moves=64,seed=arm['seed_base']+index,
        observations=OBS,fixed_shear=[0.,0.],density_batch_size=4096,blocks=BLOCKS,scale=1.,
        endpoint_target=raw_report.ENDPOINT_TARGET)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('declared protocol mismatch')
    if p.get('informed_environment') is not None:raise ValueError('undeclared informed environment')
    if p.get('physical_geometry',{}).get('sha256')!=declaration['physical_geometry']['sha256']:
        raise ValueError('geometry artifact mismatch')
    if arm['proposal']=='fitted':
        if p.get('initial_proposal',{}).get('sha256')!=declaration['initial_proposal']['sha256']:
            raise ValueError('fitted initial artifact mismatch')
    elif p.get('initial_proposal') is not None or any(q.get('component_family')!='student_t' or q.get('student_df')!=5. for q in p['proposals']):
        raise ValueError('broad arm must use original Student-t(df5) proposal')
    if len(p['proposals'])!=4:raise ValueError('four target proposal laws required')
    assert_close(p['betas'],np.linspace(0,1,65)**2,'fixed beta schedule',atol=0.)
    if p['seed'] in raw_report.pilot_seeds(p):raise ValueError('production/pilot seed overlap')


def check_phase_arrays(d,a,p,saved,coefficients):
    n=p['particles'];shape=(4,n)
    if d.get('stride')!=1 or d.get('retained_particles')!=n or d.get('observations')!=OBS:
        raise ValueError('complete ordered phase population required')
    if d.get('seed')!=p['seed'] or d.get('numerical_checks_pass') is not True:
        raise ValueError('wrong seed or failed phase replay')
    if d.get('precision_limits')!=LIMITS or d.get('derivative_protocol')!='phase-fine-difference-v2':
        raise ValueError('phase precision protocol changed')
    checks=d['numerical_checks'];fd=checks['phase_differences']
    if [x['step'] for x in fd]!=STEPS or any(x['max_scaled_error']>1e-4 or not math.isfinite(x['max_scaled_error']) for x in fd):
        raise ValueError('phase finite-difference checks failed')
    for key,limit in [('sampling_replay_max',1e-6),('float64_replay_max',1e-10),('phase_wrap_max',1e-10),('saved_score_replay_max',1e-6)]:
        if not math.isfinite(checks[key]) or not 0<=checks[key]<=limit:raise ValueError('phase replay check failed')
    if not np.array_equal(a['observations'],OBS) or not np.array_equal(a['sampled_particle_indices'],np.arange(n)):
        raise ValueError('thinned/reordered particle indices')
    for key,dims in [('raw_scores',(2,)),('saved_raw_scores',(2,)),('corrected_scores',(2,)),('phase_control',()),('delta_log_kernel',()),('density32',()),('density64',())]:
        if a[key].shape!=shape+dims or not np.isfinite(a[key]).all():raise ValueError('invalid complete phase arrays')
    if not np.array_equal(a['saved_raw_scores'],saved['scores']):raise ValueError('saved score bytes changed')
    assert_close(a['raw_scores'],saved['scores'],'phase/saved score',atol=1e-6)
    assert_close(a['density32'],saved['log_kernel'],'saved sampling kernel',atol=1e-6)
    if not np.array_equal(a['delta_log_kernel'],a['density64']-saved['log_kernel']):raise ValueError('F64 measure delta mismatch')
    assert_close(a['corrected_scores'],a['raw_scores']-coefficients[:,None,:]*a['phase_control'][:,:,None],'frozen score law',atol=1e-12)
    if d.get('exact_zero_mean_for_float32_target_established') is not False:raise ValueError('invalid F32 phase identity claim')


def guard_input_path(path,expected,source_root,source_manifest):
    """Permit pinned source filenames, never arbitrary held-out data paths.

    The caller populates source_manifest only after checking its pinned hash;
    record() still verifies the actual file bytes against expected afterwards.
    """
    path=Path(path).resolve()
    if 'heldout' not in str(path).lower():return
    pinned=(source_root is not None and Path(source_root).resolve() in path.parents
            and expected is not None and source_manifest.get(str(path))==expected)
    if not pinned:raise ValueError('heldout data inputs forbidden; only pinned source-code manifest entries are allowed')


def build_summary(root,completion):
    # No bypass: reconstruct and validate the required successful job set first.
    completion=completion_records('\n'.join(f'{j}|{r["state"]}|{r["exit_code"]}|{r["elapsed_seconds"]}|{r["allocated_tres"]}' for j,r in completion.items()))
    root=Path(root).resolve();sources={};source=None;declared_sources={}
    def record(path,expected=None):
        path=Path(path).resolve()
        guard_input_path(path,expected,source,declared_sources)
        value=sources.get(str(path)) or sha256(path)
        if expected is not None and value!=expected:raise ValueError(f'hash mismatch: {path}')
        sources[str(path)]=value;return path
    def read(path,expected=None):return json.loads(record(path,expected).read_text())
    declaration=read(root/'experiment_v10_corrected_three_way.json',DECLARATION_HASH)
    record(root/'run_v10_corrected.sh',LAUNCHER_HASH)
    if declaration['arms']!=ARMS or declaration['observations']!=OBS or declaration['moves']!=64 or declaration['stages']!=64 or declaration['replicas_per_arm']!=8:
        raise ValueError('n23 declaration differs from fixed task')
    source=Path(declaration['source']).resolve()
    manifest=record(source/'SHA256SUMS',declaration['source_manifest_sha256'])
    for line in manifest.read_text().splitlines():
        digest,name=line.split(None,1);name=name.lstrip('*');p=(source/name).resolve()
        if source not in p.parents:raise ValueError('unsafe source manifest path')
        if str(p) in declared_sources:raise ValueError('duplicate pinned source path')
        declared_sources[str(p)]=digest
    for path,digest in declared_sources.items():record(path,digest)
    for local in [Path(raw_report.__file__),REPO/'sbsi/phase_control.py',REPO/'sbsi/particle_diagnostics.py']:
        record(local,declared_sources[str(source/local.relative_to(REPO))])
    artifacts={key:read(declaration[key]['path'],declaration[key]['sha256']) for key in ('initial_proposal','physical_geometry','phase_control')}
    frozen=FrozenPhaseControl.load(declaration['phase_control']['path'])
    from sbsi.particle_initial_proposal import FrozenParticleProposal
    fitted=FrozenParticleProposal.load(declaration['initial_proposal']['path'])
    groups={};reports={};identity=None;common=None
    for arm in ARMS:
        name=arm['name'];runs=[root/f'dev_v10_{name}_{i}' for i in range(8)]
        report=read(root/f'dev_v10_{name}_report.json');phase=read(root/f'phase_v10_20261022/{name}_controls.json')
        seeds=list(range(arm['seed_base'],arm['seed_base']+8))
        if report['runs']!=[str(p) for p in runs] or report['seeds']!=seeds or phase['seeds']!=seeds or report['replicas']!=8:
            raise ValueError('eight complete declared populations required')
        if report['diagnostics_version']!=raw_report.REPORT_VERSION or phase['schema']!='sbsi-independent-phase-diagnostics-v2' or phase['moment_columns']!=COLUMNS:
            raise ValueError('raw/phase report version mismatch')
        for obj in (report,phase):
            if [x['observation'] for x in obj['observations']]!=OBS:raise ValueError('observation panel changed')
        expected_sources={str(p/f) for p in runs for f in ('protocol.json','result.json','particles.npz','scenes.npz')}
        if set(report['sources'])!=expected_sources:raise ValueError('incomplete raw provenance')
        expected_phase=[root/f'phase_v10_20261022/{name}_{i}.json' for i in range(8)]
        if set(phase['sources'])!={str(p) for p in expected_phase}:raise ValueError('incomplete phase provenance')
        for obj in (report,phase):
            for path,digest in obj['sources'].items():record(path,digest)
        record(source/'scripts/report_galsbi_particles.py',report['script_sha256'])
        record(source/'scripts/report_particle_phase_control.py',phase['script_sha256'])
        if phase['coefficients']['sha256']!=declaration['phase_control']['sha256']:raise ValueError('phase artifact changed')
        decoded=[];phase_arrays=[];diagnostics=[];timings=[];protocols=[]
        for i,run in enumerate(runs):
            p=read(run/'protocol.json');result=read(run/'result.json');check_protocol(p,arm,i,declaration)
            record(run/'protocol.json',result['protocol_sha256']);record(run/'particles.npz',result['particles_sha256']);record(run/'scenes.npz',result['scenes_sha256'])
            for path,digest in p['implementation_sha256'].items():
                if declared_sources.get(str(Path(path).resolve()))!=digest:raise ValueError('run source is not frozen support-fix source')
            for key in ('physical_geometry','initial_proposal'):
                if p.get(key) is not None:
                    artifact=artifacts[key]
                    if p[key]['metadata']!=artifact['metadata']:raise ValueError('artifact metadata differs from pinned file')
            for key in ('observations','fixed_shear','reference_protocol_sha256','observation_sha256','density_batch_size','bridge','selection'):
                if p[key]!=artifacts['initial_proposal']['metadata'][key]:raise ValueError('target differs from frozen artifact identity')
            if arm['proposal']=='fitted' and p['proposals']!=[q.metadata for q in fitted.proposals]:
                raise ValueError('effective fitted law differs from frozen artifact')
            if arm['proposal']=='student_t':
                for q,f in zip(p['proposals'],artifacts['initial_proposal']['proposals']):
                    for key in ('means','scales','weights'):
                        assert_close(q[key],f['broad_measurement_proposal'][key],'original broad law')
                    if q.get('phase')!='independent uniform periodic' or q.get('retry_or_clip_draws') is not False:
                        raise ValueError('broad phase/support law changed')
            fingerprint=raw_report.implementation_fingerprint(p)
            required={'check_galsbi_particles.py','annealed_particles.py','galsbi_particle_proposal.py','galsbi_particle_target.py',
                'scene_exchange.py','galsbi_target_coordinates.py','galsbi_summary.py','summary_gradients.py',
                'models_v3_9_like.json','particle_populations.py','particle_geometry.py'}
            if arm['proposal']=='fitted':required.add('particle_initial_proposal.py')
            if set(fingerprint)!=required or report['implementation_sha256']!=fingerprint:raise ValueError('incomplete or changed recorded implementation')
            if p['runtime'].get('device')!='cuda' or 'V100' not in p['runtime'].get('cuda_device_name',''):
                raise ValueError('declared V100 runtime required')
            if report['target_source_sha256']!={k:fingerprint[k] for k in raw_report.TARGET_SOURCES}:raise ValueError('report target source mismatch')
            for key in ('reference_protocol_sha256','observation_sha256','density_batch_size','bridge','selection','endpoint_target'):
                if p[key]!=report[key]:raise ValueError('raw report target/protocol mismatch')
            shared={k:p[k] for k in ('algorithm','reference_protocol_sha256','observation_sha256','density_batch_size','bridge','selection','endpoint_target','betas','blocks','scale')}
            if common is None:common=shared
            elif shared!=common:raise ValueError('cross-arm common protocol changed')
            record(Path(p['reference'])/'protocol.json',p['reference_protocol_sha256'])
            record(Path(p['reference'])/'observations.parquet',p['observation_sha256'])
            value=raw_report.validate_completed_result(p,result);decoded.append(value)
            # Explicit extinctions never become dropped populations or synthetic phase zeros.
            if value[4].any():raise ValueError('explicit zero-mass run retained in raw receipts; phase gate blocked, never discard/refit')
            with np.load(run/'particles.npz',allow_pickle=False) as data:saved={k:data[k] for k in data.files}
            for key,dims in [('coordinates',(10,)),('summaries',(9,)),('scores',(2,)),('log_kernel',())]:
                if saved[key].shape!=(4,arm['particles'])+dims or not np.isfinite(saved[key]).all():raise ValueError('incomplete particle arrays')
            if saved['extinct'].dtype!=np.dtype(bool) or saved['extinct'].any() or not np.array_equal(saved['log_normalizer'],value[0]):raise ValueError('zero mass/normalizer artifact mismatch')
            assert_close(saved['scores'].mean(1),value[1],'saved score means');assert_close(saved['scores'].var(1,ddof=1),value[2],'saved score variance')
            if not math.isfinite(result['final_state_replay_max']) or not 0<=result['final_state_replay_max']<=1e-6:raise ValueError('saved final state replay failed')
            with np.load(run/'scenes.npz',allow_pickle=False) as scene:
                ptr=scene['indptr'];offset=scene['offset_arcsec'];count=4*arm['particles']
                if ptr.shape!=(count+1,) or ptr.dtype.kind not in 'iu' or ptr[0]!=0 or np.any(np.diff(ptr)<0) or offset.shape!=(ptr[-1],2):
                    raise ValueError('invalid ordered complete scene CSR')
                if any(scene[k].shape[0]!=(count if k.startswith('target_') else ptr[-1]) for k in scene.files if k.startswith(('target_','neighbour_'))):
                    raise ValueError('incomplete scene marks')
            d=read(expected_phase[i]);record(expected_phase[i].with_suffix('.npz'),d['arrays_sha256'])
            if d['run']!=str(run) or d['log_normalizer']!=result['log_normalizer']:raise ValueError('phase run/normalizer mismatch')
            if set(d['source_hashes'])!={str(run/f) for f in ('protocol.json','result.json','particles.npz')}:raise ValueError('phase source list mismatch')
            for path,digest in d['source_hashes'].items():record(path,digest)
            if set(d['implementation_sha256'])!={str(source/'scripts/check_particle_phase_control.py'),str(source/'sbsi/phase_control.py')}:
                raise ValueError('incomplete phase implementation provenance')
            for path,digest in d['implementation_sha256'].items():
                if declared_sources.get(str(Path(path).resolve()))!=digest:raise ValueError('phase source not frozen support-fix source')
            if d['coefficients']['sha256']!=declaration['phase_control']['sha256']:raise ValueError('per-run frozen coefficient mismatch')
            frozen.validate_production(p['seed'],OBS,d['target_identity'])
            if identity is None:identity=d['target_identity']
            elif identity!=d['target_identity']:raise ValueError('phase target identities differ')
            if phase['target_identity']!=identity:raise ValueError('phase aggregate identity differs')
            with np.load(expected_phase[i].with_suffix('.npz'),allow_pickle=False) as data:a={k:data[k] for k in data.files}
            check_phase_arrays(d,a,p,saved,frozen.coefficients)
            phase_arrays.append(a);diagnostics.append(d);timings.append(result['timing_seconds']);protocols.append(p)
        rows=raw_rows(*[np.asarray([v[k] for v in decoded]) for k in range(4)],arm['particles'],OBS)
        phaserows=summarize_phase_populations([v[0] for v in decoded],[a['phase_control'] for a in phase_arrays],
            [a['raw_scores'] for a in phase_arrays],[a['delta_log_kernel'] for a in phase_arrays],frozen.coefficients)
        for recomputed,stored,ph,storedph in zip(rows,report['observations'],phaserows,phase['observations']):
            for key in ('score','region'):check_summary(recomputed[key],stored[key],'raw '+key)
            assert_close(recomputed['posterior_score_variance_estimate'],stored['posterior_score_variance_estimate'],'posterior score variance')
            left,right=recomputed['score_effective_count_from_replicates'],stored['score_effective_count_from_replicates']
            if len(left)!=len(right) or any((a is None)!=(b is None) for a,b in zip(left,right)):raise ValueError('undefined effective counts differ')
            assert_close([a for a in left if a is not None],[b for b in right if b is not None],'effective counts')
            if recomputed['precision_gate']!=stored['precision_gate']:raise ValueError('raw precision gate does not replay')
            for measure in MEASURES:
                check_summary(ph[measure],storedph[measure],'phase '+measure)
                if storedph[measure]['control_known_expectation']!=(None if measure==MEASURES[0] else 0.):raise ValueError('phase numerical-measure metadata changed')
            if not vectors_match_ulps(recomputed['score']['weights'],ph[MEASURES[0]]['weights']):raise ValueError('raw-phase weights do not replay')
            recomputed.update({m:qualified_moments(serial(ph[m]),m) for m in MEASURES})
        if report['all_precision_gates']!=all(r['precision_gate'] for r in rows):raise ValueError('aggregate raw precision verdict mismatch')
        groups[name]=dict(configuration=arm,seeds=seeds,observations=rows,
            all_raw_precision_gates=report['all_precision_gates'],original_comparison=comparison_status(report),
            timing_seconds_by_run=timings,total_recorded_inference_seconds=sum(t['total'] for t in timings),
            median_recorded_inference_seconds=statistics.median(t['total'] for t in timings),
            extinct_population_masks=[v[4].tolist() for v in decoded])
        reports[name]=report
    comparisons={}
    for left,right in [('small','large'),('small','broad'),('large','broad')]:
        raw_report.comparison_target_identity(reports[left],reports[right])
        current=read(root/'dev_v10_small_vs_large_report.json') if right=='large' else reports[left]
        if current['observations']!=reports[left]['observations']:raise ValueError('pairwise raw report disagrees with main report')
        record(current['comparison']['path'],current['comparison']['sha256'])
        if Path(current['comparison']['path']).resolve()!=root/f'dev_v10_{right}_report.json':raise ValueError('wrong pairwise comparator')
        rows=[]
        for a,b,old in zip(groups[left]['observations'],groups[right]['observations'],current['comparison']['observations']):
            raw=raw_pair_gate(a,b)
            if raw['pass_all']!=old['agreement']:raise ValueError('raw pair gate does not replay')
            phase={m:dict(all_moments=difference_gate(a[m],b[m]),corrected=corrected_phase_gate(a[m],b[m])) for m in MEASURES}
            rows.append(dict(observation=a['observation'],raw=raw,phase=phase))
        rawpass=all(groups[n]['all_raw_precision_gates'] for n in (left,right)) and all(r['raw']['pass_all'] for r in rows)
        if current['comparison_gate']!=rawpass:raise ValueError('aggregate raw comparison does not replay')
        comparisons[f'{left}_minus_{right}']=dict(observations=rows,all_raw_gates=rawpass,
            all_corrected_phase_gates=all(v['corrected']['pass_all'] for r in rows for v in r['phase'].values()))
    return serial(dict(schema='sbsi-n23-corrected-comparison-v1',groups=groups,comparisons=comparisons,
        development_checks_pass=all(c['all_raw_gates'] and c['all_corrected_phase_gates'] for c in comparisons.values()),
        accepted_sampler=False,heldout_status='pending; development cannot authorize acceptance',
        moment_columns=COLUMNS,target_identity=identity,completion_records=completion,
        timing=dict(recorded_inference_seconds=sum(g['total_recorded_inference_seconds'] for g in groups.values()),
            scheduler_task_elapsed_seconds_by_arm={a['name']:sum(completion[f'{ARRAY_JOB}_{3*i+j}']['elapsed_seconds'] for i in range(8)) for j,a in enumerate(ARMS)},
            shared_phase_job_wall_seconds=completion[PHASE_JOB]['elapsed_seconds'],
            shared_raw_report_job_wall_seconds=completion[REPORT_JOB]['elapsed_seconds'],
            pilot_generation_seconds=None,proposal_geometry_fit_seconds=None,phase_coefficient_fit_seconds=None,
            full_end_to_end_cost_seconds=None,preparation_cost_status='unknown reused one-off preparation; not zero',
            allocation='scheduler records preserve AllocTRES; elapsed seconds are not CPU core-seconds or summed full cost'),
        sources=sources,script_sha256=sha256(__file__),
        qualifications=['All three arms use frozen support-corrected source; no full mixed-measure finite-RNG exactness claim.',
            'Corrected phase agreement is a predeclared diagnostic gate, not proof of exact F32 zero-mean control.',
            'F32 control expectation unknown; F64 reweighting explicitly changes numerical measure and uses intended continuous identity.',
            'Eight population ratio MCSE can miss shared rare modes; no fit, empirical recentering, dropped populations or heldout inputs.',
            'Complete phase diagnostics cannot currently represent extinct populations; explicit zero mass blocks this phase gate, never masquerades as a valid zero control.']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    completion=require_completion()  # Before any v10 outcome is read.
    result=build_summary(args.root,completion)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha256(args.output),development_checks_pass=result['development_checks_pass'],accepted_sampler=False)))


if __name__=='__main__':main()
