#!/usr/bin/env python
"""CPU-only read-only sensitivity on exactly the 24 corrected v10 populations.

The audited original sources remain the import authority. This script adds only
saved-state prior evaluations and paired population math; no likelihood models
are loaded, no random draws occur, and no acceptance gate is modified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

AUDIT_SHA256 = '09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574'
OBSERVATIONS = [58165, 0, 1, 8]
ARMS = [('small', 256, 2026102900), ('large', 1024, 2026103000), ('broad', 1024, 2026103100)]
MEASURES = ('raw_float32_sampled', 'precision_reweighted_float64')
COLUMNS = ['T', 'raw_g1', 'raw_g2', 'corrected_g1', 'corrected_g2']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_audited_path(path, root, source):
    """Pinned source filenames may mention heldout; scientific data may not."""
    path, root, source = map(lambda x:Path(x).resolve(), (path, root, source))
    if source in path.parents:
        return path
    if any('heldout' in part.lower() for part in path.parts):
        raise ValueError('heldout inputs are outside this development-only audit')
    return path


def verify_audit(root):
    root = Path(root).resolve()
    path = root/'v10_corrected_comparison_20261022_v3.json'
    if digest(path) != AUDIT_SHA256:
        raise ValueError('authoritative completed v10 audit changed')
    audit = read(path)
    if audit.get('schema') != 'sbsi-n23-corrected-comparison-v1' or audit.get('development_checks_pass') is not True:
        raise ValueError('complete successful authoritative audit required')
    source = root/'source_support_fix_20261022'
    sources = audit['sources']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('complete audited input map required')
    for name, checksum in sources.items():
        p = safe_audited_path(name, root, source)
        if digest(p) != checksum:
            raise ValueError(f'audited input changed: {p}')
    declaration = read(root/'experiment_v10_corrected_three_way.json')
    if Path(declaration['source']).resolve() != source:
        raise ValueError('original corrected source required')
    manifest = source/'SHA256SUMS'
    if digest(manifest) != declaration['source_manifest_sha256']:
        raise ValueError('original source manifest changed')
    for line in manifest.read_text().splitlines():
        checksum, name = line.split(None,1)
        p=(source/name.lstrip('*')).resolve()
        if source not in p.parents or sources.get(str(p)) != checksum:
            raise ValueError('source manifest is not completely bound by authoritative audit')
    return audit, declaration, source


def close(actual, expected, label, *, atol=1e-11):
    import numpy as np
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=atol,rtol=1e-11):
        raise ValueError(f'baseline/coordinate replay failed: {label}')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def check_summary(actual, expected, label):
    return {key:close(actual[key],expected[key],label+'/'+key) for key in
        ('mean','mcse','covariance','weights','run_weight_ess','log_mean_normalizer','normalizer_relative_mcse')}


def serial(value):
    import numpy as np
    if isinstance(value, np.ndarray):return serial(value.tolist())
    if isinstance(value, np.generic):return serial(value.item())
    if isinstance(value, dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [serial(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        if np.isneginf(value):return None
        raise ValueError('unknown nonfinite audit output')
    return value


def run_audit(args, receipt):
    import numpy as np
    from scipy.special import logsumexp
    audit, declaration, source = verify_audit(args.runs_root)
    root=Path(args.runs_root).resolve()
    if 'sbsi' in sys.modules:
        raise ValueError('fresh interpreter required before importing frozen prior')
    sys.path[:0]=[str(source),str(source/'scripts')]
    import torch  # Required before PyCosmo; no learned model is loaded.
    torch.set_num_threads(2)
    import sbsi
    if Path(sbsi.__file__).resolve().parent.parent != source:
        raise ValueError('scientific imports escaped frozen corrected source')
    import check_galsbi_continuous_prior as factory
    from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
    from sbsi.numerical_native_intensity import numerical_native_log_intensity
    from sbsi.scene_exchange import numerical_over_analytic
    from sbsi.phase_control import FrozenPhaseControl, load_phase_diagnostic, summarize_phase_populations
    from sbsi.particle_diagnostics import combine_populations
    helper_path=Path(__file__).resolve().parents[1]/'sbsi/prior_sensitivity.py'
    spec=importlib.util.spec_from_file_location('_prior_sensitivity',helper_path)
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    receipt['diagnostic_source_sha256']={str(p):digest(p) for p in (Path(__file__).resolve(),helper_path)}
    receipt['audited_original_source_sha256']=audit['sources']
    receipt['authoritative_audit']=dict(path=str(root/'v10_corrected_comparison_20261022_v3.json'),sha256=AUDIT_SHA256)
    # Explicit native input identity, avoiding mutable runtime/font telemetry.
    from galsbi import load
    config=importlib.import_module('galsbi.configs.config_Fischbacher+24_intrinsic')
    prior_inputs=[Path(load.data_path('Fischbacher+24_abc_posterior'))/'abc_posterior.h5',
                  Path(config.maps_remote_dir)/config.templates_int_tables_file_name]
    package_files=set()
    for name in ('galsbi','PyCosmo','cosmo_torrent','sympy2c'):
        module=importlib.import_module(name)
        directory=Path(module.__file__).resolve().parent
        package_files.update(p.resolve() for p in directory.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.git' not in p.parts
            and p.suffix not in ('.pyc','.pyo','.log','.lock'))
    external={str(p):digest(p) for p in sorted(package_files|set(prior_inputs))}
    galaxy,cosmo,_=factory.build_prior()
    # Retain the actual numerical cosmology implementation as well as its
    # package sources; a generated equation cache is a scientific input.
    cache=Path(cosmo._cache_file).resolve()
    wrapper=Path(cosmo._wrapper._folder).resolve()
    cached_files={cache}|{p.resolve() for p in wrapper.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo','.log','.lock')}
    external.update({str(p):digest(p) for p in sorted(cached_files)})
    # The real photometry input must agree with the pre-build configuration.
    if Path(galaxy.p['maps_remote_dir'])/galaxy.p['templates_int_tables_file_name'] != prior_inputs[1]:
        raise ValueError('actual prior photometry path differs from pinned input')
    coordinates=GalSBITargetCoordinates(galaxy,lambda z:cosmo.background.dist_ang_a(a=1/(1+z)))
    receipt['native_input_sha256']=external
    receipt['constructed_native_tables']={name:{key:dict(shape=list(np.asarray(value).shape),
        dtype=str(np.asarray(value).dtype),sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        for key,value in table.items()} for name,table in galaxy.tables.items()}
    receipt['constructed_native_zmax']=galaxy.z_max
    receipt['runtime']=dict(python=sys.version.split()[0],numpy=np.__version__,torch=str(torch.__version__),cpu_threads=2)
    artifact=declaration['phase_control'];coefficient_path=Path(artifact['path'])
    if audit['sources'].get(str(coefficient_path)) != artifact['sha256'] or digest(coefficient_path)!=artifact['sha256']:
        raise ValueError('frozen phase coefficient is not audited')
    frozen=FrozenPhaseControl.load(coefficient_path)
    groups={};payload={};baseline_errors={};replay_records=[]
    for arm,n,seed_base in ARMS:
        ratios=[];values=[];deltas=[];zs=[];saved_scores=[];per_run=[]
        for i in range(8):
            directory=root/f'dev_v10_{arm}_{i}';phase_path=root/f'phase_v10_20261022/{arm}_{i}.json'
            required=[directory/name for name in ('protocol.json','result.json','particles.npz','scenes.npz')]+[phase_path,phase_path.with_suffix('.npz')]
            if any(str(p) not in audit['sources'] for p in required):
                raise ValueError('every complete population/diagnostic must be audited')
            protocol,result=read(directory/'protocol.json'),read(directory/'result.json')
            if (result['status']!='completed' or any(result['extinct']) or protocol['seed']!=seed_base+i
                    or protocol['particles']!=n or protocol['observations']!=OBSERVATIONS
                    or protocol['stages']!=64 or protocol['moves']!=64):
                raise ValueError('unexpected/incomplete original population; never drop or retry')
            phase,a=load_phase_diagnostic(phase_path)
            if (phase['run']!=str(directory) or phase['seed']!=protocol['seed']
                    or phase['retained_particles']!=n or phase['log_normalizer']!=result['log_normalizer']
                    or phase['coefficients']['sha256']!=artifact['sha256']):
                raise ValueError('phase data are not paired to the audited original population')
            frozen.validate_production(protocol['seed'],OBSERVATIONS,phase['target_identity'])
            with np.load(directory/'particles.npz',allow_pickle=False) as stored:
                x=stored['coordinates'].copy();lp_saved=stored['log_prior'].copy()
                ss=stored['scores'].copy();z=stored['log_normalizer'].copy()
            if (x.shape!=(4,n,10) or lp_saved.shape!=(4,n) or ss.shape!=(4,n,2)
                    or not all(np.isfinite(v).all() for v in (x,lp_saved,ss,z))):
                raise ValueError('finite complete original particle arrays required')
            close(z,result['log_normalizer'],'original normalizers',atol=0.)
            close(ss,a['raw_scores'],'paired saved/phase score',atol=1e-6)
            with np.load(directory/'scenes.npz',allow_pickle=False) as scene:
                native={key[len('target_'):]:scene[key].copy() for key in scene.files if key.startswith('target_')}
            encoded=coordinates.encode(native).reshape(4,n,10)
            coordinate_error=close(encoded[...,:9],x[...,:9],'native coordinate replay',atol=1e-8)
            phase_error=close(np.exp(1j*encoded[...,9]).view(float),np.exp(1j*x[...,9]).view(float),'periodic coordinate replay',atol=1e-10)
            xx=x.reshape(-1,10);decoded,jac=coordinates.decode(xx,np.zeros(len(xx),int))
            pterms=[];qterms=[];corrections=[]
            for t in range(len(galaxy.types)):
                typed=decoded|{'type':np.full(len(xx),t)}
                pterms.append(galaxy.log_intensity(typed))
                qterms.append(numerical_native_log_intensity(galaxy,typed))
                corrections.append(numerical_over_analytic(galaxy,typed))
            terms=[np.column_stack(v) for v in (pterms,qterms,corrections)]
            ratio=math.type_marginal_ratio(*terms)
            prior_error=close(ratio['analytic_log_intensity']+jac,lp_saved.ravel(),'analytic prior plus Jacobian',atol=1e-9)
            if prior_error>1e-9:raise ValueError('absolute prior-coordinate replay error exceeds1e-9')
            lr=ratio['log_ratio'].reshape(4,n)
            f=np.concatenate((a['phase_control'][...,None],a['raw_scores'],
                              a['raw_scores']-frozen.coefficients[:,None,:]*a['phase_control'][...,None]),axis=2)
            if f.shape!=(4,n,5):raise ValueError('full raw/control population required')
            ratios.append(lr);values.append(f);deltas.append(a['delta_log_kernel']);zs.append(z);saved_scores.append(ss)
            counts=[]
            for obj,oid in enumerate(OBSERVATIONS):
                sl=slice(obj*n,(obj+1)*n);finite=np.isfinite(lr[obj]);restricted=ratio['restricted_log_ratio'][sl]
                comparable=finite & np.isfinite(restricted)
                counts.append(dict(observation=oid,original_particles=n,zero_numerical_states=int((~finite).sum()),
                    finite_log_ratio_quantiles=np.quantile(lr[obj][finite],[0,.01,.5,.99,1]).tolist() if finite.any() else None,
                    q_only_type_counts=ratio['q_only_type'][sl].sum(0),p_only_type_counts=ratio['p_only_type'][sl].sum(0),
                    states_with_q_only_type=int(ratio['q_only_type'][sl].any(1).sum()),
                    restricted_zero_states=int(ratio['restricted_numerical_zero'][sl].sum()),
                    full_minus_restricted_log_ratio_max=float(np.max(lr[obj][comparable]-restricted[comparable])) if comparable.any() else None))
            per_run.append(dict(replica=i,seed=protocol['seed'],log_normalizer=z,observations=counts))
            replay_records.append(dict(arm=arm,replica=i,coordinate_max_error=coordinate_error,phase_max_error=phase_error,
                prior_max_error=prior_error,helper_shared_max_error=ratio['helper_shared_max_abs_error']))
            payload[f'{arm}_{i}_log_ratio']=lr
            payload[f'{arm}_{i}_analytic_type_log_intensity']=terms[0].reshape(4,n,-1)
            payload[f'{arm}_{i}_numerical_type_log_intensity']=terms[1].reshape(4,n,-1)
            print('PRIOR_SENSITIVITY_POPULATION',arm,i,'rows',len(xx),'prior_replay',prior_error,flush=True)
        phase_replay=summarize_phase_populations(zs,[v[...,0] for v in values],[v[...,1:3] for v in values],deltas,frozen.coefficients)
        rows=[]
        for obj,oid in enumerate(OBSERVATIONS):
            original=audit['groups'][arm]['observations'][obj]
            if original['observation']!=oid:raise ValueError('audited observations reordered')
            baseline_errors[f'{arm}/{oid}/raw_saved']=check_summary(
                combine_populations(np.asarray(zs)[:,obj],np.asarray(saved_scores)[:,obj].mean(1)),original['score'],'audited saved raw')
            row=dict(observation=oid)
            for measure in MEASURES:
                baseline_errors[f'{arm}/{oid}/{measure}']=check_summary(phase_replay[obj][measure],original[measure],'audited phase baseline')
                mw=[np.zeros(n) for _ in range(8)] if measure==MEASURES[0] else [v[obj] for v in deltas]
                sensitivity=math.paired_sensitivity(np.asarray(zs)[:,obj],[v[obj] for v in values],mw,[r[obj] for r in ratios])
                check_summary(sensitivity['analytic']['summary'],original[measure],'paired baseline consistency')
                row[measure]=sensitivity
            rows.append(row)
        groups[arm]=dict(original_populations=8,particles_per_population=n,per_population=per_run,observations=rows)
    # Nothing becomes interpretable unless all original inputs still match.
    verify_audit(root)
    for path,checksum in external.items():
        if digest(path)!=checksum:raise ValueError('native scientific input changed during audit')
    for path,checksum in receipt['diagnostic_source_sha256'].items():
        if digest(path)!=checksum:raise ValueError('diagnostic source changed during audit')
    np.savez_compressed(args.output/'prior_ratios.npz',**payload)
    receipt.update(status='passed',groups=groups,baseline_replay_errors=baseline_errors,prior_coordinate_replays=replay_records,
        arrays_sha256=digest(args.output/'prior_ratios.npz'),moment_columns=COLUMNS,
        original_populations=24,original_target_particles=sum(4*n*8 for _,n,_ in ARMS))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('CPU-only scheduled audit required')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    receipt=dict(schema='corrected-v10-prior-sensitivity-v1',status='running',cpu_only=True,
        likelihood_models_loaded=False,sampler_or_gate_changed=False,slurm_job_id=os.environ['SLURM_JOB_ID'],
        limitations=['Sensitivity is conditional on represented analytic-support states; numerical-only coordinate support and missing modes are not bounded.',
            'Full numerical type intensities include q-only type mass at represented x; saved placeholder target type is never used for the ratio.',
            'Intensity-mass ratios are not normalized model evidence ratios; omitted global prior normalization cancels from posterior means.',
            'Paired delta-method MCSE uses eight whole original populations and both ratio denominators, conditional on frozen pilot coefficients.',
            'Numerical continuous density omits clamp atoms and failed inverse-table draws; it is not literal finite-RNG exactness.',
            'F64 phase identity also holds under phase-independent numerical-prior tilting on analytic support; sampled F32 correction remains qualified.',
            'No calibration offset, acceptance gate, heldout estimate or current sampler target is changed.'])
    try:
        run_audit(args,receipt)
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(serial(receipt),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
