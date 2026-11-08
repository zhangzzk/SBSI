"""Exercise the frozen reporting adapter on completed development artifacts only."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021')
SOURCE = ROOT/'source_corrected_heldout_v3_20261022'
EXPECTED = '09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

spec = importlib.util.spec_from_file_location('frozen_heldout_runner', SOURCE/'scripts/run_corrected_particle_heldout.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert Path(m.__file__).resolve().parent.parent == SOURCE
sources = {}
def read(path):
    path = Path(path).resolve()
    assert not any(part.startswith('heldout_') for part in path.relative_to(ROOT).parts), path
    sources[str(path)] = digest(path)
    return json.loads(path.read_text())

receipt_path = ROOT/'v10_corrected_comparison_20261022_v3.json'
assert digest(receipt_path) == EXPECTED
expected = read(receipt_path)
assert expected['development_checks_pass'] and not expected['accepted_sampler']
raw = {arm:read(ROOT/f'dev_v10_{arm}_report.json') for arm in ('small','large','broad')}
phase = {arm:read(ROOT/'phase_v10_20261022'/f'{arm}_controls.json') for arm in raw}
checks = [0]
def same(actual, prior):
    if isinstance(prior, dict):
        assert set(actual) == set(prior)
        for key in prior: same(actual[key],prior[key])
    elif isinstance(prior,list):
        assert len(actual)==len(prior)
        for a,b in zip(actual,prior):same(a,b)
    elif isinstance(prior,bool) or prior is None or isinstance(prior,str):
        assert actual==prior
    else:
        np.testing.assert_allclose(actual,prior,rtol=1e-12,atol=1e-12)
    checks[0]+=1

comparisons={}
for a,b in [('small','large'),('small','broad'),('large','broad')]:
    key=a+'_minus_'+b
    comparison=m.compare_group(raw[a],raw[b],phase[a],phase[b])
    prior=expected['comparisons'][key]
    assert comparison['accuracy_pass']==(prior['all_raw_gates'] and prior['all_corrected_phase_gates'])
    for now,old in zip(comparison['observations'],prior['observations']):
        assert now['accuracy_pass']
        same({k:now[k] for k in old},old)
    comparisons[key]=dict(accuracy_pass=comparison['accuracy_pass'],observations=len(comparison['observations']))

# Temporary path rebinding tests report-format validation, not a scientific
# declaration or rerun. Every copied diagnostic retains its original bytes.
validated_reports=[]
with tempfile.TemporaryDirectory(prefix='sbsi-report-contract-') as temporary:
    directory=Path(temporary)
    for arm in raw:
        a,b=copy.deepcopy(raw[arm]),copy.deepcopy(phase[arm])
        b['sources']={}
        for i in range(8):
            original=ROOT/'phase_v10_20261022'/f'{arm}_{i}.json'
            read(original)
            target=directory/f'candidate_{i}_phase.json'
            shutil.copy2(original,target)
            b['sources'][str(target)]=digest(target)
        coefficient=Path(b['coefficients']['path']); read(coefficient)
        shutil.copy2(coefficient,directory/'phase_coefficients.json')
        base=a['seeds'][0]
        assert a['seeds']==list(range(base,base+8))
        d=dict(candidate_settings={**{k:a[k] for k in ('particles','stages','moves','replicas')},'seed_base':base},
            groups=[[o['observation'] for o in a['observations']]],shear=a['centre'],
            reference_protocol_sha256=a['reference_protocol_sha256'],observation_sha256=a['observation_sha256'],
            density_batch_size=a['density_batch_size'])
        m.validate_report(d,0,'candidate',a,b,directory)
        validated_reports.append(arm)
        broken=copy.deepcopy(b); broken['observations'][0][m.MEASURES[1]]['independent_populations']=7
        try: m.validate_report(d,0,'candidate',a,broken,directory)
        except ValueError: pass
        else: raise AssertionError('incomplete population denominator accepted')

result=dict(schema='sbsi-heldout-report-development-integration-v1',comparisons=comparisons,
    validated_report_arms=validated_reports,recursive_comparison_checks=checks[0],
    report_replay_tolerance=dict(relative=1e-12,absolute=1e-12),
    tolerance_scope='Report arithmetic replay only; no scientific gate threshold changes.',
    real_heldout_artifacts_read=False,new_model_evaluations=False,accepted_sampler=False,
    sources=sources,runner_source_sha256=digest(m.__file__),script_sha256=digest(__file__),
    scope='Frozen runner integration on already completed development, not independent scientific evidence.')
output=ROOT/'heldout_reporting_development_integration_20261022.json'
with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(output=str(output),sha256=digest(output),checks=checks[0],comparisons=comparisons)))
