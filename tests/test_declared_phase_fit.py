"""Declared phase fitting must not consume evaluated or changed populations."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sbsi.phase_control import FrozenPhaseControl, digest

SPEC = importlib.util.spec_from_file_location('declared_phase_fit',
    Path(__file__).resolve().parents[1]/'scripts/fit_declared_particle_phase_control.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path/'source'
    script = source/'scripts/fit_declared_particle_phase_control.py'
    script.parent.mkdir(parents=True)
    script.write_text('frozen fixture')
    for name in m.PROTOCOL_SOURCES | m.DIAGNOSTIC_SOURCES:
        file = source/name
        file.parent.mkdir(exist_ok=True)
        file.write_text('fixture '+name)
    manifest = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    (source/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name, h in sorted(manifest.items())))
    monkeypatch.setattr(m, '__file__', str(script))
    monkeypatch.setattr(m.phase_module, '__file__', str(source/'sbsi/phase_control.py'))
    root = tmp_path/'study'
    panel = tmp_path/'panel.json'
    groups = [list(range(20+4*i, 24+4*i)) for i in range(5)]
    write(panel, {'bands': [{'observations': g} for g in groups]})
    declaration = dict(schema='sbsi-heldout-corrected-declaration-v1', source=str(source),
        source_manifest_sha256=digest(source/'SHA256SUMS'), output=str(root),
        panel=str(panel), panel_sha256=digest(panel), groups=groups, development_ids=[0, 1, 8, 58165],
        pilot=dict(replicas=4, particles=4, stages=2, moves=2, seed_base=1000,
                   blocks=['physical'], scale=1., power=2.),
        candidate_settings=dict(replicas=8, seed_base=2000),
        reference_settings=dict(replicas=8, seed_base=3000),
        shear=[0., 0.], density_batch_size=4096,
        reference_protocol_sha256='a'*64, observation_sha256='b'*64)
    reports, protocols, paths = [], [], []
    for i in range(4):
        run = root/'group_0'/f'pilot_{i}'
        impl = {str(source/name): manifest[name] for name in m.PROTOCOL_SOURCES}
        protocol = dict(seed=1000+i, observations=groups[0], particles=4, stages=2, moves=2,
            blocks=['physical'], scale=1., betas=[0., .25, 1.], fixed_shear=[0., 0.],
            density_batch_size=4096, reference_protocol_sha256='a'*64, observation_sha256='b'*64,
            implementation_sha256=impl, selection='none in fixed-y posterior; no population shear fit',
            proposals=[dict(component_family='gaussian', faint_magnitude_shift=5.,
                means=[[0.]*9]*2, scales=[[1.]*9]*2,
                weights=[.5, .5], phase='independent uniform periodic') for _ in range(4)])
        write(run/'protocol.json', protocol)
        write(run/'result.json', {'fixture': True})
        # The loader verifies the original artifacts; full run validity is checked
        # by the phase diagnostic producer before numerical_checks_pass is set.
        np.savez(run/'particles.npz', fixture=np.arange(2))
        path = root/'group_0'/f'pilot_{i}_phase.json'
        control = np.broadcast_to(np.arange(4, dtype=float), (4, 4)).copy()+i
        scores = np.stack((2*control+3., -control+2.), axis=2)
        np.savez(path.with_suffix('.npz'), raw_scores=scores, phase_control=control,
                 delta_log_kernel=np.zeros_like(control))
        identity = {k: protocol[k] for k in ('observations', 'fixed_shear', 'reference_protocol_sha256',
                                           'observation_sha256', 'density_batch_size', 'selection')}
        identity['native_source_sha256'] = {Path(name).name: manifest[name] for name in manifest
                                            if Path(name).name in m.NATIVE_SOURCES}
        report = dict(schema='sbsi-phase-diagnostic-v1', numerical_checks_pass=True,
            arrays_sha256=digest(path.with_suffix('.npz')), stride=1, retained_particles=4,
            run=str(run), seed=1000+i, observations=groups[0], target_identity=identity,
            implementation_sha256={str(source/name): manifest[name] for name in m.DIAGNOSTIC_SOURCES},
            source_hashes={str(run/name): digest(run/name) for name in
                           ('protocol.json', 'result.json', 'particles.npz')})
        write(path, report)
        reports.append(report); protocols.append(protocol); paths.append(path)
    declaration_path = root/'declaration.json'
    declaration['pilot_proposal_metadata'] = [copy.deepcopy(protocols[0]['proposals']) for _ in groups]
    write(declaration_path, declaration)
    return declaration_path, declaration, reports, protocols, paths


def test_fits_declared_independent_pilots_with_frozen_provenance(inputs):
    path, d, _, _, _ = inputs
    output = Path(d['output'])/'group_0/phase_coefficients.json'
    receipt = m.fit_declared(path, 0, output)
    frozen = FrozenPhaseControl.load(output)
    np.testing.assert_allclose(frozen.coefficients, np.tile([2., -1.], (4, 1)))
    assert receipt['sha256'] == digest(output)
    assert frozen.metadata['declaration']['sha256'] == digest(path)
    assert frozen.metadata['production_control_centering'] is False
    with pytest.raises(ValueError, match='fresh declared'):
        m.fit_declared(path, 0, output)


@pytest.mark.parametrize('change', ['seed', 'particles', 'moves', 'betas', 'geometry', 'proposal', 'scales', 'order', 'stride'])
def test_changed_pilot_recipe_or_diagnostic_rejected(inputs, change):
    _, d, r, p, paths = inputs
    if change in ('seed', 'particles', 'moves'):
        p[0][change] += 1
    elif change == 'betas':
        p[0]['betas'][1] = .3
    elif change == 'geometry':
        p[0]['physical_geometry'] = {'unexpected': True}
    elif change == 'proposal':
        p[0]['proposals'][0]['component_family'] = 'student_t'
    elif change == 'scales':
        p[0]['proposals'][0]['scales'][0][0] = 2.
    elif change == 'order':
        paths.reverse()
    else:
        r[0]['stride'] = 2
    with pytest.raises(ValueError):
        m.validate_pilot_records(d, 0, r, p, paths)


def test_global_seed_overlap_and_development_overlap_rejected(inputs):
    _, d, r, p, paths = inputs
    changed = copy.deepcopy(d)
    changed['reference_settings']['seed_base'] = d['pilot']['seed_base']+10
    with pytest.raises(ValueError, match='seeds overlap'):
        m.validate_pilot_records(changed, 0, r, p, paths)
    changed = copy.deepcopy(d)
    changed['groups'][4][0] = 58165
    with pytest.raises(ValueError, match='held-out groups'):
        m.validate_pilot_records(changed, 0, r, p, paths)


@pytest.mark.parametrize('change', ['diagnostic', 'source', 'panel', 'groups', 'production', 'original'])
def test_changed_input_or_started_production_prevents_fit(inputs, change):
    path, d, reports, _, paths = inputs
    output = Path(d['output'])/'group_0/phase_coefficients.json'
    if change == 'diagnostic':
        reports[0]['numerical_checks_pass'] = False
        write(paths[0], reports[0])
    elif change == 'source':
        (Path(d['source'])/'scripts/fit_declared_particle_phase_control.py').write_text('changed')
    elif change == 'panel':
        write(Path(d['panel']), {'changed': True})
    elif change == 'groups':
        d['groups'].reverse(); write(path, d)
    elif change == 'production':
        (output.parent/'candidate_0').mkdir()
    else:
        write(output.parent/'pilot_0/protocol.json', {'changed': True})
    with pytest.raises(ValueError):
        m.fit_declared(path, 0, output)
    assert not output.exists()


def test_manifest_rejects_duplicate_or_outside_paths(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    data = source/'data'; data.write_text('ok')
    manifest = source/'SHA256SUMS'
    entry = f'{digest(data)}  data\n'
    manifest.write_text(entry*2)
    with pytest.raises(ValueError, match='duplicate'):
        m.verified_manifest(source, digest(manifest))
    manifest.write_text(f'{digest(data)}  ../source/../outside\n')
    with pytest.raises(ValueError, match='invalid'):
        m.verified_manifest(source, digest(manifest))


@pytest.mark.parametrize('change', ['provenance', 'protocol_sources', 'diagnostic_sources', 'identity', 'native_identity', 'module'])
def test_provenance_cannot_substitute_another_run_or_implementation(inputs, monkeypatch, change):
    path, d, reports, protocols, paths = inputs
    output = Path(d['output'])/'group_0/phase_coefficients.json'
    if change == 'provenance':
        reports[0]['source_hashes'] = reports[1]['source_hashes']
    elif change == 'protocol_sources':
        protocols[0]['implementation_sha256'] = {}
        original = Path(d['output'])/'group_0/pilot_0/protocol.json'
        write(original, protocols[0])
        reports[0]['source_hashes'][str(original)] = digest(original)
    elif change == 'diagnostic_sources':
        reports[0]['implementation_sha256'] = {}
    elif change == 'identity':
        for report in reports:
            report['target_identity']['density_batch_size'] = 128
    elif change == 'native_identity':
        for report in reports:
            report['target_identity']['native_source_sha256'] = {}
    else:
        monkeypatch.setattr(m.phase_module, '__file__', '/different/sbsi/phase_control.py')
    for diagnostic, report in zip(paths, reports):
        write(diagnostic, report)
    with pytest.raises(ValueError):
        m.fit_declared(path, 0, output)
    assert not output.exists()
