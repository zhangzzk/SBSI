"""Preflight cannot bypass a failed development decision or change the recipe."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('corrected_preflight',
    Path(__file__).resolve().parents[1]/'scripts/prepare_corrected_particle_heldout.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    evidence = tmp_path/'evidence'; evidence.write_text('unchanged scientific inputs')
    result = dict(schema='sbsi-n23-corrected-comparison-v1', development_checks_pass=True,
        accepted_sampler=False, comparisons={name: dict(all_raw_gates=True, all_corrected_phase_gates=True)
            for name in ('small_minus_large', 'small_minus_broad', 'large_minus_broad')},
        sources={str(evidence): m.digest(evidence)})
    development = tmp_path/'development.json'; development.write_text(json.dumps(result))
    panel = tmp_path/'panel.json'
    panel.write_text(json.dumps(dict(bands=[dict(observations=list(range(20+4*i, 24+4*i)))
                                            for i in range(5)], excluded_development=[0, 1, 8, 58165])))
    monkeypatch.setattr(m, 'PANEL_SHA256', m.digest(panel))
    return development, result, panel, evidence


def test_valid_complete_decision_and_frozen_panel(inputs):
    dev, result, panel, _ = inputs
    got, _, groups = m.validated_inputs(dev, m.digest(dev), panel)
    assert got == result
    assert groups[0] == [20, 21, 22, 23]


@pytest.mark.parametrize('case', ['failed', 'partial', 'raw', 'phase', 'accepted', 'changed_evidence', 'changed_receipt'])
def test_failed_or_changed_development_never_opens_panel(inputs, case):
    dev, result, panel, evidence = inputs
    old_hash = m.digest(dev)
    if case == 'failed': result['development_checks_pass'] = False
    elif case == 'partial': del result['comparisons']['large_minus_broad']
    elif case == 'raw': result['comparisons']['small_minus_broad']['all_raw_gates'] = False
    elif case == 'phase': result['comparisons']['small_minus_broad']['all_corrected_phase_gates'] = False
    elif case == 'accepted': result['accepted_sampler'] = True
    elif case == 'changed_evidence': evidence.write_text('changed')
    else: result['unexpected'] = True
    dev.write_text(json.dumps(result))
    panel.unlink()  # A premature heldout read would raise FileNotFoundError instead.
    with pytest.raises(ValueError):
        m.validated_inputs(dev, old_hash if case == 'changed_receipt' else m.digest(dev), panel)


@pytest.mark.parametrize('case', ['changed', 'duplicates', 'development', 'missing'])
def test_panel_integrity_and_disjointness(inputs, monkeypatch, case):
    dev, _, panel, _ = inputs
    p = json.loads(panel.read_text())
    if case == 'changed': p['extra'] = 'changed'
    elif case == 'duplicates': p['bands'][0]['observations'][0] = 21
    elif case == 'development': p['bands'][0]['observations'][0] = 58165
    else: p['bands'].pop()
    panel.write_text(json.dumps(p))
    if case != 'changed': monkeypatch.setattr(m, 'PANEL_SHA256', m.digest(panel))
    with pytest.raises(ValueError): m.validated_inputs(dev, m.digest(dev), panel)


def test_full_measured_laws_reproduce_original_constructor():
    from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
    y = {20: np.array([.1, -.2, 5., 100.]), 21: np.array([0., .1, 6., 20.])}
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224)
    parameters = dict(sersic_n_min=.3, sersic_n_max=6., ell_beta_emax=.95)
    laws = m.proposal_laws(y, [[20, 21]], conditions, parameters)
    for family, key in [('gaussian', 'pilot_proposal_metadata'), ('student_t', 'reference_proposal_metadata')]:
        for i, oid in enumerate((20, 21)):
            expected = MeasurementParticleProposal.from_measurement(y[oid], conditions, parameters,
                [0., 0.], component_family=family, student_df=5.).metadata
            assert laws[key][0][i] == expected
            assert expected['uses_generating_truth'] is False
    assert laws['pilot_proposal_metadata'][0][0]['means'] == laws['reference_proposal_metadata'][0][0]['means']


def test_existing_scientific_sources_cannot_change_or_disappear():
    previous = {name: 'a'*64 for name in ('sbsi/galsbi_prior.py', 'configs/models_v3_9_like.json',
        'scripts/check_galsbi_particles.py', 'scripts/check_galsbi_continuous_prior.py',
        'scripts/fit_particle_gradient_geometry.py', 'scripts/fit_particle_initial_proposal.py',
        'scripts/check_particle_phase_control.py', 'scripts/report_particle_phase_control.py',
        'scripts/report_galsbi_particles.py')}
    new = dict(previous, **{'scripts/new_report.py': 'b'*64})
    m.same_scientific_source(new, previous)
    new['sbsi/galsbi_prior.py'] = 'b'*64
    with pytest.raises(ValueError): m.same_scientific_source(new, previous)
    del new['sbsi/galsbi_prior.py']
    with pytest.raises(ValueError): m.same_scientific_source(new, previous)


def test_package_pin_omits_only_bytecode_and_vcs(tmp_path):
    for name in ('__init__.py', 'data/table.npy', 'engine.so', '__pycache__/module.pyc', '.git/index', 'build.log', 'build.lock'):
        path = tmp_path/name; path.parent.mkdir(exist_ok=True); path.write_text(name)
    module = SimpleNamespace(__file__=str(tmp_path/'__init__.py'))
    assert {p.relative_to(tmp_path).as_posix() for p in m.package_files(module)} == {
        '__init__.py', 'data/table.npy', 'engine.so'}


def test_numeric_prior_parameters_keep_shape_dtype_and_values():
    result = m.parameter_record({'table': np.arange(6, dtype=np.float64).reshape(2, 3),
                                 'scalar': np.int64(4), 'type': np.float64})
    assert result['table'] == dict(dtype='float64', shape=[2, 3], values=[[0., 1., 2.], [3., 4., 5.]])
    assert result['scalar'] == 4
    assert result['type']['python_binding'] == 'numpy.float64'


def test_config_infinities_are_metadata_not_changed_prior_values():
    p = dict(n_gal_max_blue=np.inf, n_gal_max_red=np.float64(np.inf),
             max_memlimit_gal_catalog=np.inf, n_phot_sum_gal_max=np.inf,
             array=np.array([np.inf, -np.inf, np.nan, 1.]))
    encoded = m.parameter_record(p)
    json.dumps(encoded, allow_nan=False)
    assert all(encoded[key] == {'nonfinite_float': '+inf'} for key in p if key != 'array')
    assert encoded['array']['values'] == [dict(nonfinite_float=v) for v in ('+inf', '-inf', 'nan')]+[1.]
    assert np.isinf(p['n_gal_max_blue'])
    np.testing.assert_equal(p['array'], [np.inf, -np.inf, np.nan, 1.])


@pytest.mark.parametrize('mode,flags,readonly', [
    ('r', os.O_RDONLY, True), ('rb', os.O_RDONLY, True),
    ('w', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, False),
    ('r+', os.O_RDWR, False), (None, os.O_RDONLY | os.O_CREAT, False),
    ('r', None, True), ('a', None, False), ('r+', None, False)])
def test_audit_distinguishes_actual_input_reads_from_write_outputs(tmp_path, monkeypatch, mode, flags, readonly):
    monkeypatch.chdir(tmp_path)
    path, actual = m.audited_open(('relative.pkl', mode, flags))
    assert path == tmp_path/'relative.pkl'
    assert actual is readonly
    assert m.audited_open((3, 'rb', os.O_RDONLY)) == (None, False)


def test_imported_modules_must_resolve_inside_declared_source(tmp_path):
    source = tmp_path/'source'
    good = SimpleNamespace(__file__=str(source/'sbsi/__init__.py'))
    m.require_source_modules(source, [good])
    wrong = SimpleNamespace(__file__=str(tmp_path/'editable/sbsi/__init__.py'))
    with pytest.raises(ValueError, match='actual imported'):
        m.require_source_modules(source, [good, wrong])


@pytest.mark.parametrize('path', ['/proc/cpuinfo', '/proc/stat', '/sys/devices/system/cpu/online', '/dev/urandom'])
def test_dynamic_kernel_inputs_are_not_immutable_scientific_files(path):
    assert m.virtual_system_path(path)


@pytest.mark.parametrize('path', ['/project/data/prior.h5', '/home/user/_cache/PyCosmo/table.pkl',
                                  '/opt/package/engine.so', '/project/proc/cpuinfo', '/processor/table'])
def test_real_prior_package_and_cache_files_remain_immutable_inputs(path):
    assert not m.virtual_system_path(path)


def test_virtual_input_classification_resolves_symlinks(tmp_path):
    link = tmp_path/'cpu'
    link.symlink_to('/proc/cpuinfo')
    assert m.virtual_system_path(link)


def test_font_inventory_exclusion_keeps_scientific_caches_and_configuration():
    assert m.generated_font_cache('/home/user/.cache/matplotlib/fontlist-v390.json')
    for path in ['/home/user/_cache/PyCosmo/parameters.pkl',
                 '/home/user/.cache/cosmo-torrent/abc_posterior.h5',
                 '/home/user/.astropy/config/astropy.cfg',
                 '/project/fontlist-v390.json']:
        assert not m.generated_font_cache(path)
        assert not m.virtual_system_path(path)
