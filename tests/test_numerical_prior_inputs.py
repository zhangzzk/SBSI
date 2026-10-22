"""Bound actual factory inputs once; compare contents independently of location."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

import sbsi.numerical_prior_inputs as inputs


def input_identity():
    return dict(schema=inputs.INPUT_SCHEMA, factory_arguments=inputs.FACTORY_ARGUMENTS.copy(),
        files_sha256={name:'a'*64 for name in inputs.REQUIRED_INPUTS},
        runtime_versions=dict(numpy='test-numpy', scipy='test-scipy'))


def file_fixture(root):
    identity, paths = input_identity(), {}
    for name in identity['files_sha256']:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        paths[name] = str(path)
        identity['files_sha256'][name] = inputs.digest(path)
    return identity, paths


@pytest.mark.parametrize('missing', inputs.REQUIRED_INPUTS)
def test_each_actual_input_category_is_required(missing):
    identity = input_identity(); identity['files_sha256'].pop(missing)
    with pytest.raises(ValueError, match='incomplete'): inputs.validate_prior_inputs(identity)


@pytest.mark.parametrize('change', ['arguments', 'version', 'hash', 'path'])
def test_incomplete_or_malformed_input_identity_rejected(change):
    identity = input_identity()
    if change == 'arguments': identity['factory_arguments']['grid_size'] = 4097
    if change == 'version': identity['runtime_versions'].pop('scipy')
    if change == 'hash': identity['files_sha256']['data/abc_posterior.h5'] = 'wrong'
    if change == 'path': identity['files_sha256']['/absolute/path'] = 'a'*64
    with pytest.raises(ValueError): inputs.validate_prior_inputs(identity)


def test_content_identity_is_portable_and_actual_changed_bytes_fail(tmp_path):
    a, paths_a = file_fixture(tmp_path/'one'); b, paths_b = file_fixture(tmp_path/'two')
    assert a == b and paths_a != paths_b
    inputs.verify_prior_input_files(a, paths_a); inputs.verify_prior_input_files(a, paths_b)
    Path(paths_b['data/abc_posterior.h5']).write_text('different prior parameters')
    with pytest.raises(ValueError, match='changed'): inputs.verify_prior_input_files(a, paths_b)


@pytest.mark.parametrize('change', [None, 'abc', 'local_copy', 'ufig_seed'])
def test_factory_built_once_and_input_changes_are_not_relabelled(tmp_path, monkeypatch, change):
    identity, paths = file_fixture(tmp_path/'inputs')
    original = deepcopy(identity)
    local = tmp_path/'local-table' if change == 'local_copy' else None
    monkeypatch.setattr(inputs, '_installed_inputs', lambda: (identity, paths, local))
    calls = []
    objects = (object(), object(), object())
    def factory(**kwargs):
        calls.append(kwargs)
        if change == 'abc': Path(paths['data/abc_posterior.h5']).write_text('changed')
        if change == 'ufig_seed': Path(paths['packages/ufig/config/common.py']).write_text('seed_ngal = 501')
        if change == 'local_copy': local.write_text('wrong cached table')
        return objects
    if change is not None:
        with pytest.raises(ValueError, match='changed'): inputs.build_bound_numerical_prior(factory)
    else:
        result = inputs.build_bound_numerical_prior(factory)
        assert result[:3] == objects and result[3] == original and result[4] == paths
    assert calls == [inputs.FACTORY_ARGUMENTS]
    assert identity == original


@pytest.mark.parametrize('copy_mode', ['remote', 'preexisting', 'create'])
def test_discovery_binds_actual_cached_or_remote_photometry_table(tmp_path, monkeypatch, copy_mode):
    modules = {}
    for name in inputs.PACKAGES:
        root = tmp_path/name; root.mkdir(); (root/'__init__.py').write_text(name)
        mod = ModuleType(name); mod.__file__ = str(root/'__init__.py'); modules[name] = mod
    for name, logical in inputs.CONFIG_SOURCES.items():
        path = tmp_path/Path(logical).relative_to('packages')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('seed_ngal = 500')
        mod = ModuleType(name); mod.__file__ = str(path); modules[name] = mod
    config_path = tmp_path/'galsbi/configs/config_Fischbacher+24_intrinsic.py'
    config_path.parent.mkdir(); config_path.write_text('fixed toy config')
    remote = tmp_path/'remote'; remote.mkdir(); (remote/'table.h5').write_text('remote table')
    abc = tmp_path/'abc'; abc.mkdir(); (abc/'abc_posterior.h5').write_text('fixed posterior')
    work = tmp_path/'work'; work.mkdir(); monkeypatch.chdir(work)
    if copy_mode == 'preexisting': (work/'table.h5').write_text('different existing local table')
    config = SimpleNamespace(__file__=str(config_path), maps_remote_dir=str(remote),
        templates_int_tables_file_name='table.h5', copy_template_int_tables_to_cwd=copy_mode != 'remote')
    modules[inputs.CONFIG_MODULE] = config
    modules['galsbi'].load = SimpleNamespace(data_path=lambda name: str(abc),
        load_abc_posterior=lambda name: np.zeros(1, dtype=[('toy_parameter', float)]))
    monkeypatch.setitem(sys.modules, 'galsbi', modules['galsbi'])
    monkeypatch.setattr(inputs.importlib, 'import_module', lambda name: modules[name])
    identity, paths, local = inputs._installed_inputs()
    actual = work/'table.h5' if copy_mode == 'preexisting' else remote/'table.h5'
    assert paths['data/photometry_integrals'] == str(actual)
    assert identity['files_sha256']['data/photometry_integrals'] == inputs.digest(actual)
    assert local == (None if copy_mode == 'remote' else work/'table.h5')
    for name, logical in inputs.CONFIG_SOURCES.items():
        assert paths[logical] == modules[name].__file__
        assert identity['files_sha256'][logical] == inputs.digest(modules[name].__file__)
    inputs.verify_prior_input_files(identity, paths)
    if copy_mode == 'create':
        monkeypatch.setattr(inputs, '_installed_inputs', lambda: (identity, paths, local))
        def factory(**kwargs):
            local.write_bytes(actual.read_bytes())
            return 1, 2, 3
        result = inputs.build_bound_numerical_prior(factory)
        assert result[4]['data/photometry_integrals'] == str(local)
        assert result[3]['files_sha256']['data/photometry_integrals'] == inputs.digest(actual)
