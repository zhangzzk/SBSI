"""Content identity for the installed inputs consumed by the numerical factory.

This pins the declared GalSBI construction inputs, not every system library.
Paths are separate audit evidence; compatibility compares content and settings.
No random draws, prior reconstruction, fitting or posterior reads occur here.
"""
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

INPUT_SCHEMA = 'sbsi-numerical-prior-inputs-v1'
FACTORY_ARGUMENTS = dict(model_index=0, grid_size=8193)
PACKAGES = ('galsbi', 'PyCosmo', 'cosmo_torrent', 'cosmic_toolbox')
CONFIG_MODULE = 'galsbi.configs.config_Fischbacher+24_intrinsic'
REQUIRED_INPUTS = ('data/abc_posterior.h5', 'data/photometry_integrals',
    'packages/galsbi/configs/config_Fischbacher+24_intrinsic.py',
    *(f'packages/{name}/__init__.py' for name in PACKAGES))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_prior_inputs(value):
    """Require complete declared input identities, independent of file location."""
    if (not isinstance(value, dict) or set(value) != {'schema', 'factory_arguments', 'files_sha256', 'runtime_versions'}
            or value.get('schema') != INPUT_SCHEMA
            or value.get('factory_arguments') != FACTORY_ARGUMENTS):
        raise ValueError('explicit supported numerical prior construction inputs required')
    files = value['files_sha256']
    if not isinstance(files, dict) or any(name not in files for name in REQUIRED_INPUTS):
        raise ValueError('numerical prior input content identities incomplete')
    for name, checksum in files.items():
        if (not isinstance(name, str) or not name.startswith(('packages/', 'data/'))
                or Path(name).is_absolute() or '..' in Path(name).parts
                or not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('valid logical numerical prior input names and SHA256 hashes required')
    versions = value['runtime_versions']
    if (not isinstance(versions, dict) or set(versions) != {'numpy', 'scipy'}
            or any(not isinstance(v, str) or not v for v in versions.values())):
        raise ValueError('numerical prior NumPy/SciPy versions required')
    return deepcopy(value)


def verify_prior_input_files(identity, paths):
    """Check actual bytes without making paths part of compatibility identity."""
    identity = validate_prior_inputs(identity)
    if not isinstance(paths, dict) or set(paths) != set(identity['files_sha256']):
        raise ValueError('aligned actual numerical prior input paths required')
    for name, expected in identity['files_sha256'].items():
        if digest(paths[name]) != expected:
            raise ValueError('numerical prior input changed: '+name)


def _installed_inputs():
    """Resolve the same retained F24 row and table selection as build_prior."""
    import numpy
    import scipy
    paths = {}
    for name in PACKAGES:
        module = importlib.import_module(name)
        root = Path(module.__file__).resolve().parent
        for path in sorted(root.rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts
                    and path.suffix not in ('.pyc', '.pyo', '.log', '.lock')):
                paths[f'packages/{name}/{path.relative_to(root).as_posix()}'] = str(path.resolve())
    from galsbi import load
    abc = Path(load.data_path('Fischbacher+24_abc_posterior'))/'abc_posterior.h5'
    # Pin before the metadata read as well as before/after the actual factory.
    abc_checksum = digest(abc)
    config = importlib.import_module(CONFIG_MODULE)
    parameters = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    row = load.load_abc_posterior('Fischbacher+24')[FACTORY_ARGUMENTS['model_index']]
    parameters.update({k: row[k].item() for k in row.dtype.names})
    table = Path(parameters['maps_remote_dir'])/parameters['templates_int_tables_file_name']
    local = Path.cwd()/table.name if parameters['copy_template_int_tables_to_cwd'] else None
    actual = local if local is not None and local.exists() else table
    paths.update({'data/abc_posterior.h5': str(abc.resolve()),
                  'data/photometry_integrals': str(actual.resolve())})
    identity = dict(schema=INPUT_SCHEMA, factory_arguments=FACTORY_ARGUMENTS.copy(),
        files_sha256={name: digest(path) for name, path in paths.items()},
        runtime_versions=dict(numpy=numpy.__version__, scipy=scipy.__version__))
    if identity['files_sha256']['data/abc_posterior.h5'] != abc_checksum:
        raise ValueError('ABC posterior changed during input discovery')
    return validate_prior_inputs(identity), paths, local


def build_bound_numerical_prior(factory):
    """Build once from fixed arguments; verify source/data bytes around the call.

    A preexisting local photometry copy is pinned as the actual input. If GalSBI
    creates the copy, its contents must match the input pinned before building.
    The surrounding run still needs the complete frozen-source declaration.
    """
    identity, paths, local = _installed_inputs()
    prior, cosmo, config = factory(**identity['factory_arguments'])
    if local is not None:
        paths['data/photometry_integrals'] = str(local.resolve())
    verify_prior_input_files(identity, paths)
    return prior, cosmo, config, identity, paths
