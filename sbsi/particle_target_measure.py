"""Explicit continuous numerical target-prior identity for new particle runs.

Legacy analytic protocols retain their exact fields. A numerical target must
bind the density implementation and cannot borrow analytic pilot/control fits.
This is a measure identity, not a posterior-convergence assertion.
"""
from pathlib import Path

from .numerical_prior_inputs import validate_prior_inputs

ANALYTIC_ENDPOINT = 'q_environment(E) * p_analytic(x) * K(y|x,E)'
NUMERICAL_ENDPOINT = 'q_environment(E) * q_numerical(x) * K(y|x,E)'
NUMERICAL_BRIDGE = 'q_environment(E) * q0(x)**(1-beta) * (q_numerical(x)*K(y|x,E))**beta'
ENVIRONMENT_TILT = ' * R_frozen_environment(E)**(1-beta)'
NUMERICAL_MEASURE = 'continuous-native-inverse-table-v1'
NUMERICAL_SOURCES = ('numerical_native_intensity.py', 'galsbi_prior.py', 'galsbi_scene_prior.py',
                     'check_galsbi_continuous_prior.py', 'numerical_prior_inputs.py', 'particle_target_measure.py')


def source_path(root, name):
    """Resolve declared sources without assuming every dependency is in sbsi/."""
    directory = ('scripts' if name == 'check_galsbi_continuous_prior.py' else
                 'configs' if name.endswith('.json') else 'sbsi')
    return Path(root)/directory/name


def numerical_identity(protocol, sources=None):
    """Validate and extract additive identity; return {} for legacy analytic.

    ``sources`` may be an already validated basename/hash mapping. By default
    use the full recorded implementation mapping. Numerical source hashes are
    retained in the returned identity; no path-dependent identity is introduced.
    """
    law = protocol.get('target_prior', 'analytic')
    if law not in ('analytic', 'numerical'):
        raise ValueError('unknown particle target prior')
    if law == 'analytic':
        if (protocol.get('endpoint_target') == NUMERICAL_ENDPOINT
                or 'q_numerical' in protocol.get('bridge', '')
                or 'numerical_prior_measure' in protocol
                or 'numerical_prior_source_sha256' in protocol
                or 'numerical_prior_inputs' in protocol or 'numerical_prior_input_paths' in protocol):
            raise ValueError('numerical target requires explicit numerical prior identity')
        return {}
    if protocol.get('endpoint_target') != NUMERICAL_ENDPOINT:
        raise ValueError('numerical prior requires its explicit numerical endpoint')
    if protocol.get('bridge') not in (NUMERICAL_BRIDGE, NUMERICAL_BRIDGE+ENVIRONMENT_TILT):
        raise ValueError('numerical prior requires a consistent numerical bridge')
    if protocol.get('numerical_prior_measure') != NUMERICAL_MEASURE:
        raise ValueError('explicit continuous numerical measure required')
    raw = protocol.get('implementation_sha256') if sources is None else sources
    if not isinstance(raw, dict):
        raise ValueError('numerical prior source hashes required')
    mapped = {}
    for path, value in raw.items():
        name = Path(path).name
        if name in mapped:
            raise ValueError('duplicate numerical prior source basename')
        mapped[name] = value
    selected = {}
    for name in NUMERICAL_SOURCES:
        value = mapped.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('numerical prior source hashes incomplete: '+name)
        selected[name] = value
    if ('numerical_prior_source_sha256' in protocol
            and protocol['numerical_prior_source_sha256'] != selected):
        raise ValueError('numerical prior source identity differs from implementation')
    return dict(target_prior='numerical', endpoint_target=NUMERICAL_ENDPOINT,
                numerical_prior_measure=NUMERICAL_MEASURE,
                numerical_prior_source_sha256=selected,
                numerical_prior_inputs=validate_prior_inputs(protocol.get('numerical_prior_inputs')))
