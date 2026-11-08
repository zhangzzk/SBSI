import copy

import pytest

from scripts.prepare_independent_joint_validation import require_declared_scene
from scripts.validate_independent_joint import require_design_endpoints


def test_independent_design_requires_every_declared_leg_and_exact_seeds():
    design = {'cases': [22000, 22001], 'scene_seeds': [22123, 22124],
        'noise_seeds': [4000000000, 4001000003], 'area_deg2': 1.,
        'primary_amplitude': .05, 'additional_amplitude': .02, 'additional_amplitude_cases': [22000]}
    m = {'case': 22000, 'scene_seed': 22123, 'noise_seed': 4000000000, 'area_deg2': 1.,
        'legs': [{'group': 'g0', 'h': 0., 'sky_shear': [0., 0.]}]}
    for axis in (1, 2):
        for h in (.05, .02):
            for sign in (-1, 1):
                m['legs'].append({'group': f'g{axis}_h{h:g}', 'h': sign*h,
                    'sky_shear': [sign*h, 0.] if axis == 1 else [0., sign*h]})
    require_declared_scene(m, design)
    for key, value in [('case', 30000), ('scene_seed', 22124), ('noise_seed', 9),
                        ('area_deg2', .01), ('legs', m['legs'][:-1]), ('legs', m['legs']+[m['legs'][0]])]:
        changed = copy.deepcopy(m)
        changed[key] = value
        with pytest.raises(ValueError):
            require_declared_scene(changed, design)
    m['case'], m['scene_seed'], m['noise_seed'] = 22001, 22124, 4001000003
    with pytest.raises(ValueError, match='shear legs'):
        require_declared_scene(m, design)
    m['legs'] = [leg for leg in m['legs'] if abs(leg['h']) != .02]
    require_declared_scene(m, design)


def test_endpoint_contract_rejects_scene_and_amplitude_cherry_picking():
    design = {'cases': list(range(22000, 22040)), 'primary_amplitude': .05,
              'additional_amplitude': .02, 'additional_amplitude_cases': list(range(22000, 22004))}
    endpoints = []
    for h, cases in ((.05, design['cases']), (.02, design['additional_amplitude_cases'])):
        for axis in (1, 2):
            name = f'g{axis}_{h}'
            endpoints.append({'name': name, 'axis': axis, 'h': h, 'cases': cases,
                'output': name+'.json', 'arguments': ['--axis', str(axis), '--h', str(h),
                    '--cases', *map(str, cases), '--output', name+'.json']})
    require_design_endpoints(endpoints, design)
    with pytest.raises(ValueError):
        require_design_endpoints(endpoints[:2], design)
    changed = copy.deepcopy(endpoints)
    changed[0]['cases'] = changed[0]['cases'][1:]
    with pytest.raises(ValueError):
        require_design_endpoints(changed, design)
    changed = copy.deepcopy(endpoints)
    changed[0]['arguments'][3] = '0.02'
    with pytest.raises(ValueError, match='arguments'):
        require_design_endpoints(changed, design)
