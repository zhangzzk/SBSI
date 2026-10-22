"""Geometry law routing is explicit without changing the symmetric kernel."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from sbsi.particle_geometry import FrozenParticleGeometry, geometry_target_identity
from test_particle_initial_proposal import identity_protocol
from test_particle_target_measure import numerical, module

ROOT = Path(__file__).resolve().parents[1]


def geometry(protocol=None):
    metadata = dict(observations=[11], pilot_seeds=[1, 2])
    if protocol is not None: metadata.update(geometry_target_identity(protocol))
    return FrozenParticleGeometry(np.eye(5)[None], np.array([.2]), metadata)


def test_numeric_geometry_roundtrip_and_both_cross_law_rejections(tmp_path):
    p = numerical(); numeric = geometry(p); analytic = geometry()
    artifact = tmp_path/'geometry.json'; numeric.save(artifact)
    restored = FrozenParticleGeometry.load(artifact)
    restored.validate_production(9, [11], protocol=p)
    analytic.validate_production(9, [11])
    analytic.validate_production(9, [11], protocol=identity_protocol())
    for candidate, protocol in [(restored, identity_protocol()), (analytic, p)]:
        with pytest.raises(ValueError, match='geometry target law'):
            candidate.validate_production(9, [11], protocol=protocol)
    with pytest.raises(ValueError, match='explicit production protocol'):
        restored.validate_production(9, [11])
    changed = deepcopy(p)
    changed['numerical_prior_inputs']['files_sha256']['data/abc_posterior.h5'] = 'e'*64
    with pytest.raises(ValueError, match='geometry target law'):
        restored.validate_production(9, [11], protocol=changed)
    # The identity changes neither proposal arithmetic nor RNG consumption.
    a, b = np.random.default_rng(31), np.random.default_rng(31)
    np.testing.assert_array_equal(restored.increments(np.zeros(7, int), .4, a),
                                  analytic.increments(np.zeros(7, int), .4, b))
    assert a.bit_generator.state == b.bit_generator.state


def write_pilot(path, protocol, seed):
    path.mkdir()
    p = dict(protocol, seed=seed, particles=4, reference='/fixed/reference')
    rng = np.random.default_rng(seed)
    np.savez(path/'particles.npz', coordinates=rng.normal(size=(1, 4, 10)), summaries=rng.normal(size=(1, 4, 9)))
    (path/'protocol.json').write_text(json.dumps(p))
    (path/'result.json').write_text(json.dumps(dict(status='completed', observations=[11],
        protocol_sha256=hashlib.sha256((path/'protocol.json').read_bytes()).hexdigest(),
        particles_sha256=hashlib.sha256((path/'particles.npz').read_bytes()).hexdigest())))
    return path


@pytest.mark.parametrize('fitter', ['fit_particle_geometry', 'fit_particle_gradient_geometry'])
@pytest.mark.parametrize('laws', [('analytic', 'numerical'), ('numerical', 'analytic'), ('numerical', 'numerical')])
def test_both_fitters_preserve_numeric_identity_and_reject_mixed_pilots(tmp_path, monkeypatch, fitter, laws):
    runs = [write_pilot(tmp_path/str(i), numerical() if law=='numerical' else identity_protocol(), i+1)
            for i,law in enumerate(laws)]
    script = module(fitter, monkeypatch)
    if fitter == 'fit_particle_gradient_geometry':
        if laws[0] != laws[1]:
            with pytest.raises(ValueError, match='identit'): script.load_pilots(runs)
        else:
            _, metadata = script.load_pilots(runs)
            assert metadata['geometry_target_identity'] == geometry_target_identity(numerical())['geometry_target_identity']
    else:
        output = tmp_path/'frozen.json'
        monkeypatch.setattr(sys, 'argv', [fitter, '--runs', *map(str,runs), '--output', str(output)])
        if laws[0] != laws[1]:
            with pytest.raises(ValueError, match='identity'): script.main()
            assert not output.exists()
        else:
            script.main()
            FrozenParticleGeometry.load(output).validate_production(9, [11], protocol=numerical())
