"""CPU comparison against frozen legacy source; no saved posterior outcomes."""
import argparse
from copy import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--legacy', type=Path, required=True)
parser.add_argument('--support-receipt', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.source), str(args.source/'scripts'), str(args.source/'tests')]
import numpy as np
import torch
from sbsi import scene_exchange as fixed
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from test_informed_environment import fixture, assert_state_equal
from check_galsbi_continuous_prior import build_prior


def legacy_module(filename, name):
    spec = importlib.util.spec_from_file_location('sbsi.'+name, args.legacy/'sbsi'/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


original = legacy_module('scene_exchange.py', '_frozen_exchange_support_audit')
old_target = legacy_module('galsbi_particle_target.py', '_frozen_target_support_audit')
old_target.propose_native_exchange = original.propose_native_exchange
target, state, _ = fixture(4)
class Legacy(type(target), old_target.GalSBIParticleTarget):
    evaluate = old_target.GalSBIParticleTarget.evaluate
    initialize = old_target.GalSBIParticleTarget.initialize
    mutate = old_target.GalSBIParticleTarget.mutate
legacy = copy(target)
legacy.__class__ = Legacy
rng, oldrng = np.random.default_rng(37), np.random.default_rng(37)
assert_state_equal(target.initialize(rng), legacy.initialize(oldrng))
assert rng.bit_generator.state == oldrng.bit_generator.state
oldstate = legacy.evaluate(state.x, state.scene)
rng, oldrng = np.random.default_rng(928), np.random.default_rng(928)
blocks = ['physical', 'nuisance', 'sed', 'environment', 'positions',
          'exchange', 'exchange_relocate', 'independence']
for beta in (.001, .4, 1.):
    for block in blocks:
        state, a, b = target.mutate(state, beta, rng, block=block)
        oldstate, c, d = legacy.mutate(oldstate, beta, oldrng, block=block)
        assert_state_equal(state, oldstate)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        assert rng.bit_generator.state == oldrng.bit_generator.state

galaxy, _, _ = build_prior()
generated, _ = galaxy.sample(512, 2026102209)
ratio, audit = fixed.numerical_over_analytic(galaxy, generated, return_support_audit=True)
np.testing.assert_array_equal(ratio, original.numerical_over_analytic(galaxy, generated))
assert not audit['below_uniform_clamp'].any()
support = json.loads(args.support_receipt.read_text())
examples = []
for example in support['examples']:
    native = {key: np.asarray([value]) for key, value in example['native'].items()}
    photo = galaxy.photometry(native['z'], native['M_B'], native['c'])
    native.update(photo)
    neighbour = {key: value[:1].copy() for key, value in generated.items() if key != 'proposal_index'}
    assert native.keys() == neighbour.keys()
    scene = GalSBISceneBatch(native, neighbour, [0, 1], [[.2, .4]])
    probability = np.eye(len(galaxy.types))[native['type']]
    rng, oldrng = np.random.default_rng(19), np.random.default_rng(19)
    moved, ratio, active = fixed.propose_native_exchange(galaxy, scene, probability, rng)
    oldmoved, oldratio, oldactive = original.propose_native_exchange(galaxy, scene, probability, oldrng)
    assert np.isneginf(ratio).all() and np.isfinite(oldratio).all()
    assert rng.bit_generator.state == oldrng.bit_generator.state
    np.testing.assert_array_equal(active, oldactive)
    for role in ('targets', 'neighbours'):
        for key in native:
            np.testing.assert_array_equal(getattr(moved, role)[key], getattr(oldmoved, role)[key])
    examples.append(dict(type=example['type'], legacy_correction=float(oldratio[0]),
                         fixed_rejects=True, proposal_payload_and_rng_identical=True))
result = dict(status='passed', ordinary_full_payload_and_rng_bitwise=True,
    ordinary_sequence=dict(particles=4, betas=[.001, .4, 1.], blocks=blocks),
    ordinary_production_marks=len(generated['z']), ordinary_production_ratios_bitwise=True,
    production_support_examples=examples,
    inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.source/'sbsi/scene_exchange.py', args.legacy/'sbsi/scene_exchange.py',
        args.legacy/'sbsi/galsbi_particle_target.py', args.support_receipt)})
args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2))
