#!/usr/bin/env python
"""Fit phase coefficients from exactly four independently declared pilots.

This is separate from the historical n15 fitter, whose dev_v1-only restriction
remains intact. A declaration and its frozen source manifest are mandatory.
Fitting is algorithmic preparation, never an evaluation of held-out outcomes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi import phase_control as phase_module
from sbsi.phase_control import digest, fit_frozen_phase_control, load_phase_diagnostic

PROTOCOL_SOURCES = {
    'scripts/check_galsbi_particles.py', 'sbsi/annealed_particles.py',
    'sbsi/galsbi_particle_proposal.py', 'sbsi/galsbi_particle_target.py',
    'sbsi/scene_exchange.py', 'sbsi/galsbi_target_coordinates.py',
    'sbsi/galsbi_summary.py', 'sbsi/summary_gradients.py',
    'configs/models_v3_9_like.json', 'sbsi/particle_populations.py',
}
DIAGNOSTIC_SOURCES = {'scripts/check_particle_phase_control.py', 'sbsi/phase_control.py'}
NATIVE_SOURCES = {'galsbi_target_coordinates.py', 'galsbi_summary.py',
                  'summary_gradients.py', 'models_v3_9_like.json'}


def verified_manifest(source, expected_hash):
    source = Path(source).resolve()
    manifest = source/'SHA256SUMS'
    if digest(manifest) != expected_hash:
        raise ValueError('declared source manifest changed')
    entries = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        path = (source/name).resolve()
        if source not in path.parents or name in entries or digest(path) != checksum:
            raise ValueError('invalid, duplicate, or changed frozen source')
        entries[name] = checksum
    if not entries:
        raise ValueError('empty source manifest')
    return entries


def validate_pilot_records(declaration, group, reports, protocols, paths):
    """Reject production leakage, reordered pilots, and mismatched recipes."""
    if declaration.get('schema') != 'sbsi-heldout-corrected-declaration-v1':
        raise ValueError('explicit corrected held-out declaration required')
    if type(group) is not int or not 0 <= group < len(declaration['groups']):
        raise ValueError('declared group index required')
    pilot = declaration['pilot']
    if pilot['replicas'] != 4 or not (len(reports) == len(protocols) == len(paths) == 4):
        raise ValueError('exactly four declared pilots required')
    seeds = [pilot['seed_base']+10*group+i for i in range(4)]
    all_seeds = []
    for key in ('pilot', 'candidate_settings', 'reference_settings'):
        settings = declaration[key]
        all_seeds.extend(settings['seed_base']+10*g+i
                         for g in range(len(declaration['groups']))
                         for i in range(settings['replicas']))
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError('pilot/production seeds overlap across groups or arms')
    observations = declaration['groups'][group]
    flat = [oid for row in declaration['groups'] for oid in row]
    if (len(declaration['groups']) != 5 or any(len(row) != 4 for row in declaration['groups'])
            or len(flat) != len(set(flat)) or set(flat).intersection(declaration['development_ids'])):
        raise ValueError('five distinct four-observation held-out groups required')
    directory = Path(declaration['output']).resolve()/f'group_{group}'
    identity = reports[0]['target_identity']
    for i, (report, protocol, path) in enumerate(zip(reports, protocols, paths)):
        run = directory/f'pilot_{i}'
        if (Path(path).resolve() != directory/f'pilot_{i}_phase.json'
                or Path(report['run']).resolve() != run
                or report['seed'] != seeds[i] or protocol['seed'] != seeds[i]
                or report['observations'] != observations or protocol['observations'] != observations
                or report['target_identity'] != identity
                or report['stride'] != 1 or report['retained_particles'] != pilot['particles']
                or report['numerical_checks_pass'] is not True
                or any(protocol[k] != pilot[k] for k in ('particles', 'stages', 'moves'))
                or protocol['blocks'] != pilot['blocks'] or protocol['scale'] != pilot['scale']
                or protocol['betas'] != (np.linspace(0., 1., pilot['stages']+1)**pilot['power']).tolist()
                or protocol['fixed_shear'] != declaration['shear']
                or protocol['density_batch_size'] != declaration['density_batch_size']
                or protocol.get('initial_proposal') is not None
                or protocol.get('physical_geometry') is not None
                or protocol.get('informed_environment') is not None):
            raise ValueError('pilot diagnostic or protocol differs from declaration')
        if len(protocol['proposals']) != 4 or any(
                p['component_family'] != 'gaussian' or p['faint_magnitude_shift'] != 5.
                or p['weights'] != [.5, .5] or p['phase'] != 'independent uniform periodic'
                for p in protocol['proposals']):
            raise ValueError('original Gaussian pilot proposal required')
        if protocol['proposals'] != declaration['pilot_proposal_metadata'][group]:
            raise ValueError('pilot proposal law differs from declared measured-input construction')
        for key in ('reference_protocol_sha256', 'observation_sha256'):
            if protocol[key] != declaration[key]:
                raise ValueError('pilot reference identity changed')
        for key in ('observations', 'fixed_shear', 'reference_protocol_sha256',
                    'observation_sha256', 'density_batch_size', 'selection'):
            if identity.get(key) != protocol[key]:
                raise ValueError('diagnostic target identity differs from its bound pilot protocol')
    return seeds, observations, identity


def fit_declared(declaration_path, group, output):
    declaration_path, output = Path(declaration_path).resolve(), Path(output).resolve()
    declaration = json.loads(declaration_path.read_text())
    panel_path = Path(declaration['panel'])
    if digest(panel_path) != declaration['panel_sha256']:
        raise ValueError('frozen held-out panel changed')
    panel = json.loads(panel_path.read_text())
    if declaration['groups'] != [band['observations'] for band in panel['bands']]:
        raise ValueError('declaration groups differ from frozen panel')
    source = Path(declaration['source']).resolve()
    if Path(__file__).resolve().parents[1] != source:
        raise ValueError('fitter must execute inside the declared frozen source')
    if Path(phase_module.__file__).resolve() != source/'sbsi/phase_control.py':
        raise ValueError('phase-control module must execute inside the declared frozen source')
    entries = verified_manifest(source, declaration['source_manifest_sha256'])
    if entries.get('scripts/'+Path(__file__).name) != digest(__file__):
        raise ValueError('fitter is absent from the frozen manifest')
    directory = Path(declaration['output']).resolve()/f'group_{group}'
    if output != directory/'phase_coefficients.json' or output.exists():
        raise ValueError('fresh declared coefficient output required')
    # Freeze before any production begins, including another replica in the group.
    if list(directory.glob('candidate_*')) or list(directory.glob('reference_*')):
        raise ValueError('coefficients must be frozen before production starts')
    paths = [directory/f'pilot_{i}_phase.json' for i in range(4)]
    reports, arrays = zip(*(load_phase_diagnostic(path) for path in paths))
    protocols = [json.loads((directory/f'pilot_{i}'/'protocol.json').read_text()) for i in range(4)]
    seeds, observations, identity = validate_pilot_records(declaration, group, reports, protocols, paths)
    for i, report in enumerate(reports):
        run = directory/f'pilot_{i}'
        expected = {str(run/name): digest(run/name)
                    for name in ('protocol.json', 'result.json', 'particles.npz')}
        if report['source_hashes'] != expected:
            raise ValueError('diagnostic provenance must bind exactly its declared pilot artifacts')
    for record, required in [(r, PROTOCOL_SOURCES) for r in protocols]+[(r, DIAGNOSTIC_SOURCES) for r in reports]:
        recorded = {}
        for path, checksum in record['implementation_sha256'].items():
            relative = Path(path).resolve().relative_to(source).as_posix()
            if relative in recorded or entries.get(relative) != checksum:
                raise ValueError('pilot or diagnostic implementation differs from frozen source')
            recorded[relative] = checksum
        if not required.issubset(recorded):
            raise ValueError('mandatory implementation sources missing')
    expected_native = {Path(path).name: checksum for path, checksum in entries.items()
                       if Path(path).name in NATIVE_SOURCES and path.split('/')[0] in ('configs', 'sbsi')}
    if set(expected_native) != NATIVE_SOURCES or identity.get('native_source_sha256') != expected_native:
        raise ValueError('diagnostic native source identity differs from frozen source')
    metadata = dict(observations=observations, pilot_seeds=seeds, target_identity=identity,
        declaration=dict(path=str(declaration_path), sha256=digest(declaration_path)),
        source_manifest_sha256=declaration['source_manifest_sha256'],
        pilot_diagnostics=[dict(path=str(path), sha256=digest(path),
            arrays_sha256=digest(path.with_suffix('.npz')), run=report['run'],
            original_sources=report['source_hashes']) for path, report in zip(paths, reports)],
        fit_script_sha256=digest(__file__),
        estimand='continuous float64 endpoint; raw float32-sampled use remains a qualified diagnostic')
    frozen = fit_frozen_phase_control([a['raw_scores'] for a in arrays],
                                     [a['phase_control'] for a in arrays], metadata)
    frozen.save(output)
    verified_manifest(source, declaration['source_manifest_sha256'])
    return dict(output=str(output), sha256=digest(output), pilot_seeds=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit_declared(args.declaration, args.group, args.output)), flush=True)


if __name__ == '__main__':
    main()
