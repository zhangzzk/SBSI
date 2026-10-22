#!/usr/bin/env python
"""Measurement-only preflight after the complete v10 development gate.

Run through Slurm. Produces preparation data, not a launchable declaration.
The coordinator adds the completed allocation cost receipt before freezing
the declaration. No posterior pilot or production outcome is read here.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE), str(SOURCE/'scripts')]

PANEL_SHA256 = '3042f060344e97e5620c1f3b8a2ff5e5a1e1d2f876995d2984e0b66c7f27a8b5'
BLOCKS = ['physical', 'physical', 'nuisance', 'physical', 'environment',
          'physical', 'sed', 'physical', 'positions', 'physical', 'exchange',
          'physical', 'nuisance', 'physical', 'independence', 'physical']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validated_inputs(development, expected_development_hash, panel_path):
    """Verify the completed decision before opening any held-out input."""
    if digest(development) != expected_development_hash:
        raise ValueError('development receipt changed')
    result = json.loads(Path(development).read_text())
    if (result.get('schema') != 'sbsi-n23-corrected-comparison-v1'
            or result.get('development_checks_pass') is not True
            or result.get('accepted_sampler') is not False):
        raise ValueError('complete passing n23 development gate required')
    comparisons = result.get('comparisons', {})
    if set(comparisons) != {'small_minus_large', 'small_minus_broad', 'large_minus_broad'} or any(
            c.get('all_raw_gates') is not True or c.get('all_corrected_phase_gates') is not True
            for c in comparisons.values()):
        raise ValueError('all declared development comparisons must pass')
    # The audited development receipt pins all inputs; do not accept a detached
    # boolean with changed numerical evidence.
    sources = result.get('sources')
    if not sources or any(digest(path) != checksum for path, checksum in sources.items()):
        raise ValueError('development evidence changed or missing')
    if digest(panel_path) != PANEL_SHA256:
        raise ValueError('frozen measured-property panel changed')
    panel = json.loads(Path(panel_path).read_text())
    groups = [b['observations'] for b in panel['bands']]
    flat = [oid for group in groups for oid in group]
    if (len(groups) != 5 or any(len(group) != 4 for group in groups)
            or len(set(flat)) != 20 or set(flat).intersection(panel['excluded_development'])):
        raise ValueError('twenty distinct held-out observations required')
    return result, panel, groups


def proposal_laws(measurements, groups, conditions, parameters):
    """Use only the same four measured columns and constructor as inference."""
    from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
    laws = {}
    for family, key in [('gaussian', 'pilot_proposal_metadata'),
                        ('student_t', 'reference_proposal_metadata')]:
        laws[key] = [[MeasurementParticleProposal.from_measurement(
            measurements[oid], conditions, parameters, [0., 0.],
            component_family=family, student_df=5.).metadata for oid in group] for group in groups]
    return laws


def same_scientific_source(current, development):
    required = {name for name in development if name.startswith(('sbsi/', 'configs/'))}
    required.update(('scripts/check_galsbi_particles.py', 'scripts/check_galsbi_continuous_prior.py',
                     'scripts/fit_particle_gradient_geometry.py', 'scripts/fit_particle_initial_proposal.py',
                     'scripts/check_particle_phase_control.py', 'scripts/report_particle_phase_control.py',
                     'scripts/report_galsbi_particles.py'))
    if not required or any(name not in development or current.get(name) != development[name] for name in required):
        raise ValueError('scientific source differs from completed corrected development comparison')


def package_files(module):
    """Pin package source/data/binaries, excluding mutable bytecode and VCS."""
    root = Path(module.__file__).resolve().parent
    return sorted(p.resolve() for p in root.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and '.git' not in p.parts
                  and p.suffix not in ('.pyc', '.pyo', '.log', '.lock'))


def require_source_modules(source, modules):
    source = Path(source).resolve()
    if any(source not in Path(module.__file__).resolve().parents for module in modules):
        raise ValueError('actual imported scientific module is outside frozen source')


def audited_open(values):
    """Classify an open at its actual working directory, retaining write outputs separately."""
    path, mode, flags = values
    if not isinstance(path, (str, bytes)):
        return None, False
    path = Path(os.fsdecode(path)).resolve()
    if isinstance(flags, int):
        readonly = (flags & os.O_ACCMODE) == os.O_RDONLY and not flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND)
    else:
        readonly = isinstance(mode, str) and mode.startswith('r') and not any(x in mode for x in '+wax')
    return path, bool(readonly)


def parameter_record(value):
    """Preserve numerical parameters; identify nonnumerical Python bindings."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=parameter_record(value.tolist()))
    if isinstance(value, np.generic):
        return parameter_record(value.item())
    if isinstance(value, dict):
        return {str(k): parameter_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [parameter_record(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite_float='nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf'))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return dict(python_binding=getattr(value, '__module__', type(value).__module__)+'.'+
                getattr(value, '__qualname__', getattr(value, '__name__', type(value).__qualname__)))


def prepare(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    development, panel, groups = validated_inputs(args.development_receipt,
        args.development_receipt_sha256, args.panel)
    source = SOURCE
    import sbsi
    import fit_declared_particle_phase_control as fitter
    require_source_modules(source, [sbsi, fitter, fitter.phase_module])
    from fit_declared_particle_phase_control import verified_manifest
    entries = verified_manifest(source, args.source_manifest_sha256)
    if entries.get('scripts/'+Path(__file__).name) != digest(__file__):
        raise ValueError('preflight source must belong to the frozen manifest')
    declarations = [p for p in development['sources'] if Path(p).name == 'experiment_v10_corrected_three_way.json']
    if len(declarations) != 1:
        raise ValueError('unique completed development declaration required')
    dev = json.loads(Path(declarations[0]).read_text())
    dev_entries = verified_manifest(Path(dev['source']), dev['source_manifest_sha256'])
    same_scientific_source(entries, dev_entries)
    reference = Path(panel['reference'])
    if (digest(reference/'protocol.json') != panel['reference_protocol_sha256']
            or digest(reference/'observations.parquet') != panel['observations_sha256']):
        raise ValueError('reference data changed')
    protocol = json.loads((reference/'protocol.json').read_text())
    # Torch first is required by the existing PyCosmo runtime. No model inference
    # or posterior outcomes are evaluated in this preparation stage.
    import torch
    import numpy as np
    import pandas as pd
    import check_galsbi_continuous_prior as prior_factory
    from sbsi import models, galsbi_particle_proposal
    require_source_modules(source, [prior_factory, models, galsbi_particle_proposal])
    build_prior, V39_LIKE = prior_factory.build_prior, models.V39_LIKE
    torch.set_num_threads(2)
    V39_LIKE.validate()
    opened, written = set(), set()
    collecting = [True]

    def audit(event, values):
        if collecting[0] and event == 'open':
            path, readonly = audited_open(values)
            if path is not None:
                (opened if readonly else written).add(path)

    sys.addaudithook(audit)
    try:
        prior, _, _ = build_prior()
    finally:
        collecting[0] = False
    # Read only measured observables, never generating properties or historical
    # held-out pilot files. Column order is the inference driver's four fields.
    import pyarrow.parquet as pq
    measured_columns = pq.ParquetFile(reference/'observations.parquet').schema.names[:4]
    table = pd.read_parquet(reference/'observations.parquet', columns=measured_columns)
    measurements = {oid: table.iloc[oid].to_numpy() for group in groups for oid in group}
    laws = proposal_laws(measurements, groups, protocol['conditions'], prior.p)
    modules = [importlib.import_module(name) for name in ('galsbi', 'PyCosmo', 'cosmo_torrent')]
    files = set(p for module in modules for p in package_files(module))
    files.update(p for p in opened-written if p.is_file()
                 and p.suffix not in ('.pyc', '.pyo', '.log', '.lock') and '__pycache__' not in p.parts)
    files.difference_update(written)
    # h5py's C-level h5f.open need not emit Python's open audit event.
    from galsbi.load import data_path
    files.update([Path(data_path('Fischbacher+24_abc_posterior'))/'abc_posterior.h5',
                  Path(prior.p['maps_remote_dir'])/prior.p['templates_int_tables_file_name']])
    files.update([*V39_LIKE.flow_checkpoints, *V39_LIKE.detection_classifiers,
                  V39_LIKE.emulator_model, V39_LIKE.emulator_metadata,
                  reference/'protocol.json', reference/'result.json', reference/'observations.parquet', args.panel.resolve()])
    external = {str(p.resolve()): digest(p) for p in sorted(files) if source not in p.resolve().parents}
    common = dict(stages=64, blocks=BLOCKS, scale=1., power=2.)
    draft = dict(schema='sbsi-heldout-corrected-declaration-v1', source=str(source),
        source_manifest_sha256=args.source_manifest_sha256,
        source_files={str(source/name): checksum for name, checksum in entries.items()},
        external_files=external, output=str(args.heldout_output.resolve()),
        panel=str(args.panel.resolve()), panel_sha256=PANEL_SHA256,
        groups=groups, development_ids=panel['excluded_development'],
        reference=str(reference), reference_protocol_sha256=panel['reference_protocol_sha256'],
        observation_sha256=panel['observations_sha256'], shear=[0., 0.], density_batch_size=4096,
        pilot=dict(common, replicas=4, particles=256, moves=32, seed_base=2026103200),
        candidate_settings=dict(common, replicas=8, particles=256, moves=64, seed_base=2026103300),
        reference_settings=dict(common, replicas=8, particles=1024, moves=64, seed_base=2026103400),
        development_receipt=dict(path=str(args.development_receipt.resolve()), sha256=args.development_receipt_sha256),
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__),
            cuda_device_name='Tesla V100-SXM2-32GB', slurm_node_list='th-cl-nv02', torch_num_threads=2),
        preparation_cost_receipt=None, **laws)
    # Preserve the actual constructed prior data; these are provenance, not
    # posterior moments or replacement model parameters.
    prior_record = dict(parameters=parameter_record(prior.p), z_max=prior.z_max,
        parent_rate=prior.parent_rate, tables=parameter_record(prior.tables))
    verified_manifest(source, args.source_manifest_sha256)
    result = dict(schema='sbsi-corrected-heldout-preflight-v1', declaration_draft=draft,
        constructed_prior=prior_record, source_script_sha256=digest(__file__),
        generated_or_write_capable_prior_paths=[str(p) for p in sorted(written)],
        preflight_internal_seconds=time.monotonic()-started,
        launchable=False, production_or_pilot_outcomes_read=False,
        limitations=['The coordinator must attach the completed allocation cost receipt and freeze a declaration before any warmup.',
            'Package files and Python-audited prior inputs are pinned; this is not a whole-system syscall trace or an environment image.',
            'Constructed prior tables are recorded at preflight, not asserted to be independently verified in each subsequent sampler process.',
            'The preflight internal timer excludes interpreter startup and final serialization; use the completed scheduler cost receipt.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=str(args.output), sha256=digest(args.output), launchable=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-receipt', type=Path, required=True)
    parser.add_argument('--development-receipt-sha256', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--heldout-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
