"""Whitelist-only provenance audit: never reads inference or phase outcomes."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def read_json(path):
    # Scientific outputs, logs, particles and phase results are excluded.
    assert path.name == 'protocol.json' or path.name.startswith(('experiment_', 'snapshot_')) or path.parent.name == 'configs'
    return json.loads(path.read_text())


receipt = dict(date=datetime.now(timezone.utc).isoformat(),
    purpose='Current explicit endpoint compatibility; no v8/v9 or heldout outcomes read',
    source_snapshots={}, protocols={}, external_artifacts={})
for name in ('source_v7', 'source_v8'):
    directory = root/name
    manifest = directory/'SHA256SUMS'
    verified = {}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip('*')
        path = directory/relative
        actual = digest(path)
        assert actual == expected, (name, relative, actual, expected)
        verified[str(path.relative_to(directory))] = actual
    receipt['source_snapshots'][name] = dict(manifest_sha256=digest(manifest),
        files_verified=len(verified), actual_source_sha256=verified)

a = receipt['source_snapshots']['source_v7']['actual_source_sha256']
b = receipt['source_snapshots']['source_v8']['actual_source_sha256']
receipt['changed_sbsi_files'] = [key for key in sorted(a.keys() | b.keys())
    if key.startswith('sbsi/') and key.endswith('.py') and a.get(key) != b.get(key)]
endpoint_files = ('scripts/check_galsbi_continuous_prior.py', 'sbsi/galsbi_prior.py',
    'sbsi/galsbi_scene_prior.py', 'sbsi/galsbi_target_coordinates.py', 'sbsi/galsbi_summary.py',
    'sbsi/summary_gradients.py', 'sbsi/catalogue_disk_likelihood.py', 'sbsi/catalogue_likelihood.py',
    'sbsi/catalogue_disk_response.py', 'sbsi/disk_response_transport.py', 'sbsi/disk_precision.py',
    'sbsi/flow_physical_disk.py', 'sbsi/flow_disk.py', 'sbsi/measurement_model.py',
    'sbsi/selection_model.py', 'sbsi/models.py', 'sbsi/nn_utils.py',
    'sbsi/forward_catalogue.py', 'sbsi/output_conditioned_response.py', 'sbsi/v39_inference.py',
    'configs/models_v3_9_like.json')
receipt['endpoint_factor_sha256'] = {}
for path in endpoint_files:
    actual_a, actual_b = digest(root/'source_v7'/path), digest(root/'source_v8'/path)
    assert actual_a == actual_b, path
    receipt['endpoint_factor_sha256'][path] = actual_a

groups = ('v7', 'v9', 'v8_baseline', 'v8_informed')
settings = ('observations', 'fixed_shear', 'reference', 'reference_protocol_sha256',
    'observation_sha256', 'density_batch_size', 'selection', 'betas', 'blocks', 'scale')
common = None
for group in groups:
    paths = sorted(root.glob(f'dev_{group}_*/protocol.json'))
    rows = []
    group_fingerprint = None
    for path in paths:
        d = read_json(path)
        assert d['particles'] == (1024 if group == 'v9' else 256)
        assert d['stages'] == 64 and d['moves'] == (64 if group == 'v9' else 32)
        values = {key: d[key] for key in settings}
        if common is None:
            common = values
        assert values == common, path
        assert d['physical_geometry']['sha256'] == 'a2eb847b956a986a94257bf2fce808185813e447dafcc45981f499a7c9d9b2c0'
        assert d['initial_proposal']['sha256'] == '7f11db32019d4e0014a10acc61ea890c17d6fc61b67d0fd9ff36afe3763c46e4'
        for field in ('physical_geometry', 'initial_proposal', 'informed_environment'):
            if field in d:
                artifact = d[field]
                assert digest(artifact['path']) == artifact['sha256']
                receipt['external_artifacts'][artifact['path']] = artifact['sha256']
        implementations = {}
        for name, expected in d['implementation_sha256'].items():
            actual = digest(name)
            assert actual == expected, (path, name)
            implementations[name] = actual
        declared_snapshot = 'source_v7' if group in ('v7', 'v9') else 'source_v8'
        assert all(str(root/declared_snapshot) in name for name in implementations)
        fingerprint = canonical({key: value for key, value in d.items() if key not in ('seed', 'runtime')})
        if group_fingerprint is None:
            group_fingerprint = fingerprint
        assert fingerprint == group_fingerprint, ('within-group settings mismatch', path)
        rows.append(dict(path=str(path), sha256=digest(path), seed=d['seed'],
            snapshot=declared_snapshot, implementation_sha256=implementations,
            endpoint_target=d.get('endpoint_target'), bridge=d['bridge'],
            particles=d['particles'], moves=d['moves'],
            proposal_config_sha256=canonical(d['proposals']),
            informed_config_sha256=canonical(d.get('informed_environment')),
            source_omissions=[name for name in endpoint_files
                if not any(k.endswith('/'+name) for k in implementations)]))
    receipt['protocols'][group] = dict(available_startup_protocols=len(rows),
        expected_replicates=8, completeness_not_assessed_from_outcomes=True,
        within_group_fingerprint=group_fingerprint, runs=rows)

receipt['common_settings'] = common
reference = Path(common['reference'])
reference_protocol = read_json(reference/'protocol.json')
assert digest(reference/'protocol.json') == common['reference_protocol_sha256']
assert digest(reference/'observations.parquet') == common['observation_sha256']
receipt['reference'] = dict(protocol_sha256=common['reference_protocol_sha256'],
    observations_sha256=common['observation_sha256'], conditions=reference_protocol['conditions'],
    response_pairing=reference_protocol['response_pairing'], prior=reference_protocol['prior'],
    historic_implementation_sha256=reference_protocol['implementation_sha256'],
    qualification='Historical generation source hashes are retained as declared; this audit compares inference endpoints and does not assert numerical mock-target equality.')

config = read_json(root/'source_v7/configs/models_v3_9_like.json')
roots = dict(SBSI_CACHE_DIR=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches'),
    BLENDEMU_RUNS_DIR=Path('/project/ls-gruen/users/zekang.zhang/blendemu_runs'))
for role in ('flow', 'classifier', 'emulator'):
    item = config[role]
    path = roots[item['root']]/item['path']
    assert digest(path) == item['sha256']
    receipt['external_artifacts'][str(path)] = item['sha256']
    if 'metadata_path' in item:
        path = roots[item['root']]/item['metadata_path']
        assert digest(path) == item['metadata_sha256']
        receipt['external_artifacts'][str(path)] = item['metadata_sha256']

declarations = ('experiment_v9_higher_effort_fitted.json', 'experiment_v8_informed_environment.json', 'snapshot_v8.json')
receipt['declarations'] = {name: dict(sha256=digest(root/name), content=read_json(root/name)) for name in declarations}
receipt['launchers_sha256'] = {name: digest(root/name) for name in ('run_v9.sh', 'run_v8.sh')}
receipt['missing_historic_provenance'] = [
    'v7/v9 have no endpoint_target field; inferred here from identical frozen source and bridge at beta=1.',
    'Default protocol hashes omit prior/environment/factory and several transitive likelihood sources; snapshot hashes supply current source evidence.',
    'External GalSBI/PyCosmo package/data and runtime-built prior parameter/table hashes were not recorded in the original protocols; a present environment check cannot prove historical immutability.',
    'Common declared endpoint does not establish correctness of every mutation: default exchange missed the lower uniform-clamp support bound.']
receipt['scientific_conclusion'] = 'Declared/common endpoint factors agree; v7/v9 use identical kernels. v8 default and informed share endpoint factors but only informed enforces the continuous exchange support bound. Interpret differences with that known kernel qualification until trajectory checks establish impact.'
args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(source_files={key: value['files_verified'] for key, value in receipt['source_snapshots'].items()},
    endpoint_factors_equal=len(endpoint_files), groups={key: value['available_startup_protocols'] for key, value in receipt['protocols'].items()},
    changed_sbsi_files=receipt['changed_sbsi_files']), indent=2))
