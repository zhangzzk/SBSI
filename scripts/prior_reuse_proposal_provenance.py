"""Explicit proposal provenance migration for the pinned prior-reuse experiment.

No fit is recomputed and no public artifact validator is relaxed. The original
proposal remains immutable; the derived artifact carries an auditable new
implementation identity while preserving every numerical and pilot-fit field.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

MIGRATION_KEY = 'implementation_provenance_migration'
MIGRATION_SCHEMA = 'sbsi-prior-reuse-proposal-provenance-v1'
NATIVE_NAMES = ('galsbi_target_coordinates.py', 'galsbi_summary.py',
                'summary_gradients.py', 'models_v3_9_like.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def native_sources(entries):
    result = {}
    for name in NATIVE_NAMES:
        values = [value for path, value in entries.items() if Path(path).name == name]
        if len(values) != 1:
            raise ValueError('unique complete native source identity required: ' + name)
        result[name] = values[0]
    return result


def derived_payload(original_path, native_path, arms, sources, changes):
    original = load(original_path)
    if MIGRATION_KEY in original['metadata']:
        raise ValueError('migration must start from the original unmigrated pilot fit')
    if (set(sources['baseline']) != set(sources['optimized']) or
            {name for name in sources['baseline'] if sources['baseline'][name] != sources['optimized'][name]} != set(changes)):
        raise ValueError('only the declared implementation changes may be migrated')
    for name, (old, new) in changes.items():
        if sources['baseline'][name] != old or sources['optimized'][name] != new:
            raise ValueError('unrecognized implementation migration')
    before, after = native_sources(sources['baseline']), native_sources(sources['optimized'])
    if original['metadata']['native_target_source_sha256'] != before:
        raise ValueError('original proposal does not match baseline native target')
    if {name for name in before if before[name] != after[name]} != {'galsbi_target_coordinates.py'}:
        raise ValueError('only the validated coordinate implementation identity may change')
    evidence = load(native_path)
    if (evidence.get('status') != 'passed' or evidence.get('all_exactly_equal') is not True or
            evidence.get('post_run_hashes_match') is not True or
            evidence.get('development_rows') != 73728 or evidence.get('probe_rows') != 15):
        raise ValueError('complete successful native prior/photometry replay required')
    for name, pair in changes.items():
        observed = {v for p, v in evidence['source_sha256'].items() if p.endswith('/' + name)}
        if observed != set(pair):
            raise ValueError('native replay does not bind both source identities: ' + name)
    receipt = dict(schema=MIGRATION_SCHEMA,
        original_artifact=dict(path=str(Path(original_path).resolve()), sha256=digest(original_path)),
        native_replay=dict(path=str(Path(native_path).resolve()), sha256=digest(native_path),
                           development_rows=73728, probe_rows=15),
        source_manifests={arm: dict(path=str(Path(item['source']).resolve()), sha256=item['manifest_sha256'])
                          for arm, item in arms.items()},
        source_changes={name: dict(original_sha256=pair[0], derived_sha256=pair[1]) for name, pair in changes.items()},
        original_native_target_source_sha256=before,
        derived_native_target_source_sha256=after,
        unchanged_proposal_content_sha256=canonical_digest({key: value for key, value in original.items() if key != 'metadata'}),
        unchanged_pilot_metadata_sha256=canonical_digest({key: value for key, value in original['metadata'].items()
                                                         if key != 'native_target_source_sha256'}),
        operation='Implementation provenance only; all numerical proposal parameters and pilot-fit metadata preserved; no refit or new samples.')
    derived = deepcopy(original)
    derived['metadata']['native_target_source_sha256'] = after
    derived['metadata'][MIGRATION_KEY] = receipt
    return derived


def validate_migration(d, sources, changes):
    item = d['proposal_migration']
    if (item['original'] != d['initial_proposal'] or d['initial_proposals']['baseline'] != item['original'] or
            d['initial_proposals']['optimized'] != item['derived'] or
            item['original'] == item['derived'] or digest(item['native_replay']) != item['native_replay_sha256']):
        raise ValueError('explicit original/derived proposal and evidence identities required')
    expected = derived_payload(item['original'], item['native_replay'], d['arms'], sources, changes)
    actual = load(item['derived'])
    if canonical_digest(actual) != canonical_digest(expected):
        raise ValueError('derived proposal changed beyond the validated provenance migration')
    return load(item['original']), actual


def normalize_initial_protocol(protocol, d, arm, sources, changes):
    original, derived = validate_migration(d, sources, changes)
    p = deepcopy(protocol)
    artifact = original if arm == 'baseline' else derived
    path = d['initial_proposals'][arm]
    raw = p['initial_proposal']
    if (raw['path'] != str(Path(path).resolve()) or raw['sha256'] != digest(path) or
            canonical_digest(raw['metadata']) != canonical_digest(artifact['metadata'])):
        raise ValueError('raw production proposal provenance differs from its declared artifact')
    # This is solely comparison bookkeeping after validating the entire derived
    # payload. No scientific value, source validator or stored output is altered.
    p['initial_proposal'] = dict(raw, path=str(Path(d['initial_proposal']).resolve()),
        sha256=digest(d['initial_proposal']), metadata=original['metadata'])
    return p


def validate_failed_startup(item):
    """Retain the original unsuccessful attempt; never classify it as a pair."""
    for key in ('receipt', 'log', 'cost'):
        if digest(item[key]['path']) != item[key]['sha256']:
            raise ValueError('original failed startup evidence changed')
    receipt, cost = load(item['receipt']['path']), load(item['cost']['path'])
    if (receipt['status'] != 'failed' or receipt['resources']['slurm_job_id'] != '17446115' or
            receipt.get('replica') != 0 or receipt.get('seed') != 2026103900 or
            receipt.get('order') != ['baseline', 'optimized'] or
            receipt['arms']['baseline']['workers']['0']['returncode'] != 0 or
            receipt['arms']['optimized']['workers']['0']['returncode'] != 1 or
            cost['exit_code'] != 1 or cost != item['process_cost'] or
            'production target identity differs from pilot: native_target_source_sha256' not in Path(item['log']['path']).read_text()):
        raise ValueError('original failure must be the preserved pre-sampling provenance rejection')
    optimized = Path(item['optimized_output'])
    if not optimized.is_dir() or any(optimized.iterdir()):
        raise ValueError('optimized v1 must not contain any sampled or protocol artifacts')
    return cost
