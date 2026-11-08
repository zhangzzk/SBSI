"""Audit saved pilot after metadata-only failure; never change its arrays."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = CACHE / 'alignment_regularization_shift_20261106'
assert os.environ.get('SLURM_JOB_ID'), 'Slurm required'
dest = ROOT / 'fourstate/nine_weak/case1000'
receipt = dest.with_suffix('.json')
assert not receipt.exists()
log = (ROOT / 'logs/guard_shift_mixed_17805334_1.log').read_text()
assert "KeyError: 'portable_source_sha256'" in log
assert "out.with_suffix('.json').write_text" in log
assert 'Tesla V100-SXM2-32GB' in log
pin = json.loads((HERE / 'manifest.json').read_text())['arms']['nine_weak']
assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256']
bank = json.loads((CACHE / 'alignment_target9_20261106/device_verification.json').read_text())
assert bank['portable_source_sha256'] == hashlib.sha256(
    (HERE.parent / 'alignment_target9_20261106/portable_draws.py').read_bytes()).hexdigest()
baseline = ROOT / 'fourstate/baseline/case1000'
with np.load(dest.with_suffix('.npz')) as z, np.load(baseline.with_suffix('.npz')) as b:
    assert int(z['case']) == 1000
    assert z['model'].shape == (3, 4, 4, 2, 5)
    assert np.isfinite(z['model']).all()
    np.testing.assert_array_equal(z['cuts'], b['cuts'])
    np.testing.assert_array_equal(z['sim'], b['sim'])
    cached = CACHE / 'alignment_regularized_20261106/constant/nine_weak/axis1/case1000.npz'
    with np.load(cached) as old:
        for k, name in enumerate(z['cuts']):
            if name not in old['cuts']:
                continue
            j = list(old['cuts']).index(name)
            for leg, state in enumerate([0, 3]):
                np.testing.assert_allclose(z['sim'][0, state, k].sum(0),
                    old['sim'][leg, j].sum(0), rtol=2e-10, atol=2e-7)
                for mode in [0, 1]:
                    np.testing.assert_allclose(z['model'][mode, state, k].sum(0),
                        old['model'][mode, leg, j].sum(0), rtol=2e-10, atol=2e-7)
replay = json.loads((dest.parent / 'replay_case1000.json').read_text())
assert len(replay) == 6
for check in replay:
    assert max(abs(x) for x in check['mass_fractional_change']) < 1e-7
    assert check['max_per_leg_mean_change'] < 1e-7
    assert max(abs(x) for x in check['component_response_change'] + check['total_response_change']) < 5e-6
meta = {
    'case': 1000, 'arm': 'nine_weak', 'checkpoint': pin,
    'source_sha256': '28fa0c26d932eea37e46fbc63db0cc25198aca30586de3c4345c2e022cad26b9',
    'latent_source_sha256': bank['portable_source_sha256'],
    'measured': json.loads(baseline.with_suffix('.json').read_text())['measured'],
    'measured_receipt_provenance': 'Baseline on identical images; measured sums verified identical.',
    'coherent_replay_passed': True, 'cache_replay': replay,
    'recovery': {
        'failed_job': '17805334_1', 'recovery_job': os.environ['SLURM_JOB_ID'],
        'reason': 'Replay-loop variable shadowed device-verification metadata after all numerical checks and NPZ save.',
        'array_sha256': hashlib.sha256(dest.with_suffix('.npz').read_bytes()).hexdigest(),
        'checks': 'Finite arrays, identical measured sums, strict historical coherent replay, original physical bounds. Original execution reached NPZ save after strict same-device checks. Arrays unchanged.',
    },
}
receipt.write_text(json.dumps(meta, indent=2) + '\n')
print('RECOVERED metadata only; strict saved-output replay passed', flush=True)
