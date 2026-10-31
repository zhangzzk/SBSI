"""Test (a), owner request 2026-10-31: paired +/-0.02 model mock of the TRUE ConstGold galaxies, inferred exactly like the
measured 1M paired run (GalSBI F24 prior, symmetric F8 + C8 + disk, same 10.01M-scene normalization, same sampler
recipes, C1 at zero + C2 rerun of mag<21 at the rounded C1 shear, same estimator and pairing). Only the measurements
differ: model draws of the real galaxies (make_mock.py) instead of images. Cases 40-119 (80 cases).

actions (under Slurm): legs | preflight | estimate <leg> <c1|c2> | pair
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PY = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'
TEMPLATE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_minus_signsym_20261029')
EXTENSION = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028')
NORM = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold100k_signsym_20261027/preliminary_10m')
SNAPSHOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/cacheoff_validation_v1/source_snapshot')  # what the measured 1M legs ran
SEEDS = dict(plus=dict(c1=(2031000001, 2031500001), c2=(2031100001, 2031600001)),
             minus=dict(c1=(2032000001, 2032500001), c2=(2032100001, 2032600001)))


def load(path, name='study_dependency'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def legs():
    import numpy as np
    import pandas as pd
    units = load(EXTENSION/'study.py').units
    for leg in ('plus', 'minus'):
        H = ROOT/leg
        assert (H/'reference/result.json').exists()
        (H/'source_snapshot').symlink_to(SNAPSHOT)
        for stage in ('c1', 'c2'):
            (H/stage/'claims').mkdir(parents=True)
            (H/stage/'source_snapshot').symlink_to(SNAPSHOT)
        (H/'logs').mkdir()
        obs = pd.read_parquet(H/'reference/observations.parquet')
        mag = 30-2.5*np.log10(obs.measured_flux_from_mag_auto.to_numpy())
        ids = np.arange(len(obs))
        groups = dict(bright=ids[mag < 23], faint=ids[mag >= 23], bright21=ids[mag < 21])
        for name, values in groups.items():
            np.savetxt(H/(name+'_ids.txt'), values, fmt='%d')
        lines = units('bright', 'bright_ids.txt', len(groups['bright']), 640)
        lines += units('faint', 'faint_ids.txt', len(groups['faint']), 1024)
        (H/'c1/units.txt').write_text('\n'.join(lines)+'\n')
        (H/'c2/units.txt').write_text('\n'.join(units('bright', 'bright21_ids.txt', len(groups['bright21']), 61))+'\n')
        # Same worker and estimator as the measured minus leg; only the leg directory changes.
        for name in ('worker.sh', 'estimator_base.py', 'detect_resources.py'):
            text = (TEMPLATE/name).read_text()
            old = f"H = Path('{TEMPLATE}')" if name == 'estimator_base.py' else f'H={TEMPLATE}\n'
            new = f"H = Path('{H}')" if name == 'estimator_base.py' else f'H={H}\n'
            if name != 'detect_resources.py':
                assert text.count(old) == 1, name
                text = text.replace(old, new)
            if name == 'estimator_base.py':
                text = text.replace('100*q[0]/t[0]', '100*q[0]/abs(t[0])')
            (H/name).write_text(text)
        (H/'verify.py').write_text(
            'import hashlib, json, sys\nfrom pathlib import Path\n'
            'H = Path(__file__).resolve().parent\n'
            "r = json.loads((H/'reference/result.json').read_text())\n"
            "assert hashlib.sha256((H/'reference/observations.parquet').read_bytes()).hexdigest() == r['observations_sha256']\n"
            "print('LEG_REFERENCE_VERIFIED', H.name, r['observation_count'])\n")
        save(H/'reused_units.json', {})
        save(H/'reused_references.json', {})
        print('LEG_READY', leg, len(obs), {k: len(v) for k, v in groups.items()}, 'c1 units', len(lines), flush=True)
    save(ROOT/'rule.json', dict(plus_root=str(ROOT/'plus'), normalization=str(NORM), bootstrap_seed=2026103103,
        cases=[40, 120], seeds=SEEDS, template=str(TEMPLATE),
        question='Does the GalSBI prior alone produce the measured paired m = -5.5% on the ConstGold population?',
        reading='paired m ~ -6%: prior/population mismatch is the cause; ~ 0: the cause is model vs images',
        authorization='Owner 2026-10-31: run (a) with 4 GPUs'))


def preflight():
    out = ROOT/'preflight'
    out.mkdir()
    (out/'source_snapshot').symlink_to(SNAPSHOT)
    for leg in ('plus', 'minus'):
        H = ROOT/leg
        for group in ('bright', 'faint'):
            ids = (H/(group+'_ids.txt')).read_text().split()[:2]
            opts = ['--betas', '1', '.7883', '.6167', '.4449', '.2849', '.1473', '.05014', '.005',
                    '--narrow-step-probability', '.5'] if group == 'bright' else []
            label = f'{leg}_{group}'
            subprocess.run(['bash', str(SNAPSHOT/'jobs/job_galsbi_tempered.sh'), label, '--reference', str(H/'reference'),
                '--shear', '0', '0', '--seed', '2031900001', '--initial-bank-seed', '2031900002',
                '--observation-indices', *ids, '--steps', '64', '--warmup', '32', '--profile-stride', '16', *opts],
                env=dict(os.environ, SBSI_TEMPER_ROOT=str(out)), check=True)
            a = json.loads((out/label/'invariant_replay.json').read_text())
            assert json.loads((out/label/'protocol.json').read_text())['nuisance_likelihood_cache'] is False
            assert a['likelihood_calls_skipped'] == 0 and a['checked_states'] > 0
    save(ROOT/'preflight_passed.json', dict(status='passed', limitation='integration, not posterior convergence'))
    print('PREFLIGHT_PASSED', flush=True)


def estimate(leg, stage):
    H = ROOT/leg
    module = load(H/'estimator_base.py', 'estimator_'+leg)
    selection = load(NORM/'study.py', 'normalization')
    module.selection = lambda: selection.load_selection(NORM)
    sys.argv = [str(H/'estimator_base.py'), stage]
    module.main()


def pair():
    text = (TEMPLATE/'pair.py').read_text()
    for old, new in (('np.arange(40,140)', 'np.arange(40,120)'), ('np.zeros((100,2)),np.zeros((100,2,2))',
                     'np.zeros((80,2)),np.zeros((80,2,2))'), ('rng.integers(0,100,100)', 'rng.integers(0,80,80)')):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    (ROOT/'pair.py').write_text(text)
    (ROOT/'minus/rule.json').write_text((ROOT/'rule.json').read_text())
    load(ROOT/'pair.py', 'pair').run(ROOT/'minus')
    (ROOT/'paired_estimate.json').write_text((ROOT/'minus/paired_estimate.json').read_text())


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required')
    action = sys.argv[1]
    if action == 'legs':
        legs()
    elif action == 'preflight':
        preflight()
    elif action == 'estimate':
        estimate(sys.argv[2], sys.argv[3])
    elif action == 'pair':
        pair()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
