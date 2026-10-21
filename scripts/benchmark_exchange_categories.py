"""CPU-only fixed-input categorical exchange replay and paired timings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_exchange_categorical_replay import make_scene, load_baseline, assert_scene_equal, assert_rng_equal
from sbsi.scene_exchange import propose_native_exchange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = ['sbsi/scene_exchange.py', 'doc/evidence/scene_exchange_baseline_55f1220.py',
             'tests/test_exchange_categorical_replay.py', 'scripts/benchmark_exchange_categories.py']
    hashes = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    baseline = load_baseline()
    report = dict(numpy_version=np.__version__, seed=2026103200, warmup=2, repetitions=11,
        source=str(ROOT), sha256=hashes, cases=[], all_bitwise_replay=True,
        primary_metric='Geometric mean baseline/candidate proposal speed at ordinary 1024 and4096 rows without relocation',
        no_gpu_or_scientific_efficiency_claim=True)
    primary = []
    for size in (1024, 4096):
        patterns = dict(empty=[0], singleton=[1], ordinary=[0,1,2,4,8,16], many=[0,1,17,64], support_zero=[0,1,2,4,8,16])
        for pattern_index, (name, pattern) in enumerate(patterns.items()):
            counts = np.resize(pattern, size)
            prior, scene = make_scene(counts, support_zero=name == 'support_zero')
            probability = np.random.default_rng(482).dirichlet([.3,.7], size=size)
            probability[::7] = [0.,1.]
            for relocation in (False, True):
                options = dict(radius_scale=1.)
                if relocation:
                    options.update(relocation_scale=.1, aperture_arcsec=10.)
                timings = {'baseline':[], 'candidate':[]}
                for repetition in range(-2, 11):
                    results, randoms = {}, {}
                    for label in (('baseline','candidate') if repetition % 2 else ('candidate','baseline')):
                        fn = baseline.propose_native_exchange if label == 'baseline' else propose_native_exchange
                        rng = np.random.default_rng(2026103200+100*pattern_index+repetition+2)
                        started = time.perf_counter()
                        result = fn(prior, scene, probability, rng, **options)
                        elapsed = time.perf_counter()-started
                        results[label], randoms[label] = result, rng
                        if repetition >= 0:
                            timings[label].append(elapsed)
                    a, b = results['baseline'], results['candidate']
                    assert_scene_equal(a[0], b[0])
                    assert a[1].tobytes() == b[1].tobytes() and a[2].tobytes() == b[2].tobytes()
                    assert_rng_equal(randoms['baseline'], randoms['candidate'])
                medians = {k:float(np.median(v)) for k,v in timings.items()}
                speedup = medians['baseline']/medians['candidate']
                entry = dict(rows=size, pattern=name, relocation=relocation, active=int((counts>0).sum()),
                    neighbours=int(counts.sum()), seconds=timings, median_seconds=medians, speedup=speedup,
                    replay_pass=True)
                report['cases'].append(entry)
                if name == 'ordinary' and not relocation:
                    primary.append(speedup)
                print(json.dumps({k:v for k,v in entry.items() if k != 'seconds'}), flush=True)
    report['dev_score'] = float(np.exp(np.log(primary).mean()))
    report['substantive_gain'] = report['dev_score'] >= 1.1
    for n,h in hashes.items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h, n
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
