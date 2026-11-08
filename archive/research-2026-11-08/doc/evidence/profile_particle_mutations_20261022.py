"""Profile frozen mutation calls; this is not an inference or speedup trial."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import runpy
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert digest(args.benchmark) == '9bff2aca28e450248b79f51397b46f6100ac96ec27f6dbb8c4364a7af618ae46'
    sys.path[:0] = [str(args.source.resolve()), str((args.source/'scripts').resolve())]
    import sbsi
    import torch
    from sbsi.galsbi_particle_target import GalSBIParticleTarget
    assert Path(sbsi.__file__).resolve().parent.parent == args.source.resolve()
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name()
    original = GalSBIParticleTarget.mutate
    profiles, counts = {}, {}

    def instrument(self, state, beta, rng, **kwargs):
        key = f"{kwargs['block']}_{len(state.x)}"
        profiler = profiles.setdefault(key, cProfile.Profile())
        torch.cuda.synchronize()
        profiler.enable()
        try:
            result = original(self, state, beta, rng, **kwargs)
            torch.cuda.synchronize()
            counts[key] = counts.get(key, 0)+1
            return result
        finally:
            profiler.disable()

    GalSBIParticleTarget.mutate = instrument
    sys.argv = [str(args.benchmark), '--source', str(args.source), '--run', str(args.run),
                '--output', str(args.output/'instrumented_benchmark.json'),
                '--warmup', '1', '--repetitions', '5', '--require-gpu-model', 'V100']
    status = 'failed'
    try:
        runpy.run_path(str(args.benchmark), run_name='__main__')
        status = 'completed'
    finally:
        GalSBIParticleTarget.mutate = original
        entries = {}
        for key, profiler in profiles.items():
            binary = args.output/f'{key}.prof'
            profiler.dump_stats(str(binary))
            with (args.output/f'{key}.txt').open('w') as handle:
                stats = pstats.Stats(profiler, stream=handle)
                stats.sort_stats('cumulative').print_stats(45)
                stats.sort_stats('tottime').print_stats(30)
            entries[key] = dict(calls=counts.get(key, 0), profile_sha256=digest(binary))
        receipt = dict(status=status, profiles=entries, script_sha256=digest(__file__),
            benchmark_sha256=digest(args.benchmark), source=str(args.source.resolve()),
            gpu=torch.cuda.get_device_name(), fixed_density_batch=4096,
            limitations=['Profiling changes timing; these are cost attribution clues, not speedup measurements.',
                'Warmup and five measured calls share each profile; native caches affect their costs.',
                'Each call restarts from saved beta=1 states; no posterior accuracy claim.',
                '4096 rows duplicate four observations; no larger observation grouping is validated.',
                'Frozen benchmark checks original source/input hashes and full retained-state fingerprints.'])
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
