"""Persistent workers for a pinned study; checkpoint original 1,000-row parts."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_STUDY_ROOT'])
OPT = ROOT/'batch_optimization'
sys.path.insert(0, str(OPT))
import benchmark_v39_batches as bench  # noqa: E402
from v39_batched_proposal24 import BatchedProposal  # noqa: E402

study, np, torch = bench.study, bench.np, bench.torch


def amendment():
    record = study.read(OPT/'amendment.json')
    assert record['science_execution'] == study.identity()
    study.verify(__file__, record['worker_sha256'])
    study.verify(bench.__file__, record['benchmark_sha256'])
    study.verify(OPT/'validate_v39_batched_proposal24.py', record['validation_sha256'])
    study.verify(OPT/'v39_batched_proposal24.py', record['proposal_sha256'])
    parts = record['partitions']
    assert len(parts) == len(set(parts)) and all(0 <= p < 500 for p in parts)
    return record


def validate_profile(path, expected_sha=None):
    if expected_sha is not None:
        study.verify(path, expected_sha)
    profile = study.read(path)
    assert profile['status'] == 'complete'
    assert profile['execution'] == study.identity()
    assert profile['script_sha256'] == study.file_hash(OPT/'validate_v39_batched_proposal24.py')
    assert profile['proposal_script_sha256'] == study.file_hash(OPT/'v39_batched_proposal24.py')
    assert profile['benchmark_sha256'] == study.file_hash(bench.__file__)
    assert profile['accepted'] and profile['exact_draw_replay']
    assert profile['mode'] == 'batched_proposal_original_density'
    assert profile['density_chunk'] == 4 and profile['proposal_chunk'] == 128
    assert profile['tilt_chunk'] == 24
    assert profile['tolerance'] == dict(rtol=1.e-9, atol=1.e-8)
    assert profile['peak_allocated'] <= .85*profile['total_memory']
    study.verify(path.with_suffix('.npz'), profile['moments_sha256'])
    return profile


def choose_profile():
    gpu = torch.cuda.get_device_properties(0)
    record = amendment()
    filename = record['selected_validation_file']
    assert Path(filename).name == filename
    path = OPT/filename
    profile = validate_profile(path, record['selected_validation_sha256'])
    assert (profile['gpu'], profile['total_memory']) == (gpu.name, gpu.total_memory)
    return path, profile


def worker():
    record = amendment()
    profile_path, profile = choose_profile()
    chunk = profile['proposal_chunk']
    started = time.monotonic()
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    claims = OPT/'claims'
    claims.mkdir(exist_ok=True)
    runtime = dict(producer='persistent_worker', amendment_sha256=study.file_hash(OPT/'amendment.json'),
                   benchmark_file=profile_path.name, benchmark_sha256=study.file_hash(profile_path),
                   object_chunk=4, proposal_chunk=chunk, tilt_chunk=24,
                   gpu=profile['gpu'], job=os.environ['SLURM_JOB_ID'])
    processed = []
    for partition in record['partitions']:
        claim = claims/f'part_{partition:02d}'
        out = ROOT/f'inference/part_{partition:02d}'
        try:
            claim.mkdir()
        except FileExistsError:
            if out.exists() or not (claim/'owner.json').exists():
                continue
            owner = study.read(claim/'owner.json')
            if owner['job'] != os.environ['SLURM_JOB_ID']:
                continue
            # Slurm requeue retains the job ID. Only that same job may reclaim
            # its unpublished checkpoint; other live workers never steal it.
            if (claim/'output').exists():
                (claim/'output').rename(claim/f'incomplete_{time.time_ns()}')
        study.write_json(claim/'owner.tmp', runtime)
        (claim/'owner.tmp').replace(claim/'owner.json')
        # Existing outputs are never overwritten, including incomplete original jobs.
        assert not out.exists(), f'partition already exists: {out}'
        start, stop = 1000*partition, 1000*(partition+1)
        tick = time.monotonic()
        obs = mock.measurements.iloc[start:start+2]
        gaussian = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=131072, torch_device='cuda')
        distance = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=1024, torch_device='cuda')
        kwargs = dict(n_draws=16384, delta=.1, seed=8701, object_ids=np.arange(start, start+2), device='cuda')
        a, b = proposal.draw_tilted(gaussian, obs, **kwargs), proposal.draw_tilted(distance, obs, **kwargs)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.probability, b.probability)
        del a, b
        print(f'PERSISTENT_START part={partition} proposal_chunk={chunk} density_chunk=4 loaded_seconds={time.monotonic()-started:.2f}', flush=True)
        cached = BatchedProposal(study, proposal, mock.measurements.iloc[start:stop], np.arange(start, stop), chunk)
        arrays = bench.evaluate(likelihood, cached, mock, receipt, np.arange(start, stop), 4,
            progress=lambda done, total, elapsed: print(
                f'PERSISTENT part={partition} rows={done}/{total} seconds={elapsed:.2f}', flush=True))
        cached.complete()
        del cached
        arrays['object_rows'] = np.arange(start, stop)
        # Publish a checkpoint only after its arrays and receipt are complete.
        temporary = claim/'output'
        temporary.mkdir()
        np.savez_compressed(temporary/'distance_moments.npz', **arrays)
        study.write_json(temporary/'manifest.json', dict(status='complete', smoke=False,
            start=start, stop=stop, execution=record['science_execution'],
            mock_manifest_sha256=study.file_hash(ROOT/'mock/manifest.json'),
            assembled_manifest_sha256=study.file_hash(ROOT/'assembled/manifest.json'),
            same_draws_check=True, arms=['distance'], primary_rung=16384,
            output_sha256={'distance_moments.npz': study.file_hash(temporary/'distance_moments.npz')},
            elapsed_seconds=time.monotonic()-tick, runtime_amendment=runtime))
        temporary.rename(out)
        processed.append(partition)
        print(f'PERSISTENT_COMPLETE part={partition} seconds={time.monotonic()-tick:.2f}', flush=True)
    study.write_json(OPT/f"worker_{os.environ['SLURM_JOB_ID']}.json",
                     dict(status='complete', runtime=runtime, partitions=processed,
                          observations=1000*len(processed), elapsed_seconds=time.monotonic()-started))


def combine():
    record = amendment()
    expected_hash = study.file_hash(OPT/'amendment.json')
    profiles = {}
    for partition in range(500):
        receipt = study.read(ROOT/f'inference/part_{partition:02d}/manifest.json')
        runtime = receipt.get('runtime_amendment')
        if partition in record['partitions']:
            assert runtime and runtime['producer'] == 'persistent_worker'
            assert runtime['amendment_sha256'] == expected_hash
            filename = runtime['benchmark_file']
            assert Path(filename).name == filename
            assert filename == record['selected_validation_file']
            assert runtime['benchmark_sha256'] == record['selected_validation_sha256']
            key = (filename, runtime['benchmark_sha256'])
            if key not in profiles:
                profiles[key] = validate_profile(OPT/filename, key[1])
            profile = profiles[key]
            assert runtime['object_chunk'] == profile['density_chunk']
            assert runtime['proposal_chunk'] == profile['proposal_chunk']
            assert runtime['tilt_chunk'] == profile['tilt_chunk']
            assert runtime['gpu'] == profile['gpu']
        else:
            assert runtime is None, 'unexpected amended original partition'
    # The original combiner independently checks every hash, row, and science identity.
    study.combine()
    result = study.read(ROOT/'result.json')
    result['runtime_amendment'] = dict(amendment_sha256=expected_hash,
                                      optimized_partitions=record['partitions'],
                                      benchmark_files=[key[0] for key in profiles])
    study.write_json(ROOT/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.stage]()
