"""Require strict replay for batched proposals with original density grouping."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_STUDY_ROOT'])
OPT = ROOT/'batch_optimization'
sys.path.insert(0, str(OPT))
import validate_v39_batch_effect as validation  # noqa: E402
from v39_batched_proposal24 import BatchedProposal  # noqa: E402

bench, study, np, torch = validation.bench, validation.study, validation.np, validation.torch


def main():
    torch.set_num_threads(4)
    gpu = torch.cuda.get_device_properties(0)
    assert gpu.name == 'Tesla V100-SXM2-32GB'
    ids = np.concatenate([np.arange(234000, 235024), np.arange(86172, 86176),
                          np.arange(112152, 112156)])
    reference = validation.read_rows(ids)
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    warm = bench.evaluate(likelihood, proposal, mock, receipt, ids[:8], 4)
    bench.compare(validation.read_rows(ids[:8]), warm)
    likelihood._response_cache.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    cached = BatchedProposal(study, proposal, mock.measurements.iloc[ids], ids, 128)
    actual = bench.evaluate(likelihood, cached, mock, receipt, ids, 4,
        progress=lambda done, total, elapsed: print(
            f'BATCHED_PROPOSAL rows={done}/{total} seconds={elapsed:.2f}', flush=True))
    cached.complete()
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    difference = bench.compare(reference, actual)
    # Check proposal records independently against the original four-row calls.
    for start in range(0, len(ids), 4):
        selected = ids[start:start+4]
        obs = mock.measurements.iloc[selected].reset_index(drop=True)
        local = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=1024, torch_device='cuda')
        draw = proposal.draw_tilted(local, obs, n_draws=16384, delta=.1, seed=8701,
                                    object_ids=selected, device='cuda')
        np.testing.assert_array_equal(local.indices, cached.cached_candidates.indices[start:start+4])
        np.testing.assert_array_equal(draw.indices, cached.cached_draw.indices[start:start+4])
        np.testing.assert_array_equal(draw.probability, cached.cached_draw.probability[start:start+4])
    report = dict(status='complete', accepted=True, mode='batched_proposal_original_density',
        proposal_chunk=128, tilt_chunk=24, density_chunk=4, gpu=gpu.name, total_memory=gpu.total_memory,
        peak_reserved=torch.cuda.max_memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
        seconds=elapsed, observations_per_second=len(ids)/elapsed,
        proposal_seconds=cached.elapsed_seconds, exact_draw_replay=True,
        max_absolute_difference=difference, execution=study.identity(), object_ids=ids.tolist(),
        tolerance=dict(rtol=1.e-9, atol=1.e-8), script_sha256=study.file_hash(__file__),
        proposal_script_sha256=study.file_hash(OPT/'v39_batched_proposal24.py'),
        benchmark_sha256=study.file_hash(bench.__file__), job=os.environ['SLURM_JOB_ID'])
    report['accepted'] = report['peak_allocated'] <= .85*gpu.total_memory
    path = OPT/f"proposal_validation_{os.environ['SLURM_JOB_ID']}.json"
    np.savez_compressed(path.with_suffix('.npz'), object_rows=ids, **actual)
    report['moments_sha256'] = study.file_hash(path.with_suffix('.npz'))
    study.write_json(path, report)
    print({key: report[key] for key in ('accepted', 'observations_per_second',
        'peak_allocated', 'peak_reserved', 'max_absolute_difference')}, flush=True)


if __name__ == '__main__':
    main()
