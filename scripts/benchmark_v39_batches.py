"""Replay the pinned 500k sampler at larger observation batches on a GPU worker."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_STUDY_ROOT'])
sys.path.insert(0, str(ROOT))
import run_v39_sampling_study as study  # noqa: E402

np, torch = study.np, study.torch
KEYS = ('score', 'information', 'draw_counts', 'unique_counts', 'ladder_score',
        'ladder_information', 'weight_ess', 'weight_max_fraction',
        'weight_relative_error', 'weight_pareto_k')


def save_profile(path, record):
    temporary = path.with_suffix('.tmp')
    study.write_json(temporary, record)
    temporary.replace(path)


def evaluate(likelihood, proposal, mock, receipt, ids, chunk, progress=None):
    window = study.MockCatalogue(mock.measurements.iloc[ids].reset_index(drop=True),
                                 mock.truth.iloc[ids].reset_index(drop=True))
    result = study.run_adaptive_section5(
        likelihood, window, proposal, center=receipt['center'], h=.001,
        draw_ladder=(512, 1024, 2048, 4096, 8192, 16384), n_candidates=1024,
        proposal_prefilter_candidates=1024, epsilon=.1, proposal_seed=8701,
        min_ess=32., max_weight_fraction=.5, candidate_backend='torch',
        estimator_mode='tilted_stratified', tilt_delta=.1, retain_full_ladder=True,
        full_information=True, object_ids=np.asarray(ids), object_chunk=chunk,
        atom_chunk=4096, progress=progress)
    return {key: getattr(result, key) for key in KEYS}


def compare(reference, actual):
    differences = {}
    for key in KEYS:
        a, b = reference[key], actual[key]
        np.testing.assert_allclose(a, b, rtol=1.e-9, atol=1.e-8, equal_nan=True,
                                   err_msg=key)
        finite = np.isfinite(a) & np.isfinite(b)
        differences[key] = float(np.max(np.abs(a[finite]-b[finite]), initial=0))
    return differences


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    gpu = torch.cuda.get_device_properties(0)
    output = ROOT/'batch_optimization'/f"benchmark_{os.environ['SLURM_JOB_ID']}.json"
    output.parent.mkdir(exist_ok=True)
    record = dict(status='running', execution=study.identity(),
                  script_sha256=study.file_hash(__file__), gpu=gpu.name,
                  total_memory=gpu.total_memory, job=os.environ['SLURM_JOB_ID'],
                  load_seconds=time.monotonic()-started, results=[],
                  tolerance=dict(rtol=1.e-9, atol=1.e-8))
    # Include the two rounded-boundary observations and retain their original RNG IDs.
    # Same V100 hardware as these original checkpoints, retaining each original
    # four-row group around both boundary cases to isolate batching from hardware.
    ids = np.concatenate([np.arange(3000, 3120), np.arange(86172, 86176),
                          np.arange(112152, 112156)])
    record['object_ids'] = ids.tolist()
    checkpoints = {p: dict(np.load(ROOT/f'inference/part_{p:02d}/distance_moments.npz'))
                   for p in sorted(set(ids//1000))}
    historical = {}
    for key in KEYS:
        rung_axis = key.startswith(('ladder_', 'weight_'))
        samples = []
        for row in ids:
            partition, offset = divmod(int(row), 1000)
            saved = checkpoints[partition]
            samples.append(saved[key][:, offset] if rung_axis else saved[key][offset])
        historical[key] = np.stack(samples, axis=1 if rung_axis else 0)
    traces = dict(candidates=[], draws=[])
    original_candidates, original_draw = proposal.candidates, proposal.draw_tilted

    def candidates(*args, **kwargs):
        value = original_candidates(*args, **kwargs)
        traces['candidates'].extend(hashlib.sha256(row.tobytes()).hexdigest()
                                    for row in value.indices)
        return value

    def draw(*args, **kwargs):
        value = original_draw(*args, **kwargs)
        traces['draws'].extend(hashlib.sha256(i.tobytes()+p.tobytes()).hexdigest()
                              for i, p in zip(value.indices, value.probability))
        return value

    proposal.candidates, proposal.draw_tilted = candidates, draw
    baseline, baseline_trace = None, None
    for chunk in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        traces = dict(candidates=[], draws=[])
        torch.cuda.synchronize()
        tick = time.monotonic()
        try:
            if baseline is not None:
                evaluate(likelihood, proposal, mock, receipt, ids[:chunk], chunk)
                torch.cuda.synchronize()
                traces = dict(candidates=[], draws=[])
                tick = time.monotonic()
            actual = evaluate(likelihood, proposal, mock, receipt, ids, chunk,
                progress=lambda done, total, elapsed: print(
                    f'BENCH chunk={chunk} rows={done}/{total} seconds={elapsed:.2f}', flush=True))
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tick
            if baseline is None:
                record['original_checkpoint_replay'] = compare(historical, actual)
                baseline, baseline_trace = actual, traces
                # Repeat baseline after compilation for an honest warm-time comparison.
                traces = dict(candidates=[], draws=[])
                tick = time.monotonic()
                actual = evaluate(likelihood, proposal, mock, receipt, ids, chunk)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-tick
            assert traces == baseline_trace, 'candidate or draw/probability replay changed'
            differences = compare(baseline, actual)
            result = dict(chunk=chunk, status='passed', seconds=elapsed,
                          observations_per_second=len(ids)/elapsed,
                          peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          max_absolute_difference=differences, exact_draw_replay=True)
        except torch.cuda.OutOfMemoryError as error:
            result = dict(chunk=chunk, status='out_of_memory', error=str(error))
        except AssertionError as error:
            result = dict(chunk=chunk, status='replay_failed', error=str(error))
        record['results'].append(result)
        save_profile(output, record)
        print(json.dumps(result), flush=True)
        if result['status'] == 'out_of_memory':
            break
        if chunk == 4 and result['status'] != 'passed':
            raise RuntimeError('baseline replay failed; investigate before changing batching')
        if result['status'] == 'passed' and result['peak_reserved'] > .85*gpu.total_memory:
            break
    valid = [r for r in record['results'] if r['status'] == 'passed'
             and r['peak_reserved'] <= .85*gpu.total_memory]
    assert valid, 'no replay-validated batch with memory headroom'
    fastest = max(valid, key=lambda r: r['observations_per_second'])
    record.update(status='complete', selected_chunk=fastest['chunk'],
                  selection_rule='fastest replay-passing batch with at least 15% memory headroom',
                  elapsed_seconds=time.monotonic()-started)
    save_profile(output, record)
    print(f'BENCHMARK_COMPLETE {output} selected_chunk={fastest["chunk"]}', flush=True)


if __name__ == '__main__':
    main()
