"""Held-out paired check of batching roundoff at the shear-estimator scale."""
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_STUDY_ROOT'])
OPT = ROOT/'batch_optimization'
sys.path.insert(0, str(OPT))
import benchmark_v39_batches as bench  # noqa: E402

study, np, torch = bench.study, bench.np, bench.torch


def read_rows(ids):
    parts = {p: dict(np.load(ROOT/f'inference/part_{p:02d}/distance_moments.npz'))
             for p in sorted(set(ids//1000))}
    result = {}
    for key in bench.KEYS:
        rung = key.startswith(('ladder_', 'weight_'))
        rows = [parts[int(row)//1000][key][:, int(row)%1000] if rung else
                parts[int(row)//1000][key][int(row)%1000] for row in ids]
        result[key] = np.stack(rows, axis=1 if rung else 0)
    return result


def main():
    torch.set_num_threads(4)
    chunk = int(os.environ['SBSI_VALIDATION_CHUNK'])
    gpu = torch.cuda.get_device_properties(0)
    assert gpu.name == 'Tesla V100-SXM2-32GB'
    # Fresh observations, separate from the 128-row speed benchmark. Preserve
    # original four-row groups, plus oversample both known boundary groups.
    ids = np.concatenate([np.arange(234000, 235024), np.arange(86172, 86176),
                          np.arange(112152, 112156)])
    original = read_rows(ids)
    total_score, total_info, n = np.zeros(2), np.zeros((2, 2)), 0
    for partition in range(238):
        with np.load(ROOT/f'inference/part_{partition:02d}/distance_moments.npz') as data:
            total_score += data['score'].sum(0)
            total_info += data['information'].sum(0)
            n += len(data['score'])
    assert np.linalg.eigvalsh(total_info).min() > 0
    hbar, sbar = total_info/n, total_score/n
    step = np.linalg.solve(hbar, sbar)
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    baseline = bench.evaluate(likelihood, proposal, mock, receipt, ids[:8], 4)
    bench.compare(read_rows(ids[:8]), baseline)
    # Warm up the larger graph before timing, including all its row positions.
    bench.evaluate(likelihood, proposal, mock, receipt, ids[:chunk], chunk)
    traces = dict(candidates=[], draws=[])
    old_candidates, old_draw = proposal.candidates, proposal.draw_tilted

    def candidates(*args, **kwargs):
        value = old_candidates(*args, **kwargs)
        traces['candidates'].extend(hashlib.sha256(row.tobytes()).hexdigest() for row in value.indices)
        return value

    def draw(*args, **kwargs):
        value = old_draw(*args, **kwargs)
        traces['draws'].extend(hashlib.sha256(i.tobytes()+p.tobytes()).hexdigest()
                              for i, p in zip(value.indices, value.probability))
        return value

    proposal.candidates, proposal.draw_tilted = candidates, draw
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    actual = bench.evaluate(likelihood, proposal, mock, receipt, ids, chunk,
        progress=lambda done, total, elapsed: print(
            f'VALIDATION chunk={chunk} rows={done}/{total} seconds={elapsed:.2f}', flush=True))
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    peak = torch.cuda.max_memory_reserved()
    actual_trace = traces
    traces = dict(candidates=[], draws=[])
    for start in range(0, len(ids), 4):
        selected = ids[start:start+4]
        obs = mock.measurements.iloc[selected].reset_index(drop=True)
        local = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=1024, torch_device='cuda')
        proposal.draw_tilted(local, obs, n_draws=16384, delta=.1, seed=8701,
                             object_ids=selected, device='cuda')
    assert traces == actual_trace, 'candidate membership/order or draws/probabilities changed'
    for key in ('draw_counts', 'unique_counts'):
        np.testing.assert_array_equal(actual[key], original[key])
    for key in bench.KEYS:
        np.testing.assert_array_equal(np.isfinite(actual[key]), np.isfinite(original[key]))
    ds, dh = actual['score']-original['score'], actual['information']-original['information']
    influence = np.linalg.solve(hbar, (ds-np.einsum('nij,j->ni', dh, step)).T).T
    ordinary_fraction = 1-8/500000
    mean = ordinary_fraction*influence[:1024].mean(0)+influence[1024:].sum(0)/500000
    se = ordinary_fraction*influence[:1024].std(0, ddof=1)/np.sqrt(1024)
    mean_ds = ordinary_fraction*ds[:1024].mean(0)+ds[1024:].sum(0)/500000
    mean_dh = ordinary_fraction*dh[:1024].mean(0)+dh[1024:].sum(0)/500000
    exact_shift = np.linalg.solve(hbar+mean_dh, sbar+mean_ds)-step
    upper_pp = 5000*(np.abs(exact_shift)+2*se)
    # Predeclared before this held-out run: at most 1% of the 0.3% bias target,
    # including a two-standard-error margin in both components. This is a
    # numerical replay criterion, not a new calibration/closure acceptance.
    passed = bool(np.all(upper_pp < .003) and peak <= .85*gpu.total_memory)
    report = dict(status='complete', accepted=passed, chunk=chunk, gpu=gpu.name,
        total_memory=gpu.total_memory, peak_reserved=peak, seconds=elapsed,
        observations_per_second=len(ids)/elapsed, exact_draw_replay=True,
        execution=study.identity(), script_sha256=study.file_hash(__file__),
        benchmark_sha256=study.file_hash(bench.__file__), job=os.environ['SLURM_JOB_ID'],
        object_ids=ids.tolist(), reference_observations=n,
        reference_information_per_observation=hbar.tolist(), reference_step=step.tolist(),
        estimated_shift_pp=(5000*exact_shift).tolist(), paired_standard_error_pp=(5000*se).tolist(),
        upper_two_se_pp=upper_pp.tolist(), acceptance_limit_pp=.003,
        linearized_shift_pp=(5000*mean).tolist(),
        known_boundary_group_shift_pp=(5000*influence[1024:].sum(0)/500000).tolist(),
        max_absolute_difference={key: float(np.max(np.abs(actual[key][np.isfinite(actual[key])]-
            original[key][np.isfinite(original[key])]), initial=0)) for key in bench.KEYS},
        interpretation='Paired numerical batching check; finite-sample evidence, not exact derivative replay or scientific bias acceptance.')
    path = OPT/f"validation_{os.environ['SLURM_JOB_ID']}.json"
    np.savez_compressed(path.with_suffix('.npz'), object_rows=ids, **actual)
    report['moments_sha256'] = study.file_hash(path.with_suffix('.npz'))
    study.write_json(path, report)
    print({key: report[key] for key in ('accepted', 'chunk', 'observations_per_second',
        'peak_reserved', 'estimated_shift_pp', 'upper_two_se_pp')}, flush=True)


if __name__ == '__main__':
    main()
