"""Four fixed magnitude20--24 probes; reuse existing sampler and exact-tail repair."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_INTERMEDIATE_ROOT'])
PARENT = Path(os.environ['SBSI_PARENT_INFERENCE_ROOT'])
sys.path.insert(0, str(PARENT))
import run_v39_parent_inference as parent  # noqa: E402

tail, domain, study = parent.tail, parent.domain, parent.study
np, torch, save = parent.np, parent.torch, parent.save


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    parent.protocol()
    for path, sha in p['sources'].items():
        study.verify(path, sha)
    return p


def prepare():
    p = protocol()
    fresh, rec = parent.observations()
    actual_input = Path(p['actual_root'])/'input'
    for name, sha in study.read(actual_input/'manifest.json')['output_sha256'].items():
        study.verify(actual_input/name, sha)
    actual = study.MockCatalogue.load(actual_input)
    assert len(actual.measurements) == 20000
    result = study.read(PARENT/'result.json')
    study.verify(PARENT/'moments.npz', result['moments_sha256'])
    valid = ~fresh.truth.numerical_padding.to_numpy()
    assert valid[:5130].all() and not valid[5130:].any()
    magnitude = fresh.measurements.measured_mag_auto.to_numpy()[:5130]
    effects = []
    with np.load(PARENT/'moments.npz') as f:
        for seed in parent.protocol()['seeds']:
            s, h = [f[f'bright_repaired_{seed}_ladder_{key}'] for key in ('score', 'information')]
            step = np.linalg.solve(h[-2].sum(0), s[-2].sum(0))
            effect = np.linalg.solve(h[-1].sum(0),
                ((s[-1]-s[-2])-np.einsum('nij,j->ni', h[-1]-h[-2], step)).T).T
            expected = np.linalg.solve(h[-1].sum(0), s[-1].sum(0))-step
            np.testing.assert_allclose(effect.sum(0), expected, rtol=0, atol=1.e-14)
            effects.append(effect[:, 0])
    priority = np.max(abs(np.asarray(effects)), axis=0)
    rng = np.random.default_rng(p['control_seed'])
    groups = []
    for label, sample in [('fresh_mock', fresh), ('actual', actual)]:
        mags = sample.measurements.measured_mag_auto.to_numpy()
        for low, high in [(20, 22), (22, 24)]:
            eligible = np.flatnonzero((mags >= low) & (mags < high))
            if label == 'fresh_mock':
                eligible = eligible[eligible < len(magnitude)]
                focal = int(eligible[np.argmax(priority[eligible])])
                selection = 'Largest absolute g1 step-change contribution over seeds8701--8703 within this measured-magnitude bin; post hoc diagnostic.'
            else:
                focal = int(rng.choice(eligible))
                selection = 'Uniform random control within measured-magnitude bin; no score or truth selection.'
            rows = np.arange(focal//4*4, focal//4*4+4)
            window = study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
                sample.truth.iloc[rows].reset_index(drop=True))
            if label == 'actual':
                window.truth['parent_sample_row'] = rows
            assert (window.measurements.measured_flux_radius > 3).all()
            assert (window.measurements.measured_mag_auto < 25.8).all()
            index = len(groups)
            folder = ROOT/f'group_{index}'
            folder.mkdir()
            window.save(folder/'input')
            groups.append(dict(index=index, sample=label, focal_row=focal, rows=rows.tolist(),
                measured_magnitude=float(mags[focal]), magnitude_bin=[low, high], selection=selection,
                object_ids=window.truth.parent_sample_row.to_numpy().tolist(),
                maximum_previous_absolute_g1_movement=float(priority[focal]) if label == 'fresh_mock' else None,
                input_sha256={f.name:study.file_hash(f) for f in (folder/'input').glob('*.parquet')}))
    assert len(groups) == 4 and rec['n_scientific'] == 5130
    save(ROOT/'cohort.json', dict(protocol_sha256=study.file_hash(ROOT/'protocol.json'), groups=groups))
    print('INTERMEDIATE_PREPARED', groups, flush=True)


def worker(index):
    protocol()
    p = domain.protocol()
    cohort = study.read(ROOT/'cohort.json')
    assert cohort['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    group = cohort['groups'][index]
    out = ROOT/f'group_{index}'
    for name, sha in group['input_sha256'].items():
        study.verify(out/'input'/name, sha)
    window = study.MockCatalogue.load(out/'input')
    np.testing.assert_array_equal(window.truth.parent_sample_row, group['object_ids'])
    likelihood, proposal = domain.load()
    sampled = {}
    for seed in p['proposal_seeds']:
        values = parent.evaluate(likelihood, proposal, window, seed)
        if group['sample'] == 'fresh_mock':
            source = PARENT/f'baseline/group_{group["focal_row"]//4:04d}.npz'
            study.verify(source, study.read(source.with_suffix('.json'))['output_sha256'])
            with np.load(source) as old:
                for key, value in values.items():
                    np.testing.assert_array_equal(value, old[f'{seed}_{key}'])
        sampled.update({f'baseline_{seed}_{key}':value for key, value in values.items()})
    np.savez_compressed(out/'baseline.npz', **sampled)
    print('INTERMEDIATE_BASELINE_COMPLETE', index, flush=True)

    points = [point for point in study.stencil(p['center'], p['h']) if point != (0., 0.)]
    assert len(points) == 9
    zero = points.index(tuple(np.round(p['center'], 14)))
    n, k = p['n_atoms'], parent.protocol()['k']
    logs = np.lib.format.open_memmap(out/'center.npy', mode='w+', dtype=np.float64, shape=(4, n))
    best = torch.full((4, k), -torch.inf, dtype=torch.float64, device='cuda')
    best_ids = torch.full((4, k), -1, dtype=torch.long, device='cuda')
    sums = np.full((9, 4), -np.inf)
    second, maximum = np.full_like(sums, -np.inf), np.full_like(sums, -np.inf)
    started = time.monotonic()
    for start in range(0, n, 16384):
        stop = min(start+16384, n)
        ids = np.arange(start, stop)
        for pi, point in enumerate(points):
            values = likelihood.log_importance_weights_tensor(window.measurements, *point,
                atom_indices=ids, proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
            assert not torch.isnan(values).any() and not torch.isposinf(values).any()
            sums[pi] = np.logaddexp(sums[pi], torch.logsumexp(values, dim=1).cpu().numpy())
            second[pi] = np.logaddexp(second[pi], torch.logsumexp(2*values, dim=1).cpu().numpy())
            maximum[pi] = np.maximum(maximum[pi], values.max(1).values.cpu().numpy())
            if pi == zero:
                logs[:, start:stop] = values.cpu().numpy()
                joined = torch.cat([best_ids, torch.as_tensor(ids, device='cuda').expand(4, -1)], dim=1)
                best, pos = torch.topk(torch.cat([best, values], dim=1), k, dim=1, sorted=True)
                best_ids = joined.gather(1, pos)
        if start//16384 % 64 == 0 or stop == n:
            elapsed = time.monotonic()-started
            save(out/'progress.json', dict(atoms=stop, total_atoms=n, elapsed_seconds=elapsed))
            print(f'INTERMEDIATE_EXACT worker={index} atoms={stop}/{n} seconds={elapsed:.1f}', flush=True)
    logs.flush()
    norms = np.asarray([likelihood.population_normalization.log_mass(*point) for point in points])
    np.savez_compressed(out/'exact.npz', points=np.asarray(points), log_numerator=sums,
        log_selected=sums-norms[:, None], log_second=second, log_maximum=maximum)
    atoms, scores = best_ids.cpu().numpy(), best.cpu().numpy()
    assert (atoms >= 0).all() and np.isfinite(scores).all()
    coverage_logs = []
    for pi, point in enumerate(points):
        values = likelihood.log_importance_weights_tensor(window.measurements, *point, atom_indices=atoms,
            proposal_probability=np.ones_like(atoms, float), object_chunk=4, atom_chunk=4096)
        if pi == zero:
            np.testing.assert_allclose(values.cpu().numpy(), scores, rtol=0, atol=1.e-7)
        coverage_logs.append(torch.logsumexp(values, dim=1).cpu().numpy())
    np.savez_compressed(out/'candidates.npz', atoms=atoms, center_log_contributions=scores,
        log_candidate_numerator=np.asarray(coverage_logs))
    weights = np.load(tail.PILOT/'weights.npy', mmap_mode='r')
    distributions = [tail.build_tail_distribution(logs[j], weights, atoms[j], delta=.1) for j in range(4)]
    distance = scores[:, :1]-scores
    candidates = domain.ProposalCandidates(atoms, distance, distance[:, -1])
    adapter = tail.ExactTailProposal(proposal, candidates, distributions, group['object_ids'])
    for seed in p['proposal_seeds']:
        values = parent.evaluate(likelihood, adapter, window, seed)
        sampled.update({f'repaired_{seed}_{key}':value for key, value in values.items()})
    np.savez_compressed(out/'sampled.npz', **sampled)
    save(out/'manifest.json', dict(status='complete', group=group,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), elapsed_seconds=time.monotonic()-started,
        center_replay_passed=True, baseline_replay_passed=group['sample']=='fresh_mock',
        output_sha256={f.name:study.file_hash(f) for f in out.iterdir() if f.suffix in ('.npz', '.npy')}))
    print('INTERMEDIATE_WORKER_COMPLETE', index, flush=True)


def combine():
    protocol()
    p, reports = domain.protocol(), []
    for index in range(4):
        out = ROOT/f'group_{index}'
        rec = study.read(out/'manifest.json')
        assert rec['status'] == 'complete' and rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        for name, sha in rec['output_sha256'].items():
            study.verify(out/name, sha)
        with np.load(out/'exact.npz') as f:
            exact = {key:f[key] for key in f.files}
        score, information = domain.moments(exact['points'], exact['log_selected'])
        zero = list(map(tuple, exact['points'])).index(tuple(np.round(p['center'], 14)))
        with np.load(out/'sampled.npz') as data, np.load(out/'candidates.npz') as candidates:
            for j, row in enumerate(rec['group']['rows']):
                report = dict(group=index, sample=rec['group']['sample'], row=row,
                    focal=row==rec['group']['focal_row'], exact_score=score[j].tolist(),
                    exact_information=information[j].tolist(),
                    center_distinct_atom_ess=float(np.exp(2*exact['log_numerator'][zero,j]-exact['log_second'][zero,j])),
                    center_max_atom_fraction=float(np.exp(exact['log_maximum'][zero,j]-exact['log_numerator'][zero,j])),
                    minimum_candidate_coverage=float(np.exp(candidates['log_candidate_numerator'][:,j]-exact['log_numerator'][:,j]).min()),
                    arms={})
                for arm in ('baseline', 'repaired'):
                    checks = []
                    for seed in p['proposal_seeds']:
                        for rung, draws in [(-2, 8192), (-1, 16384)]:
                            ds = data[f'{arm}_{seed}_ladder_score'][rung,j]-score[j]
                            di = data[f'{arm}_{seed}_ladder_information'][rung,j]-information[j]
                            passed = bool(np.all(abs(ds) <= p['score_atol']+p['moment_rtol']*abs(score[j]))
                                and np.all(abs(di) <= p['information_atol']+p['moment_rtol']*abs(information[j])))
                            checks.append(dict(seed=seed, draws=draws, score_error=ds.tolist(),
                                information_error=di.tolist(), passed=passed))
                    report['arms'][arm] = dict(checks=checks, passed=all(c['passed'] for c in checks))
                reports.append(report)
    save(ROOT/'result.json', dict(status='complete', reports=reports,
        numerical_pilot_pass={arm:all(r['arms'][arm]['passed'] for r in reports if r['focal'])
            for arm in ('baseline', 'repaired')},
        companion_all_pass={arm:all(r['arms'][arm]['passed'] for r in reports if not r['focal'])
            for arm in ('baseline', 'repaired')},
        limitation='Four fixed magnitude20--24 focal probes and twelve incidental companions; unchanged numerical gate. Not full-sample inference, finite-prior convergence, or 0.3% acceptance.'))
    print('INTERMEDIATE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID", "cpu")}.json',
            dict(traceback=traceback.format_exc()))
        raise
