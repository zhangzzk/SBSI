"""Four post hoc cohort probes: exhaustive moments under both bright-prior weights."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_CURVATURE_ROOT'])
SOURCE = Path(os.environ['SBSI_COHORT_RECOVERY_ROOT'])
sys.path.insert(0, str(SOURCE))
import run_v39_cohort_recovery as cohort  # noqa: E402

domain, study = cohort.domain, cohort.study
np, torch, save = cohort.np, cohort.torch, cohort.save
PRIORS = cohort.PRIORS


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    cohort.protocol()
    for path, sha in p['sources'].items():
        study.verify(path, sha)
    for directory in set(str(Path(value).parent) for value in p.get('reused_groups', {}).values()):
        previous = study.read(Path(directory)/'protocol.json')
        for key in ('center', 'h', 'priors', 'source_root', 'probes', 'numerical', 'normalization'):
            assert previous[key] == p[key], key
    assert p['center'] == domain.protocol()['center']
    assert p['h'] == domain.protocol()['h']
    return p


def prepare():
    p = protocol()
    samples, _ = cohort.observations()
    result = study.read(SOURCE/'result.json')
    study.verify(SOURCE/'moments.npz', result['moments_sha256'])
    groups = []
    for name, focal in p['probes']:
        sample = samples[name]
        rows = np.arange(focal//4*4, focal//4*4+4)
        w = cohort.window(sample, rows)
        np.testing.assert_array_equal(w.truth.parent_sample_row, rows)
        valid = ~w.truth.numerical_padding.to_numpy()
        assert valid[focal % 4]
        assert (w.measurements.measured_mag_auto < 25.8).all()
        assert (w.measurements.measured_flux_radius > 3).all()
        item = dict(sample=name, focal_row=focal, rows=rows.tolist(), scientific=valid.tolist(),
            measured_magnitudes=w.measurements.measured_mag_auto.tolist())
        if name == 'actual':
            old = SOURCE/f'repair/actual/group_{focal//4:04d}'
            receipt = study.read(old/'manifest.json')
            study.verify(old/'expanded_center.npy', receipt['output_sha256']['expanded_center.npy'])
            item['center_reference'] = str(old/'expanded_center.npy')
        groups.append(item)
    save(ROOT/'cohort.json', dict(protocol_sha256=study.file_hash(ROOT/'protocol.json'), groups=groups))
    print('CURVATURE_PREPARED', groups, flush=True)


def worker(index):
    p = protocol()
    rec = study.read(ROOT/'cohort.json')
    assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    group = rec['groups'][index]
    samples, _ = cohort.observations()
    w = cohort.window(samples[group['sample']], np.asarray(group['rows']))
    likelihoods, proposal, ratio = cohort.load_pair()
    del proposal, samples
    gc.collect()
    points = [point for point in study.stencil(p['center'], p['h']) if point != (0., 0.)]
    assert len(points) == 9
    zero = points.index(tuple(np.round(p['center'], 14)))
    n = domain.protocol()['n_atoms']
    out = ROOT/f'group_{index}'
    out.mkdir(exist_ok=False)

    # Check reweighting against direct target densities, including zero-weight atoms.
    mixed = np.r_[np.flatnonzero(np.isfinite(ratio))[:8192], np.flatnonzero(~np.isfinite(ratio))[:8192]]
    assert len(mixed) == 16384
    for point in points:
        terms = {key:likelihood.log_importance_weights_tensor(w.measurements, *point,
            atom_indices=mixed, proposal_probability=np.ones(len(mixed)), object_chunk=4,
            atom_chunk=4096).cpu().numpy() for key, likelihood in likelihoods.items()}
        np.testing.assert_allclose(terms['expanded']+ratio[mixed], terms['corrected'], rtol=0, atol=1.e-7)
        assert np.isneginf(terms['corrected'][:, 8192:]).all()
    print('CURVATURE_WEIGHT_REPLAY_PASS', index, flush=True)

    reference = np.load(group['center_reference'], mmap_mode='r') if 'center_reference' in group else None
    if reference is not None:
        # The old centre-only scan used the unrounded centre; stencil keys use 14 decimals.
        # Reproduce the discrepancy in the failed interval before accepting this explanation.
        replay = dict(rounded_max_error=0., original_max_error=0., rounded_violations=0,
            rounded_center=list(points[zero]), original_center=p['center'])
        for start in range(*p['replay_probe_atom_range'], 16384):
            ids = np.arange(start, start+16384)
            expected = reference[:,start:start+16384]
            for label, point in [('rounded', points[zero]), ('original', p['center'])]:
                values = likelihoods['expanded'].log_importance_weights_tensor(w.measurements, *point,
                    atom_indices=ids, proposal_probability=np.ones(len(ids)),
                    object_chunk=4, atom_chunk=4096).cpu().numpy()
                np.testing.assert_array_equal(np.isfinite(values), np.isfinite(expected))
                finite = np.isfinite(values)
                delta = abs(values[finite]-expected[finite])
                replay[label+'_max_error'] = max(replay[label+'_max_error'], float(delta.max(initial=0)))
                if label == 'rounded':
                    replay['rounded_violations'] += int((delta > 1.e-7).sum())
                else:
                    np.testing.assert_allclose(values, expected, rtol=0, atol=1.e-7)
        save(out/'center_representation_check.json', replay)
        assert replay['rounded_violations'] > 0, 'The proposed centre-representation explanation did not reproduce the failure'
        print('CURVATURE_CENTER_REPRESENTATION_PASS', replay, flush=True)
    logs = None if reference is not None else np.lib.format.open_memmap(
        out/'expanded_center.npy', mode='w+', dtype=np.float64, shape=(4, n))
    sums = np.full((2, 9, 4), -np.inf)
    second, maximum = np.full_like(sums, -np.inf), np.full_like(sums, -np.inf)
    started, replay_error = time.monotonic(), 0.
    for start in range(0, n, 16384):
        stop = min(start+16384, n)
        ids = np.arange(start, stop)
        offset = torch.as_tensor(ratio[start:stop], device='cuda')
        for pi, point in enumerate(points):
            raw = likelihoods['expanded'].log_importance_weights_tensor(w.measurements, *point,
                atom_indices=ids, proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
            assert not torch.isnan(raw).any() and not torch.isposinf(raw).any()
            for ai, values in enumerate((raw, raw+offset)):
                sums[ai, pi] = np.logaddexp(sums[ai, pi], torch.logsumexp(values, dim=1).cpu().numpy())
                second[ai, pi] = np.logaddexp(second[ai, pi], torch.logsumexp(2*values, dim=1).cpu().numpy())
                maximum[ai, pi] = np.maximum(maximum[ai, pi], values.max(1).values.cpu().numpy())
            if pi == zero:
                values = raw.cpu().numpy()
                if reference is None:
                    logs[:, start:stop] = values
                else:
                    expected = reference[:, start:stop]
                    # Replay with the original exact float inputs. The scientific nine-node sums
                    # above still use the unchanged rounded stencil, as do the completed mocks.
                    replay_values = likelihoods['expanded'].log_importance_weights_tensor(w.measurements,
                        *p['center'], atom_indices=ids, proposal_probability=np.ones(len(ids)),
                        object_chunk=4, atom_chunk=4096).cpu().numpy()
                    np.testing.assert_allclose(replay_values, expected, rtol=0, atol=1.e-7)
                    finite = np.isfinite(replay_values)
                    replay_error = max(replay_error, float(np.max(abs(replay_values[finite]-expected[finite]), initial=0)))
        if start//16384 % 64 == 0 or stop == n:
            progress = dict(atoms=stop, total_atoms=n, elapsed_seconds=time.monotonic()-started)
            save(out/'progress.json', progress)
            print('CURVATURE_EXACT', index, progress, flush=True)
    if logs is not None:
        logs.flush()
    norms = np.asarray([[likelihoods[key].population_normalization.log_mass(*point)
        for point in points] for key in PRIORS])
    assert np.isfinite(sums).all() and np.isfinite(norms).all()
    np.savez_compressed(out/'exact.npz', points=np.asarray(points), log_numerator=sums,
        log_selected=sums-norms[:, :, None], log_normalization=norms,
        log_second=second, log_maximum=maximum)
    save(out/'manifest.json', dict(status='complete', group=group,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), weight_ratio_replay_passed=True,
        center_reference_replayed=reference is not None, maximum_center_replay_error=replay_error,
        elapsed_seconds=time.monotonic()-started,
        output_sha256={f.name:study.file_hash(f) for f in out.iterdir() if f.suffix in ('.npz', '.npy')}))
    print('CURVATURE_WORKER_COMPLETE', index, flush=True)


def combine():
    p = protocol()
    source = study.read(SOURCE/'result.json')
    study.verify(SOURCE/'moments.npz', source['moments_sha256'])
    samples, _ = cohort.observations()
    with np.load(SOURCE/'moments.npz') as f:
        saved = {key:f[key] for key in f.files}
    patched = {key:value.copy() for key, value in saved.items() if '_bright_repaired_' in key}
    reports = []
    changed = dict(actual=set(), mock=set())
    gate = domain.protocol()
    for index in range(len(p['probes'])):
        out = Path(p.get('reused_groups', {}).get(str(index), str(ROOT/f'group_{index}')))
        receipt = study.read(out/'manifest.json')
        assert receipt['status'] == 'complete' and receipt['weight_ratio_replay_passed']
        assert receipt['protocol_sha256'] == study.file_hash(out.parent/'protocol.json')
        assert receipt['group'] == study.read(ROOT/'cohort.json')['groups'][index]
        for name, sha in receipt['output_sha256'].items():
            study.verify(out/name, sha)
        group = receipt['group']
        with np.load(out/'exact.npz') as f:
            exact = {key:f[key] for key in f.files}
        zero = list(map(tuple, exact['points'])).index(tuple(np.round(p['center'], 14)))
        for ai, prior in enumerate(PRIORS):
            score, info = domain.moments(exact['points'], exact['log_selected'][ai])
            for j, row in enumerate(group['rows']):
                report = dict(sample=group['sample'], row=row, focal=row==group['focal_row'],
                    scientific=group['scientific'][j], measured_magnitude=group['measured_magnitudes'][j],
                    prior=prior, exact_score=score[j].tolist(), exact_information=info[j].tolist(),
                    exact_eigenvalues=np.linalg.eigvalsh(info[j]).tolist(),
                    center_distinct_atom_ess=float(np.exp(2*exact['log_numerator'][ai,zero,j]-exact['log_second'][ai,zero,j])),
                    center_max_atom_fraction=float(np.exp(exact['log_maximum'][ai,zero,j]-exact['log_numerator'][ai,zero,j])),
                    checks=[])
                for seed in gate['proposal_seeds']:
                    for arm in ('baseline', 'bright_repaired'):
                        prefix = f'{group["sample"]}_{arm}_{prior}_{seed}_ladder_'
                        for rung, draws in [(-2, 8192), (-1, 16384)]:
                            ds, di = saved[prefix+'score'][rung,row]-score[j], saved[prefix+'information'][rung,row]-info[j]
                            passed = bool(np.all(abs(ds) <= gate['score_atol']+gate['moment_rtol']*abs(score[j]))
                                and np.all(abs(di) <= gate['information_atol']+gate['moment_rtol']*abs(info[j])))
                            report['checks'].append(dict(seed=seed, arm=arm, draws=draws, passed=passed,
                                score_error=ds.tolist(), information_error=di.tolist()))
                    if group['scientific'][j]:
                        changed[group['sample']].add(row)
                        prefix = f'{group["sample"]}_bright_repaired_{prior}_{seed}_ladder_'
                        patched[prefix+'score'][:,row] = score[j]
                        patched[prefix+'information'][:,row] = info[j]
                reports.append(report)
    for key, value in patched.items():
        name = key.split('_')[0]
        untouched = ~np.isin(np.arange(value.shape[1]), sorted(changed[name]))
        np.testing.assert_array_equal(value[:,untouched], saved[key][:,untouched])
    totals = []
    retained = ~samples['mock'].truth.numerical_padding.to_numpy()
    for label, name, valid in [('actual_subset','actual',np.ones(len(retained),bool)),
            ('actual_same_ids','actual',retained),('mock','mock',retained)]:
        for prior in PRIORS:
            for seed in gate['proposal_seeds']:
                prefix = f'{name}_bright_repaired_{prior}_{seed}_ladder_'
                for rung, draws in [(-2,8192),(-1,16384)]:
                    arms, influences = {}, {}
                    for arm, data in [('source_bright_repaired',saved), ('partial_exact',patched)]:
                        s, h = data[prefix+'score'][rung,valid], data[prefix+'information'][rung,valid]
                        result, influences[arm] = domain.baseline.old.summarize(p['center'],s,h)
                        arms[arm] = dict(result, score_sum=s.sum(0).tolist(), information_sum=h.sum(0).tolist())
                    previous = next(x for x in source['reports'] if x['sample']==label and x['seed']==seed)
                    expected = previous['rungs'][rung]['arms'][prior+'_bright_repaired']
                    assert arms['source_bright_repaired']['status'] == expected['status']
                    for key in ('score_sum', 'information_sum', 'eigenvalues'):
                        np.testing.assert_array_equal(arms['source_bright_repaired'][key], expected[key])
                    totals.append(dict(sample=label, prior=prior, seed=seed, draws=draws, arms=arms,
                        paired=domain.baseline.old.paired(arms['source_bright_repaired'],arms['partial_exact'],
                            influences['source_bright_repaired'],influences['partial_exact'])))
    save(ROOT/'result.json', dict(status='complete', protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        reports=reports, partial_exact_totals=totals,
        exact_replacement_rows={name:sorted(rows) for name,rows in changed.items()},
        limitation='Post hoc diagnostic probes and incidental companions; exhaustive finite-prior numerator sums at h=.001, existing64-QMC normalization. Partial replacement only, not full-cohort closure or0.3% validation.'))
    print('CURVATURE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','worker','combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        worker(int(os.environ['SLURM_ARRAY_TASK_ID'])) if args.stage == 'worker' else globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID","cpu")}.json',
            dict(traceback=traceback.format_exc()))
        raise
