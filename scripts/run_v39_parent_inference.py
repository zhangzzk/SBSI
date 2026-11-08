"""First-pass off-prior mock diagnostic with validated bright-tail integration."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_PARENT_INFERENCE_ROOT'])
TAIL = Path(os.environ['SBSI_EXACT_TAIL_ROOT'])
PARENT = Path(os.environ['SBSI_PARENT_MOCK_ROOT'])
sys.path.insert(0, str(TAIL))
import probe_v39_exact_tail as tail  # noqa: E402

domain = tail.domain
study, np, pd, torch, save = domain.study, domain.np, domain.pd, domain.torch, domain.save
KEYS = domain.baseline.old.KEYS


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    study.verify(TAIL/'result.json', p['tail_result_sha256'])
    study.verify(PARENT/'mock/manifest.json', p['parent_manifest_sha256'])
    tail.protocol()
    assert study.read(TAIL/'result.json')['numerical_pilot_pass']['16384']
    assert p['center'] == domain.protocol()['center'] and p['k'] == 16384
    return p


def initialize():
    p = protocol()
    # Large retained centre arrays are verified here on the scheduled CPU job.
    for index in range(4):
        rec = study.read(TAIL/f'worker_{index}.json')
        study.verify(TAIL/f'worker_{index}.npz', rec['output_sha256'])
        study.verify(TAIL/f'center_log_contributions_{index}.npy', rec['center_sha256'])
        assert rec['exact_center_replay_passed']
    source = study.read(PARENT/'mock/manifest.json')
    assert source['normal_forward_draw'] and not source['input_measurement_selection']
    assert not source['inserted_into_inference_prior'] and source['no_top_up']
    for name, sha in source['output_sha256'].items():
        study.verify(PARENT/'mock'/name, sha)
    mock = study.MockCatalogue.load(PARENT/'mock')
    n = len(mock.measurements)
    assert n == source['n_retained'] == 5130
    assert (mock.measurements.measured_mag_auto < 25.8).all()
    assert (mock.measurements.measured_flux_radius > 3.).all()
    measured, truth = mock.measurements.copy(), mock.truth.copy()
    truth['numerical_padding'] = False
    padding = (-n) % 4
    if padding:
        measured = pd.concat([measured, measured.iloc[[-1]*padding]], ignore_index=True)
        padded_truth = truth.iloc[[-1]*padding].copy()
        padded_truth['numerical_padding'] = True
        # Seed IDs for numerical padding cannot collide with scientific inputs.
        padded_truth['parent_sample_row'] = source['n_inputs']+np.arange(padding)
        truth = pd.concat([truth, padded_truth], ignore_index=True)
    assert truth.parent_sample_row.is_unique
    study.MockCatalogue(measured, truth).save(ROOT/'input')
    bright = (measured.measured_mag_auto.to_numpy() < 20) & ~truth.numerical_padding.to_numpy()
    groups = np.flatnonzero(bright.reshape(-1, 4).any(1))
    assert bright.sum() == source['n_bright'] == 44
    rec = dict(status='complete', n_scientific=n, n_numerical=len(measured), padding=padding,
        bright_rows=np.flatnonzero(bright).tolist(), bright_groups=groups.tolist(),
        all_group_count=len(measured)//4, center=p['center'],
        protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        output_sha256={f.name:study.file_hash(f) for f in (ROOT/'input').glob('*.parquet')})
    save(ROOT/'input/manifest.json', rec)
    print('PARENT_INFERENCE_INITIALIZED', rec, flush=True)


def observations():
    rec = study.read(ROOT/'input/manifest.json')
    assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    for name, sha in rec['output_sha256'].items():
        study.verify(ROOT/'input'/name, sha)
    return study.MockCatalogue.load(ROOT/'input'), rec


def evaluate(likelihood, proposal, window, seed):
    p = protocol()
    result = study.run_adaptive_section5(likelihood, window, proposal,
        center=p['center'], h=.001, draw_ladder=domain.baseline.old.LADDER,
        n_candidates=p['k'], proposal_prefilter_candidates=131072,
        proposal_seed=seed, candidate_backend='torch', estimator_mode='tilted_stratified',
        tilt_delta=.1, retain_full_ladder=True, full_information=True,
        object_ids=window.truth.parent_sample_row.to_numpy(np.int64), object_chunk=4, atom_chunk=4096)
    return {k:getattr(result, k) for k in KEYS}


def preflight():
    p = protocol()
    likelihood, proposal = domain.load()
    sample, _ = observations()
    window = study.MockCatalogue(sample.measurements.iloc[:8], sample.truth.iloc[:8])
    joint = evaluate(likelihood, proposal, window, p['seeds'][0])
    for i in range(2):
        w = study.MockCatalogue(window.measurements.iloc[4*i:4*i+4].reset_index(drop=True),
                               window.truth.iloc[4*i:4*i+4].reset_index(drop=True))
        separate = evaluate(likelihood, proposal, w, p['seeds'][0])
        for key in KEYS:
            expected = joint[key][:, 4*i:4*i+4] if key.startswith(('ladder_', 'weight_')) else joint[key][4*i:4*i+4]
            np.testing.assert_array_equal(separate[key], expected, err_msg=key)
    # Diagnose the seed-stable K1024 curvature residual without changing kernels.
    samples, _ = domain.baseline.observations()
    audit = []
    for index in (1, 3):
        old = study.read(tail.RANK/f'worker_{index}.json')
        study.verify(tail.RANK/f'worker_{index}.npz', old['output_sha256'])
        group = old['group']
        frame = samples[group['sample']].measurements.iloc[group['rows']].reset_index(drop=True)
        with np.load(tail.RANK/f'worker_{index}.npz') as f:
            atoms, points = f['atom_indices'], f['points']
            reference_small, reference_large = f['candidate_log_numerator_1024'], f['candidate_log_numerator_16384']
        small, large = [], []
        for point in points:
            for k, values in [(1024, small), (16384, large)]:
                terms = likelihood.log_importance_weights_tensor(frame, *point,
                    atom_indices=atoms[:, :k], proposal_probability=np.ones((4, k)),
                    object_chunk=4, atom_chunk=4096)
                values.append(torch.logsumexp(terms, dim=1).cpu().numpy())
        np.testing.assert_allclose(large, reference_large, rtol=0, atol=1.e-7)
        ds, di = domain.moments(points, np.array(small)-reference_small)
        j = group['focal_row'] % 4
        audit.append(dict(group=group, small_candidate_log_error=(np.array(small)-reference_small)[:,j].tolist(),
            small_candidate_score_error=ds[j].tolist(), small_candidate_information_error=di[j].tolist(),
            canonical_16384_replay=True))
    save(ROOT/'preflight.json', dict(status='passed', protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        original_four_row_group_replay=True, k=p['k'], smaller_batch_diagnostic=audit,
        gpu=torch.cuda.get_device_name(0)))
    print('PARENT_INFERENCE_PREFLIGHT_PASS', audit, flush=True)


def repair_group(likelihood, proposal, window, group):
    p = protocol()
    k, n = p['k'], domain.protocol()['n_atoms']
    out = ROOT/f'bright/group_{group:04d}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    logs = np.lib.format.open_memmap(out/'center.npy', mode='w+', dtype=np.float64, shape=(4, n))
    best = torch.full((4, k), -torch.inf, dtype=torch.float64, device='cuda')
    best_ids = torch.full((4, k), -1, dtype=torch.long, device='cuda')
    total = np.full(4, -np.inf)
    for start in range(0, n, 16384):
        ids = np.arange(start, min(start+16384, n))
        values = likelihood.log_importance_weights_tensor(window.measurements, *p['center'],
            atom_indices=ids, proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
        assert not torch.isnan(values).any() and not torch.isposinf(values).any()
        logs[:, start:start+len(ids)] = values.cpu().numpy()
        total = np.logaddexp(total, torch.logsumexp(values, dim=1).cpu().numpy())
        joined_ids = torch.cat([best_ids, torch.as_tensor(ids, device='cuda').expand(4, -1)], dim=1)
        best, pos = torch.topk(torch.cat([best, values], dim=1), k, dim=1, sorted=True)
        best_ids = joined_ids.gather(1, pos)
        if start//16384 % 256 == 0 or start+len(ids) == n:
            print(f'PARENT_BRIGHT_SCAN group={group} atoms={start+len(ids)}/{n} seconds={time.monotonic()-started:.1f}', flush=True)
    logs.flush()
    atoms, scores = best_ids.cpu().numpy(), best.cpu().numpy()
    assert (atoms >= 0).all() and np.isfinite(scores).all()
    replay = likelihood.log_importance_weights_tensor(window.measurements, *p['center'],
        atom_indices=atoms, proposal_probability=np.ones_like(atoms, float), object_chunk=4, atom_chunk=4096)
    np.testing.assert_allclose(replay.cpu().numpy(), scores, rtol=0, atol=1.e-7)
    weights = np.load(tail.PILOT/'weights.npy', mmap_mode='r')
    distributions = [tail.build_tail_distribution(logs[j], weights, atoms[j], delta=.1) for j in range(4)]
    distance = scores[:, :1]-scores
    candidates = domain.ProposalCandidates(atoms, distance, distance[:, -1])
    adapter = tail.ExactTailProposal(proposal, candidates, distributions,
                                    window.truth.parent_sample_row.to_numpy(np.int64))
    values = {}
    for seed in p['seeds']:
        result = evaluate(likelihood, adapter, window, seed)
        values.update({f'{seed}_{key}':value for key, value in result.items()})
    np.savez_compressed(out/'moments.npz', **values)
    np.savez_compressed(out/'candidates.npz', atoms=atoms, center_log_contributions=scores,
                        center_log_numerator=total)
    save(out/'manifest.json', dict(status='complete', group=group,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), elapsed_seconds=time.monotonic()-started,
        output_sha256={f.name:study.file_hash(f) for f in out.iterdir() if f.suffix in ('.npz','.npy')}))
    print(f'PARENT_BRIGHT_COMPLETE group={group} seconds={time.monotonic()-started:.1f}', flush=True)


def worker(index):
    p = protocol()
    check = study.read(ROOT/'preflight.json')
    assert check['status'] == 'passed' and check['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    likelihood, proposal = domain.load()
    sample, rec = observations()
    groups = np.arange(index, rec['all_group_count'], 4)
    # Finish and checkpoint the baseline first, then the expensive bright scans.
    for position, group in enumerate(groups):
        rows = np.arange(4*group, 4*group+4)
        window = study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
                                     sample.truth.iloc[rows].reset_index(drop=True))
        values = {}
        for seed in p['seeds']:
            values.update({f'{seed}_{key}':v for key,v in evaluate(likelihood, proposal, window, seed).items()})
        path = ROOT/f'baseline/group_{group:04d}.npz'
        np.savez_compressed(path, **values)
        save(path.with_suffix('.json'), dict(group=int(group), output_sha256=study.file_hash(path)))
        if position % 8 == 0 or position+1 == len(groups):
            print(f'PARENT_BASELINE worker={index} groups={position+1}/{len(groups)}', flush=True)
    save(ROOT/f'baseline_worker_{index}.json', dict(status='complete', groups=groups.tolist(),
        protocol_sha256=study.file_hash(ROOT/'protocol.json')))
    for group in rec['bright_groups']:
        if group % 4 != index:
            continue
        rows = np.arange(4*group, 4*group+4)
        window = study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
                                     sample.truth.iloc[rows].reset_index(drop=True))
        repair_group(likelihood, proposal, window, group)
        gc.collect()
        torch.cuda.empty_cache()
    save(ROOT/f'worker_{index}.json', dict(status='complete', protocol_sha256=study.file_hash(ROOT/'protocol.json')))
    print('PARENT_INFERENCE_WORKER_COMPLETE', index, flush=True)


def combine():
    p = protocol()
    sample, rec = observations()
    for i in range(4):
        w = study.read(ROOT/f'worker_{i}.json')
        assert w['status'] == 'complete' and w['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    baseline = {seed:{key:[] for key in ('ladder_score','ladder_information')} for seed in p['seeds']}
    for group in range(rec['all_group_count']):
        path = ROOT/f'baseline/group_{group:04d}.npz'
        study.verify(path, study.read(path.with_suffix('.json'))['output_sha256'])
        with np.load(path) as f:
            for seed in p['seeds']:
                for key in baseline[seed]:
                    baseline[seed][key].append(f[f'{seed}_{key}'])
    baseline = {seed:{key:np.concatenate(a, axis=1) for key,a in v.items()} for seed,v in baseline.items()}
    repaired = {seed:{key:a.copy() for key,a in v.items()} for seed,v in baseline.items()}
    for group in rec['bright_groups']:
        folder = ROOT/f'bright/group_{group:04d}'
        m = study.read(folder/'manifest.json')
        study.verify(folder/'moments.npz', m['output_sha256']['moments.npz'])
        rows = np.arange(4*group, 4*group+4)
        local = np.flatnonzero(np.isin(rows, rec['bright_rows']))
        with np.load(folder/'moments.npz') as f:
            for seed in p['seeds']:
                for key in repaired[seed]:
                    repaired[seed][key][:, rows[local]] = f[f'{seed}_{key}'][:, local]
    reports = []
    valid = ~sample.truth.numerical_padding.to_numpy()
    assert int(valid.sum()) == rec['n_scientific']
    untouched = ~np.isin(np.arange(len(valid)), rec['bright_rows'])
    for seed in p['seeds']:
        for key in repaired[seed]:
            np.testing.assert_array_equal(repaired[seed][key][:, untouched], baseline[seed][key][:, untouched])
    for seed in p['seeds']:
        rungs, influences = [], {}
        for rung, draws in enumerate(domain.baseline.old.LADDER):
            row = dict(draws=draws)
            for label, data in [('baseline',baseline[seed]),('bright_repaired',repaired[seed])]:
                score, info = data['ladder_score'][rung,valid], data['ladder_information'][rung,valid]
                result, influences[rung,label] = domain.baseline.old.summarize(p['center'], score, info)
                row[label] = dict(result, score_sum=score.sum(0).tolist(), information_sum=info.sum(0).tolist())
            row['paired_repair'] = domain.baseline.old.paired(row['baseline'],row['bright_repaired'],
                influences[rung,'baseline'],influences[rung,'bright_repaired'])
            rungs.append(row)
        change = {label:domain.baseline.old.paired(rungs[-2][label],rungs[-1][label],
            influences[4,label],influences[5,label]) for label in ('baseline','bright_repaired')}
        reports.append(dict(seed=seed,rungs=rungs,paired_8192_to_16384=change))
    np.savez_compressed(ROOT/'moments.npz', **{
        f'{label}_{seed}_{key}':a[:,valid] for label, data in [('baseline',baseline),('bright_repaired',repaired)]
        for seed, values in data.items() for key,a in values.items()})
    save(ROOT/'result.json', dict(status='complete', n_observations=rec['n_scientific'],n_bright=len(rec['bright_rows']),
        center=p['center'], reports=reports, protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        moments_sha256=study.file_hash(ROOT/'moments.npz'),
        limitation='One fixed-centre Newton diagnostic, not converged inference, independent validation, or 0.3% acceptance. Only measured-mag<20 integration is repaired; other rows and finite-prior resolution remain to be assessed.'))
    print('PARENT_INFERENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['initialize','preflight','worker','combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID","cpu")}.json',
             dict(traceback=traceback.format_exc()))
        raise
