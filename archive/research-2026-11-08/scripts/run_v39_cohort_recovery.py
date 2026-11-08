"""Fixed original cohorts: separate prior correction from bright integration repair."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_COHORT_RECOVERY_ROOT'])
PARENT = Path(os.environ['SBSI_PARENT_INFERENCE_ROOT'])
sys.path.insert(0, str(PARENT))
import run_v39_parent_inference as parent  # noqa: E402

tail, domain, study = parent.tail, parent.domain, parent.study
np, torch, save = parent.np, parent.torch, parent.save
PRIORS = ('expanded', 'corrected')
KEYS = domain.baseline.old.KEYS
MOMENTS = ('ladder_score', 'ladder_information')


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    parent.protocol()
    for path, sha in p['sources'].items():
        study.verify(path, sha)
    gate = study.read(Path(p['intermediate_root'])/'result.json')
    assert gate['numerical_pilot_pass']['repaired'] and gate['companion_all_pass']['repaired']
    return p


def prepare():
    p = protocol()
    n_rows = p.get('first_rows', 20000)
    assert 4 <= n_rows <= 20000 and n_rows % 4 == 0
    source = Path(p['baseline_source_root']) if 'baseline_source_root' in p else None
    if source is not None:
        previous = study.read(source/'protocol.json')
        study.verify(source/'run_v39_cohort_recovery.py', previous['script_sha256'])
        for key in ('center', 'proposal_seeds', 'draw_ladder', 'k', 'h', 'priors',
                    'chunk_rows', 'baseline_proposal', 'normalization', 'repair_measured_mag_lt'):
            assert p[key] == previous[key], key
        previous_gate = study.read(source/'preflight.json')
        assert previous_gate['status'] == 'passed'
        assert previous_gate['protocol_sha256'] == study.file_hash(source/'protocol.json')
    matched = domain.baseline.matched
    actual_input = matched.actual_root()/'input'
    for name, sha in study.read(actual_input/'manifest.json')['output_sha256'].items():
        study.verify(actual_input/name, sha)
    for name, sha in study.read(domain.baseline.MATCHED/'mock/manifest.json')['output_sha256'].items():
        if name.startswith('evaluation/'):
            study.verify(domain.baseline.MATCHED/'mock'/name, sha)
    samples, gen = domain.baseline.observations()
    records, chunks, repairs = {}, [], []
    for name, sample in samples.items():
        assert len(sample.measurements) == 20000
        np.testing.assert_array_equal(sample.truth.parent_sample_row, np.arange(20000))
        if 'numerical_padding' not in sample.truth:
            sample.truth['numerical_padding'] = False
        valid = ~sample.truth.numerical_padding.to_numpy()
        assert int(valid.sum()) == (16314 if name == 'mock' else 20000)
        if name == 'mock':
            np.testing.assert_array_equal(np.flatnonzero(valid), gen['selected_parent_rows'])
        sample = window(sample, np.arange(n_rows))
        valid = valid[:n_rows]
        assert (sample.measurements.measured_flux_radius > 3).all()
        assert (sample.measurements.measured_mag_auto < 25.8).all()
        folder = ROOT/f'input/{name}'
        sample.save(folder)
        bright = valid & (sample.measurements.measured_mag_auto.to_numpy() < 20)
        groups = np.unique(np.flatnonzero(bright)//4)
        records[name] = dict(n_scientific=int(valid.sum()), n_padding=int((~valid).sum()),
            bright_rows=np.flatnonzero(bright).tolist(), bright_groups=groups.tolist(),
            output_sha256={f.name:study.file_hash(f) for f in folder.glob('*.parquet')})
        chunks.extend(dict(sample=name, start=lo, stop=min(lo+p['chunk_rows'], n_rows))
            for lo in range(0, n_rows, p['chunk_rows']))
        repairs.extend(dict(sample=name, group=int(group)) for group in groups)
    if n_rows == 20000:
        assert len(records['actual']['bright_rows']) == 135 and len(records['mock']['bright_rows']) == 134
        assert len(repairs) == 266
    if source is not None:
        for task in chunks:
            relative = Path(f'baseline/{task["sample"]}/chunk_{task["start"]:05d}.npz')
            receipt = study.read((source/relative).with_suffix('.json'))
            assert receipt['task'] == task
            study.verify(source/relative, receipt['output_sha256'])
            with np.load(source/relative) as f:
                np.testing.assert_array_equal(f['rows'], np.arange(task['start'], task['stop']))
            (ROOT/relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, ROOT/relative)
            shutil.copy2((source/relative).with_suffix('.json'), (ROOT/relative).with_suffix('.json'))
        save(ROOT/'preflight.json', dict(status='passed',
            protocol_sha256=study.file_hash(ROOT/'protocol.json'),
            inherited_from=str(source/'preflight.json'),
            inherited_sha256=study.file_hash(source/'preflight.json'),
            note='Unchanged scientific settings; reuse validated baseline chunks. Each new repair still requires direct centre-density replay.'))
    save(ROOT/'inputs.json', dict(protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        samples=records, baseline_chunks=chunks, repair_groups=repairs))
    print('COHORT_PREPARED', {k:(v['n_scientific'],len(v['bright_rows'])) for k,v in records.items()}, flush=True)


def observations():
    rec = study.read(ROOT/'inputs.json')
    assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    for name, m in rec['samples'].items():
        for filename, sha in m['output_sha256'].items():
            study.verify(ROOT/f'input/{name}'/filename, sha)
    return {name:study.MockCatalogue.load(ROOT/f'input/{name}') for name in rec['samples']}, rec


def window(sample, rows):
    return study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
        sample.truth.iloc[rows].reset_index(drop=True))


def load_pair():
    expanded, proposal = domain.baseline.load()
    corrected, unused = domain.load()
    del unused
    gc.collect()
    old, new = expanded.cache.prior.weights, corrected.cache.prior.weights
    assert (old > 0).all() and (new >= 0).all()
    with np.errstate(divide='ignore'):
        ratio = np.log(new/old)
    return dict(expanded=expanded, corrected=corrected), proposal, ratio


def evaluate(likelihood, proposal, sample, seed):
    result = study.run_adaptive_section5(likelihood, sample, proposal,
        center=domain.protocol()['center'], h=.001, draw_ladder=domain.baseline.old.LADDER,
        n_candidates=16384, proposal_prefilter_candidates=131072, proposal_seed=seed,
        candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
        retain_full_ladder=True, full_information=True,
        object_ids=sample.truth.parent_sample_row.to_numpy(np.int64), object_chunk=4, atom_chunk=4096)
    return {key:getattr(result, key) for key in KEYS}


def preflight():
    protocol()
    samples, _ = observations()
    likelihoods, proposal, ratio = load_pair()
    checks = []
    for index in (0, 2):
        group = study.read(tail.PILOT/'cohort.json')['groups'][index]
        name, rows = group['sample'], np.asarray(group['rows'])
        w = window(samples[name], rows)
        # Bridge to the historical K1024 calculation before changing K.
        actual = domain.baseline.matched.evaluate(likelihoods['expanded'], proposal,
            samples[name], rows, domain.protocol()['center'], f'historical_replay_{name}')
        source = domain.SOURCE/f'moments/{name}/worker_{rows[0]//5000}.npz'
        study.verify(source, study.read(source.with_suffix('.json'))['output_sha256'])
        with np.load(source) as old:
            local = rows % 5000
            for key, values in actual.items():
                expected = old[key][:,local] if key.startswith(('ladder_', 'weight_')) else old[key][local]
                np.testing.assert_array_equal(values, expected, err_msg=f'historical replay {name}/{key}')
        record = study.read(tail.RANK/f'worker_{index}.json')
        study.verify(tail.RANK/f'worker_{index}.npz', record['output_sha256'])
        with np.load(tail.RANK/f'worker_{index}.npz') as f:
            atoms, points, reference = f['atom_indices'], f['points'], f['candidate_log_numerator_16384']
        error = 0.
        for pi, point in enumerate(points):
            values = {key:likelihood.log_importance_weights_tensor(w.measurements, *point, atom_indices=atoms,
                proposal_probability=np.ones_like(atoms, float), object_chunk=4, atom_chunk=4096).cpu().numpy()
                for key,likelihood in likelihoods.items()}
            np.testing.assert_allclose(values['expanded']+ratio[atoms], values['corrected'], rtol=0, atol=1.e-7)
            summed = np.logaddexp.reduce(values['corrected'], axis=1)
            np.testing.assert_allclose(summed, reference[pi], rtol=0, atol=1.e-7)
            error = max(error, float(np.max(abs(summed-reference[pi]))))
        # Include zeroed added atoms, not just selected positive-mass candidates.
        zeros = np.flatnonzero(~np.isfinite(ratio))[:8192]
        positive = np.flatnonzero(np.isfinite(ratio))[:8192]
        mixed = np.r_[zeros, positive]
        assert len(mixed) == 16384
        terms = {key:likelihood.log_importance_weights_tensor(w.measurements, *domain.protocol()['center'],
            atom_indices=mixed, proposal_probability=np.ones(len(mixed)), object_chunk=4, atom_chunk=4096).cpu().numpy()
            for key,likelihood in likelihoods.items()}
        np.testing.assert_allclose(terms['expanded']+ratio[mixed], terms['corrected'], rtol=0, atol=1.e-7)
        assert np.isneginf(terms['corrected'][:,:8192]).all()
        # Exercise both targets with the same full-support proposal and random stream.
        for prior, likelihood in likelihoods.items():
            result = evaluate(likelihood, proposal, w, 8701)
            assert np.isfinite(result['ladder_score']).all() and np.isfinite(result['ladder_information']).all()
        checks.append(dict(sample=name, focal_row=group['focal_row'], historical_replay=True,
            maximum_corrected_log_sum_error=error, weight_ratio_and_zero_support_passed=True))
    save(ROOT/'preflight.json', dict(status='passed', protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        checks=checks, gpu=torch.cuda.get_device_name(0), maximum_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('COHORT_PREFLIGHT_PASS', checks, flush=True)


def repair(likelihoods, proposal, ratio, w, task):
    p = domain.protocol()
    out = ROOT/f'repair/{task["sample"]}/group_{task["group"]:04d}'
    out.mkdir(parents=True, exist_ok=False)
    started, n, k = time.monotonic(), p['n_atoms'], 16384
    logs = np.lib.format.open_memmap(out/'expanded_center.npy', mode='w+', dtype=np.float64, shape=(4,n))
    best = {key:torch.full((4,k), -torch.inf, device='cuda', dtype=torch.float64) for key in PRIORS}
    atoms = {key:torch.full((4,k), -1, device='cuda', dtype=torch.long) for key in PRIORS}
    totals = {key:np.full(4, -np.inf) for key in PRIORS}
    for lo in range(0, n, 16384):
        hi = min(lo+16384, n)
        ids = np.arange(lo,hi)
        raw = likelihoods['expanded'].log_importance_weights_tensor(w.measurements, *p['center'],
            atom_indices=ids, proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
        assert not torch.isnan(raw).any() and not torch.isposinf(raw).any()
        logs[:,lo:hi] = raw.cpu().numpy()
        for key in PRIORS:
            values = raw if key == 'expanded' else raw+torch.as_tensor(ratio[lo:hi], device='cuda')
            totals[key] = np.logaddexp(totals[key], torch.logsumexp(values,dim=1).cpu().numpy())
            joined = torch.cat([atoms[key],torch.as_tensor(ids,device='cuda').expand(4,-1)],dim=1)
            best[key], pos = torch.topk(torch.cat([best[key],values],dim=1),k,dim=1,sorted=True)
            atoms[key] = joined.gather(1,pos)
        if lo//16384 % 256 == 0 or hi == n:
            print(f'COHORT_SCAN {task} atoms={hi}/{n} seconds={time.monotonic()-started:.1f}',flush=True)
    logs.flush()
    output, candidate_output = {}, {}
    for key, likelihood in likelihoods.items():
        ids, scores = atoms[key].cpu().numpy(), best[key].cpu().numpy()
        assert (ids >= 0).all() and np.isfinite(scores).all()
        replay = likelihood.log_importance_weights_tensor(w.measurements, *p['center'], atom_indices=ids,
            proposal_probability=np.ones_like(ids,float), object_chunk=4, atom_chunk=4096).cpu().numpy()
        np.testing.assert_allclose(replay, scores, rtol=0, atol=1.e-7)
        distributions = [tail.build_tail_distribution(logs[j] if key=='expanded' else logs[j]+ratio,
            likelihood.cache.prior.weights, ids[j], delta=.1) for j in range(4)]
        distance = scores[:,:1]-scores
        candidates = domain.ProposalCandidates(ids,distance,distance[:,-1])
        adapter = tail.ExactTailProposal(proposal,candidates,distributions,w.truth.parent_sample_row.to_numpy(np.int64))
        for seed in p['proposal_seeds']:
            output.update({f'{key}_{seed}_{name}':value for name,value in evaluate(likelihood,adapter,w,seed).items()})
        candidate_output.update({f'{key}_atoms':ids,f'{key}_center_scores':scores,f'{key}_center_total':totals[key]})
        del adapter, distributions
        gc.collect()
    np.savez_compressed(out/'moments.npz',**output)
    np.savez_compressed(out/'candidates.npz',**candidate_output)
    save(out/'manifest.json',dict(status='complete',task=task,center_replay_passed=True,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'),elapsed_seconds=time.monotonic()-started,
        output_sha256={f.name:study.file_hash(f) for f in out.iterdir() if f.suffix in ('.npz','.npy')}))
    print('COHORT_REPAIR_COMPLETE',task,'seconds',time.monotonic()-started,flush=True)


def worker(index):
    protocol()
    check = study.read(ROOT/'preflight.json')
    assert check['status']=='passed' and check['protocol_sha256']==study.file_hash(ROOT/'protocol.json')
    samples, rec = observations()
    likelihoods, proposal, ratio = load_pair()
    seeds = domain.protocol()['proposal_seeds']
    for ci in range(index,len(rec['baseline_chunks']),4):
        task = rec['baseline_chunks'][ci]
        cached = ROOT/f'baseline/{task["sample"]}/chunk_{task["start"]:05d}.npz'
        if 'baseline_source_root' in study.read(ROOT/'protocol.json'):
            receipt = study.read(cached.with_suffix('.json'))
            assert receipt['task'] == task
            study.verify(cached, receipt['output_sha256'])
            print('COHORT_BASELINE_REUSED', index, ci, task, flush=True)
            continue
        parts = {}
        for lo in range(task['start'],task['stop'],4):
            w = window(samples[task['sample']],np.arange(lo,lo+4))
            for seed in seeds:
                for prior, likelihood in likelihoods.items():
                    for key,value in evaluate(likelihood,proposal,w,seed).items():
                        parts.setdefault(f'{prior}_{seed}_{key}',[]).append(value)
            if ci == index and lo == task['start']:
                print('COHORT_BASELINE_FIRST_GROUP',index,task['sample'],lo,flush=True)
        output = {f'{prior}_{seed}_{key}':np.concatenate(parts[f'{prior}_{seed}_{key}'],
            axis=1 if key.startswith(('ladder_','weight_')) else 0)
            for prior in PRIORS for seed in seeds for key in KEYS}
        folder=ROOT/f'baseline/{task["sample"]}'
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'chunk_{task["start"]:05d}.npz'
        np.savez_compressed(path,rows=np.arange(task['start'],task['stop']),**output)
        save(path.with_suffix('.json'),dict(task=task,output_sha256=study.file_hash(path)))
        print('COHORT_BASELINE_COMPLETE',index,ci,task,flush=True)
    save(ROOT/f'baseline_worker_{index}.json',dict(status='complete',protocol_sha256=study.file_hash(ROOT/'protocol.json')))
    for ri in range(index,len(rec['repair_groups']),4):
        task = rec['repair_groups'][ri]
        rows=np.arange(4*task['group'],4*task['group']+4)
        repair(likelihoods,proposal,ratio,window(samples[task['sample']],rows),task)
        gc.collect()
        torch.cuda.empty_cache()
    save(ROOT/f'worker_{index}.json',dict(status='complete',protocol_sha256=study.file_hash(ROOT/'protocol.json')))


def combine():
    protocol()
    samples, rec = observations()
    for i in range(4):
        w=study.read(ROOT/f'worker_{i}.json')
        assert w['status']=='complete' and w['protocol_sha256']==study.file_hash(ROOT/'protocol.json')
    seeds=domain.protocol()['proposal_seeds']
    all_data={}
    for name,sample in samples.items():
        base={f'{prior}_{seed}_{key}':[] for prior in PRIORS for seed in seeds for key in MOMENTS}
        for task in rec['baseline_chunks']:
            if task['sample']!=name:
                continue
            path=ROOT/f'baseline/{name}/chunk_{task["start"]:05d}.npz'
            study.verify(path,study.read(path.with_suffix('.json'))['output_sha256'])
            with np.load(path) as f:
                np.testing.assert_array_equal(f['rows'],np.arange(task['start'],task['stop']))
                for key in base:
                    base[key].append(f[key])
        base={key:np.concatenate(value,axis=1) for key,value in base.items()}
        repaired={key:value.copy() for key,value in base.items()}
        bright=np.asarray(rec['samples'][name]['bright_rows'])
        for group in rec['samples'][name]['bright_groups']:
            folder=ROOT/f'repair/{name}/group_{group:04d}'
            receipt=study.read(folder/'manifest.json')
            assert receipt['status']=='complete' and receipt['center_replay_passed']
            assert receipt['protocol_sha256']==study.file_hash(ROOT/'protocol.json')
            study.verify(folder/'moments.npz',receipt['output_sha256']['moments.npz'])
            rows=np.arange(4*group,4*group+4)
            local=np.flatnonzero(np.isin(rows,bright))
            with np.load(folder/'moments.npz') as f:
                for key in repaired:
                    repaired[key][:,rows[local]]=f[key][:,local]
        nonbright=~np.isin(np.arange(len(sample.measurements)),bright)
        for key in base:
            np.testing.assert_array_equal(base[key][:,nonbright],repaired[key][:,nonbright])
        all_data[name]=(base,repaired)
    reports=[]
    retained=~samples['mock'].truth.numerical_padding.to_numpy()
    actual_label = 'actual_all20k' if len(samples['actual'].measurements) == 20000 else 'actual_subset'
    for label,name,valid in [(actual_label,'actual',np.ones(len(samples['actual'].measurements),bool)),
            ('actual_same_ids','actual',retained),('mock','mock',retained)]:
        base,repaired=all_data[name]
        for seed in seeds:
            rungs,influences=[],{}
            for rung,draws in enumerate(domain.baseline.old.LADDER):
                row=dict(draws=draws,arms={})
                for integration,data in [('baseline',base),('bright_repaired',repaired)]:
                    for prior in PRIORS:
                        arm=f'{prior}_{integration}'
                        score,info=[data[f'{prior}_{seed}_{key}'][rung,valid] for key in MOMENTS]
                        result,influences[rung,arm]=domain.baseline.old.summarize(domain.protocol()['center'],score,info)
                        row['arms'][arm]=dict(result,score_sum=score.sum(0).tolist(),information_sum=info.sum(0).tolist())
                comparisons={}
                for first,second in [('expanded_baseline','corrected_baseline'),
                        ('expanded_baseline','expanded_bright_repaired'),
                        ('corrected_baseline','corrected_bright_repaired'),
                        ('expanded_bright_repaired','corrected_bright_repaired')]:
                    comparisons[f'{first}_to_{second}']=domain.baseline.old.paired(row['arms'][first],row['arms'][second],
                        influences[rung,first],influences[rung,second])
                row['paired_changes']=comparisons
                rungs.append(row)
            reports.append(dict(sample=label,n_observations=int(valid.sum()),seed=seed,rungs=rungs,
                paired_8192_to_16384={arm:domain.baseline.old.paired(rungs[-2]['arms'][arm],rungs[-1]['arms'][arm],
                    influences[4,arm],influences[5,arm]) for arm in rungs[-1]['arms']}))
    path=ROOT/'moments.npz'
    np.savez_compressed(path,**{f'{name}_{integration}_{key}':value
        for name,pair in all_data.items() for integration,data in zip(('baseline','bright_repaired'),pair)
        for key,value in data.items()})
    save(ROOT/'result.json',dict(status='complete',protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        moments_sha256=study.file_hash(path),reports=reports,
        limitation='Fixed-centre partial integration repair: only measured-mag<20 rows replaced. Fainter integration remains approximate; matched mocks have double selection. No optimizer continuation or 0.3% acceptance.'))
    print('COHORT_RECOVERY_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','worker','combine'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    try:
        worker(int(os.environ['SLURM_ARRAY_TASK_ID'])) if args.stage=='worker' else globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID","cpu")}.json',dict(traceback=traceback.format_exc()))
        raise
