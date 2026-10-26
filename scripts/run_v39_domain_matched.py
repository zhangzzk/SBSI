"""Frozen domain-matched prior and exhaustive bright-sampling pilot."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_DOMAIN_ROOT'])
SOURCE = Path(os.environ['SBSI_EXPANDED_ROOT'])
sys.path.insert(0, str(SOURCE))
import run_v39_expanded_prior as baseline  # noqa: E402
from sbsi.catalogue_sampling import ProposalCandidates  # noqa: E402

study, np, pd, torch, save = baseline.study, baseline.np, baseline.pd, baseline.torch, baseline.save
ARMS = ('gaussian', 'exact_location', 'exact_union')


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    return p


def identity():
    p, rec = protocol(), study.read(ROOT/'execution.json')
    assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    assert rec['baseline_execution'] == baseline.identity()
    for path, sha in rec['sources'].items():
        study.verify(path, sha)
    assert p['n_atoms'] == 28012032
    return rec


def initialize():
    p = protocol()
    assert not (ROOT/'execution.json').exists()
    source = baseline.expanded()
    receipt = study.read(source/'support/manifest.json')
    for name, sha in receipt['output_sha256'].items():
        study.verify(source/'support'/name, sha)
    z = np.load(source/'support/zero.npy', mmap_mode='r')
    original = np.load(source/'support/weights.npy', mmap_mode='r')
    mask_path = study.BASE/'inference_bright_extension_20260925_v1/catalogue_audit_masks.npz'
    domain = np.load(mask_path)['within_generator_domain']
    assert len(z) == p['n_atoms'] and len(domain) == 7780811
    n_old = len(z)-len(domain)
    bright = z[:, 3] < 20
    good = np.r_[np.ones(n_old, bool), domain]
    assert bright[n_old:].all() and int((~good).sum()) == 848650
    mass = original[bright].sum()
    w = np.array(original)
    w[~good] = 0.
    w[bright] *= mass/w[bright].sum()
    np.testing.assert_array_equal(w[~bright], original[~bright])
    np.testing.assert_allclose(w[bright].sum(), mass, rtol=0, atol=1.e-15)
    np.testing.assert_allclose(w.sum(), original.sum(), rtol=0, atol=1.e-12)
    assert int((bright & (w > 0)).sum()) == 7201662
    np.save(ROOT/'weights.npy', w)
    samples, gen = baseline.observations()
    assert gen['center'] == p['center']
    rng = np.random.default_rng(p['cohort_seed'])
    groups = []
    for name, stress in [('actual', 1958), ('mock', 4460)]:
        sample = samples[name]
        valid = sample.measurements.measured_mag_auto.to_numpy() < 20
        if name == 'mock':
            valid &= ~sample.truth.numerical_padding.to_numpy()
        candidates = np.flatnonzero(valid & (np.arange(len(valid))//4 != stress//4))
        random_row = int(rng.choice(candidates))
        for label, row in [('known_failure', stress), ('fixed_random_bright', random_row)]:
            groups.append(dict(sample=name, selection=label, focal_row=row,
                rows=list(range(row//4*4, row//4*4+4)),
                measured_magnitude=float(sample.measurements.measured_mag_auto.iloc[row])))
    save(ROOT/'cohort.json', dict(groups=groups, seed=p['cohort_seed'],
        selection='Two known failures plus one fixed-seed measured-bright row per sample; original four-row groups retained'))
    paths = [mask_path, source/'support/manifest.json', SOURCE/'assembled.json',
             SOURCE/'prior_audit.json', ROOT/'weights.npy', ROOT/'cohort.json']
    save(ROOT/'execution.json', dict(protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        baseline_execution=baseline.identity(), sources={str(x):study.file_hash(x) for x in paths}))
    save(ROOT/'prior_audit.json', dict(status='passed', n_atoms=len(w), zero_weight_added=int((~good).sum()),
        bright_positive_atoms=int((bright & (w > 0)).sum()), bright_mass=float(mass),
        valid_bright_weight=float(w[bright & (w > 0)][0]), total_mass=float(w.sum()),
        unchanged_nonbright=True, unchanged_observations=True, groups=groups))
    print('DOMAIN_INITIALIZED', groups, flush=True)


def wait(paths):
    start = time.monotonic()
    while not all(path.exists() for path in paths):
        if list(ROOT.glob('failure_*.json')):
            raise RuntimeError('another domain pilot worker failed')
        if time.monotonic()-start > 3*3600:
            raise TimeoutError('domain pilot stage barrier timed out')
        time.sleep(10)


def normalize(index):
    """Replay identical forward samples; reduce old and new weights together."""
    source, p = baseline.expanded(), protocol()
    support = study.read(source/'support/manifest.json')
    offsets = np.r_[0, np.cumsum([r['n_rows'] for r in support['receipts']])]
    zero = np.load(source/'support/zero.npy', mmap_mode='r')
    weights = np.load(ROOT/'weights.npy', mmap_mode='r')
    old_weights = np.load(source/'support/weights.npy', mmap_mode='r')
    probs = np.load(SOURCE/'probability.npy', mmap_mode='r')
    points = study.read(SOURCE/'assembled.json')['points']
    assert points == list(map(list, study.stencil(p['center'], p['h'])))
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    cut = study.output_cut(flow)
    for shard in range(index, 28, p['workers']):
        lo, hi = offsets[shard:shard+2]
        frame = pd.DataFrame(zero[lo:hi], columns=study.FEATURES)
        cache = study.V39CatalogueCache(frame, weights[lo:hi], support['receipts'][shard]['conditions'],
            probabilities=dict(zip(map(tuple, points), probs[:, lo:hi])))
        masses, replay = [], []
        for point in points:
            view = cache.get(*point)
            seed = study.seed(8101, shard)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            new_mass, old_mass = 0., 0.
            for start in range(0, len(frame), 8192):
                stop = min(start+8192, len(frame))
                draws = flow.sample(view.flow.iloc[start:stop], n_samples=64, batch_size=8192, qmc=True)
                passed = np.asarray(cut(draws), bool)
                assert passed.shape == (stop-start, 64)
                probability = passed.mean(1, dtype=np.float64)
                detection = view.detection_probability[start:stop]
                new_mass += float(np.sum(weights[lo+start:lo+stop]*detection*probability, dtype=np.float64))
                old_mass += float(np.sum(old_weights[lo+start:lo+stop]*detection*probability, dtype=np.float64))
            masses.append(new_mass)
            replay.append(old_mass)
            cache.discard_views([point])
            print(f'DOMAIN_NORMALIZATION worker={index} shard={shard} point={point} mass={new_mass:.10g}', flush=True)
        reference = study.read(SOURCE/f'normalization/shard_{shard:02d}/manifest.json')
        assert reference['points'] == points
        np.testing.assert_allclose(replay, reference['masses'], rtol=0, atol=2.e-13,
                                   err_msg='unchanged old-weight normalization replay')
        save(ROOT/f'normalization/shard_{shard:02d}.json', dict(status='complete', shard=shard,
            atom_range=[int(lo), int(hi)], points=points, masses=masses, baseline_replay_masses=replay,
            maximum_replay_error=float(np.max(np.abs(np.array(replay)-reference['masses']))),
            execution_sha256=study.file_hash(ROOT/'execution.json')))
    del flow, cache
    gc.collect()
    torch.cuda.empty_cache()


def assemble():
    p, end = protocol(), 0
    masses = np.zeros(10)
    points = list(map(list, study.stencil(p['center'], p['h'])))
    for shard in range(28):
        rec = study.read(ROOT/f'normalization/shard_{shard:02d}.json')
        assert rec['status'] == 'complete' and rec['points'] == points
        assert rec['execution_sha256'] == study.file_hash(ROOT/'execution.json')
        assert rec['atom_range'][0] == end
        end = rec['atom_range'][1]
        masses += rec['masses']
    assert end == p['n_atoms'] and np.all((masses > 0) & (masses <= 1))
    study.ExactPopulationNormalization(dict(zip(map(tuple, points), masses)), p['h'], identity(),
        dict(method_detail='Domain-matched weights; same64 QMC draws and shard seeds; expanded baseline replay checked')
        ).save(ROOT/'normalization.json')
    save(ROOT/'assembled.json', dict(status='complete', points=points, selected_mass=masses.tolist(),
        normalization_sha256=study.file_hash(ROOT/'normalization.json')))
    print('DOMAIN_NORMALIZATION_COMPLETE', flush=True)


def load():
    source, rec = baseline.expanded(), study.read(SOURCE/'assembled.json')
    weights = np.load(ROOT/'weights.npy')
    support = study.read(source/'support/manifest.json')
    zero = pd.DataFrame(np.load(source/'support/zero.npy', mmap_mode='r'), columns=study.FEATURES)
    cache = study.V39CatalogueCache(zero, weights, support['receipts'][0]['conditions'],
        probabilities=dict(zip(map(tuple, rec['points']), np.load(SOURCE/'probability.npy', mmap_mode='r'))))
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    response = study.response_from_receipts(support['receipts'], study.load_disk_response(study.V39_LIKE, device='cuda', nthread=4))
    likelihood = study.DiskCatalogueLikelihood(flow, cache, disk_response=response,
        selection=study.CatalogueSelection(study.output_cut(flow)))
    norm_receipt = study.read(ROOT/'assembled.json')
    study.verify(ROOT/'normalization.json', norm_receipt['normalization_sha256'])
    likelihood.population_normalization = study.ExactPopulationNormalization.from_payload(study.read(ROOT/'normalization.json'))
    assert likelihood.population_normalization.identity == identity()
    likelihood.compile_precision_density()
    coords = study.ProposalCoordinateTable.load(source/'assembled/proposal')
    coord_rec = study.read(source/'assembled/manifest.json')
    object.__setattr__(coords, '_metric_center', np.asarray(coord_rec['metric_center']))
    object.__setattr__(coords, '_metric_scale', np.asarray(coord_rec['metric_scale']))
    proposal = study.DefensiveLocalProposal(coords, weights,
        local_base_weights=np.load(SOURCE/'local_probability.npy'), score_dtype=torch.float64)
    return likelihood, proposal


class FixedCandidates:
    def __init__(self, proposal, candidates):
        self.proposal, self.local = proposal, candidates

    def __getattr__(self, name):
        return getattr(self.proposal, name)

    def candidates(self, observed, *, n_candidates, **kwargs):
        assert len(observed) == len(self.local.indices) == 4 and n_candidates == 1024
        return self.local


def build_candidates(likelihood, proposal, frame, center):
    """Keep K fixed; replace only the ranking of two declared screening pools."""
    count = protocol()['screen_count']
    gaussian = proposal.candidates(frame, n_candidates=1024, prefilter_candidates=count, torch_device='cuda')
    location = proposal.candidates(frame, n_candidates=count, prefilter_candidates=count, torch_device='cuda')
    whole = proposal.whole_catalogue_proxy_candidates(frame, n_candidates=count, device='cuda', object_chunk=4)
    pools = dict(location=location.indices, whole=whole.indices)
    scored = {}
    for label, atoms in pools.items():
        values = []
        for start in range(0, count, 16384):
            grid = atoms[:, start:start+16384]
            value = likelihood.log_importance_weights_tensor(frame, *center, atom_indices=grid,
                proposal_probability=np.ones_like(grid, dtype=float), object_chunk=4, atom_chunk=4096)
            values.append(value.cpu().numpy())
        scored[label] = np.concatenate(values, axis=1)
    result = dict(gaussian=gaussian)
    for name in ('exact_location', 'exact_union'):
        indices, distances = [], []
        for row in range(4):
            atoms = pools['location'][row]
            scores = scored['location'][row]
            if name == 'exact_union':
                atoms = np.concatenate([atoms, pools['whole'][row]])
                scores = np.concatenate([scores, scored['whole'][row]])
                atoms, unique = np.unique(atoms, return_index=True)
                scores = scores[unique]
            assert np.isfinite(scores).sum() >= 1024
            keep = np.argsort(-scores, kind='stable')[:1024]
            indices.append(atoms[keep])
            distances.append(scores[keep[0]]-scores[keep])
        distances = np.asarray(distances)
        result[name] = ProposalCandidates(np.asarray(indices), distances, distances[:, -1])
    return result, pools


def sampled(likelihood, proposal, window, candidates, rows, index):
    p = protocol()
    outputs = {}
    for name, local in candidates.items():
        for seed in p['proposal_seeds']:
            result = study.run_adaptive_section5(likelihood, window, FixedCandidates(proposal, local),
                center=p['center'], h=p['h'], draw_ladder=baseline.old.LADDER,
                n_candidates=1024, proposal_prefilter_candidates=p['screen_count'],
                proposal_seed=seed, candidate_backend='torch', estimator_mode='tilted_stratified',
                tilt_delta=.1, retain_full_ladder=True, full_information=True,
                object_ids=np.asarray(rows), object_chunk=4, atom_chunk=4096)
            for key in baseline.old.KEYS:
                outputs[f'{name}_{seed}_{key}'] = getattr(result, key)
            print(f'DOMAIN_SAMPLED worker={index} arm={name} seed={seed}', flush=True)
    np.savez_compressed(ROOT/f'sampled_{index}.npz', **outputs)


def exact(likelihood, frame, candidates, pools, index):
    p = protocol()
    points = [point for point in study.stencil(p['center'], p['h']) if point != (0., 0.)]
    assert len(points) == 9
    masks = dict(location=pools['location'], union=np.concatenate(list(pools.values()), axis=1),
                 **{name:rec.indices for name,rec in candidates.items()})
    membership = {}
    for name, ids in masks.items():
        flags = np.zeros((4, p['n_atoms']), bool)
        for row in range(4):
            flags[row, ids[row]] = True
        membership[name] = flags
    sums = np.full((len(points), 4), -np.inf)
    restricted = {name:np.full_like(sums, -np.inf) for name in membership}
    second, maximum = np.full_like(sums, -np.inf), np.full_like(sums, -np.inf)
    started = time.monotonic()
    for start in range(0, p['n_atoms'], 16384):
        stop = min(start+16384, p['n_atoms'])
        atoms = np.arange(start, stop)
        for pi, point in enumerate(points):
            terms = likelihood.log_importance_weights_tensor(frame, *point, atom_indices=atoms,
                proposal_probability=np.ones(len(atoms)), object_chunk=4, atom_chunk=4096)
            assert not torch.isnan(terms).any() and not torch.isposinf(terms).any()
            sums[pi] = np.logaddexp(sums[pi], torch.logsumexp(terms, dim=1).cpu().numpy())
            second[pi] = np.logaddexp(second[pi], torch.logsumexp(2*terms, dim=1).cpu().numpy())
            maximum[pi] = np.maximum(maximum[pi], terms.max(1).values.cpu().numpy())
            for name, flags in membership.items():
                mask = torch.as_tensor(flags[:, start:stop], device=terms.device)
                value = torch.logsumexp(terms.masked_fill(~mask, -torch.inf), dim=1).cpu().numpy()
                restricted[name][pi] = np.logaddexp(restricted[name][pi], value)
        if start//16384 % 64 == 0 or stop == p['n_atoms']:
            print(f'DOMAIN_EXACT worker={index} atoms={stop}/{p["n_atoms"]} seconds={time.monotonic()-started:.1f}', flush=True)
    norms = np.array([likelihood.population_normalization.log_mass(*point) for point in points])
    np.savez_compressed(ROOT/f'exact_{index}.npz', points=np.asarray(points), log_numerator=sums,
        log_selected=sums-norms[:, None], log_second=second, log_maximum=maximum,
        **{f'log_subset_{name}':values for name,values in restricted.items()})


def worker(index):
    identity()
    normalize(index)
    wait([ROOT/f'normalization/shard_{shard:02d}.json' for shard in range(28)])
    if index == 0:
        assemble()
    wait([ROOT/'assembled.json'])
    likelihood, proposal = load()
    group = study.read(ROOT/'cohort.json')['groups'][index]
    samples, _ = baseline.observations()
    sample = samples[group['sample']]
    rows = group['rows']
    frame = sample.measurements.iloc[rows].reset_index(drop=True)
    window = study.MockCatalogue(frame, sample.truth.iloc[rows].reset_index(drop=True))
    candidates, pools = build_candidates(likelihood, proposal, frame, protocol()['center'])
    np.savez_compressed(ROOT/f'candidates_{index}.npz',
        **{name:rec.indices for name,rec in candidates.items()}, **{f'pool_{name}':ids for name,ids in pools.items()})
    sampled(likelihood, proposal, window, candidates, rows, index)
    # Release dense proposal buffers before the long exhaustive density scan.
    del proposal, window, sample, samples
    gc.collect()
    torch.cuda.empty_cache()
    exact(likelihood, frame, candidates, pools, index)
    gpu = torch.cuda.get_device_properties(0)
    paths = [ROOT/f'{name}_{index}.npz' for name in ('candidates', 'sampled', 'exact')]
    save(ROOT/f'worker_{index}.json', dict(status='complete', group=group, gpu=gpu.name,
        gpu_total_memory=gpu.total_memory, execution_sha256=study.file_hash(ROOT/'execution.json'),
        output_sha256={path.name:study.file_hash(path) for path in paths}))
    print('DOMAIN_WORKER_COMPLETE', index, flush=True)


def moments(points, logs):
    p = protocol()
    table = {tuple(point):value for point,value in zip(points, logs)}
    center, h = p['center'], p['h']
    def value(i, j):
        return table[(round(center[0]+i*h, 14), round(center[1]+j*h, 14))]
    score = np.stack([(value(1, 0)-value(-1, 0))/(2*h), (value(0, 1)-value(0, -1))/(2*h)], axis=-1)
    info = np.zeros((4, 2, 2))
    info[:, 0, 0] = -(value(1, 0)-2*value(0, 0)+value(-1, 0))/h**2
    info[:, 1, 1] = -(value(0, 1)-2*value(0, 0)+value(0, -1))/h**2
    info[:, 0, 1] = info[:, 1, 0] = -(value(1, 1)-value(1, -1)-value(-1, 1)+value(-1, -1))/(4*h*h)
    return score, info


def combine():
    p, reports = protocol(), []
    for index in range(p['workers']):
        rec = study.read(ROOT/f'worker_{index}.json')
        assert rec['status'] == 'complete' and rec['execution_sha256'] == study.file_hash(ROOT/'execution.json')
        for name, sha in rec['output_sha256'].items():
            study.verify(ROOT/name, sha)
        with np.load(ROOT/f'exact_{index}.npz') as data:
            exact_data = {name:data[name] for name in data.files}
        score, info = moments(exact_data['points'], exact_data['log_selected'])
        j = rec['group']['focal_row'] % 4
        center_index = list(map(tuple, exact_data['points'])).index(tuple(np.round(p['center'], 14)))
        masses = exact_data['log_numerator']
        report = dict(group=rec['group'], exact_score=score[j].tolist(), exact_information=info[j].tolist(),
            center_distinct_atom_ess=float(np.exp(2*masses[center_index,j]-exact_data['log_second'][center_index,j])),
            center_max_atom_fraction=float(np.exp(exact_data['log_maximum'][center_index,j]-masses[center_index,j])),
            minimum_stencil_coverage={key.removeprefix('log_subset_'):float(np.exp(values[:,j]-masses[:,j]).min())
                for key,values in exact_data.items() if key.startswith('log_subset_')}, arms={})
        with np.load(ROOT/f'sampled_{index}.npz') as sampled_data:
            for arm in ARMS:
                checks = []
                for seed in p['proposal_seeds']:
                    for rung, draws in [(-2, 8192), (-1, 16384)]:
                        ds = sampled_data[f'{arm}_{seed}_ladder_score'][rung,j]-score[j]
                        di = sampled_data[f'{arm}_{seed}_ladder_information'][rung,j]-info[j]
                        good = bool(np.all(abs(ds) <= p['score_atol']+p['moment_rtol']*abs(score[j]))
                            and np.all(abs(di) <= p['information_atol']+p['moment_rtol']*abs(info[j])))
                        checks.append(dict(seed=seed, draws=draws, score_error=ds.tolist(), information_error=di.tolist(), passed=good))
                report['arms'][arm] = dict(checks=checks, passed=all(x['passed'] for x in checks))
        reports.append(report)
    passed = {arm:all(report['arms'][arm]['passed'] for report in reports) for arm in ARMS}
    save(ROOT/'result.json', dict(status='complete', reports=reports, numerical_pilot_pass=passed,
        limitations=['Four focal bright observations; two known failures and two fixed-seed draws, not population acceptance.',
                     'h=.001, existing floating-point density, 64-draw QMC selection normalization.',
                     'No scientific shear estimate or 0.3% acceptance; broad inference and off-prior closure remain gated on numerical evidence.']))
    print('DOMAIN_PILOT_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'worker', 'combine'])
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
