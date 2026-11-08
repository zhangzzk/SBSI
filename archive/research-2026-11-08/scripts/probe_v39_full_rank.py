"""Exact whole-prior centre ranking, checked against the frozen nine-view pilot."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_FULL_RANK_ROOT'])
PILOT = Path(os.environ['SBSI_DOMAIN_ROOT'])
sys.path.insert(0, str(PILOT))
import run_v39_domain_matched as domain  # noqa: E402

study, np, torch, save = domain.study, domain.np, domain.torch, domain.save


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    study.verify(PILOT/'result.json', p['reference_result_sha256'])
    study.verify(PILOT/'run_v39_domain_matched.py', domain.protocol()['script_sha256'])
    return p


class FixedCandidates(domain.FixedCandidates):
    def candidates(self, observed, *, n_candidates, **kwargs):
        assert len(observed) == 4 and n_candidates == self.local.indices.shape[1]
        return self.local


def worker(index):
    p = protocol()
    likelihood, proposal = domain.load()
    group = study.read(PILOT/'cohort.json')['groups'][index]
    samples, _ = domain.baseline.observations()
    sample = samples[group['sample']]
    rows = np.asarray(group['rows'])
    window = study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
                                 sample.truth.iloc[rows].reset_index(drop=True))
    reference = study.read(PILOT/f'worker_{index}.json')
    study.verify(PILOT/f'exact_{index}.npz', reference['output_sha256'][f'exact_{index}.npz'])
    center, n_atoms, maximum_k = domain.protocol()['center'], domain.protocol()['n_atoms'], max(p['candidate_ladder'])
    best = torch.full((4, maximum_k), -torch.inf, dtype=torch.float64, device='cuda')
    best_ids = torch.full((4, maximum_k), -1, dtype=torch.long, device='cuda')
    total = np.full(4, -np.inf)
    started = time.monotonic()
    for start in range(0, n_atoms, 16384):
        ids = np.arange(start, min(start+16384, n_atoms))
        values = likelihood.log_importance_weights_tensor(window.measurements, *center, atom_indices=ids,
            proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
        assert not torch.isnan(values).any() and not torch.isposinf(values).any()
        total = np.logaddexp(total, torch.logsumexp(values, dim=1).cpu().numpy())
        ids = torch.as_tensor(ids, device=best.device).expand(4, -1)
        all_ids = torch.cat([best_ids, ids], dim=1)
        best, positions = torch.topk(torch.cat([best, values], dim=1), maximum_k, dim=1, sorted=True)
        best_ids = all_ids.gather(1, positions)
        if start//16384 % 64 == 0 or start+16384 >= n_atoms:
            print(f'FULL_RANK worker={index} atoms={min(start+16384,n_atoms)}/{n_atoms} seconds={time.monotonic()-started:.1f}', flush=True)
    assert torch.isfinite(best).all() and (best_ids >= 0).all()
    scores, atoms = best.cpu().numpy(), best_ids.cpu().numpy()
    with np.load(PILOT/f'exact_{index}.npz') as data:
        reference_data = {key:data[key] for key in data.files}
    points = reference_data['points']
    zero = list(map(tuple, points)).index(tuple(np.round(center, 14)))
    np.testing.assert_allclose(total, reference_data['log_numerator'][zero], rtol=0, atol=1.e-7)
    candidate_logs = {k:[] for k in p['candidate_ladder']}
    for pi, point in enumerate(points):
        terms = likelihood.log_importance_weights_tensor(window.measurements, *point, atom_indices=atoms,
            proposal_probability=np.ones_like(atoms, dtype=float), object_chunk=4, atom_chunk=4096)
        if pi == zero:
            np.testing.assert_allclose(terms.cpu().numpy(), scores, rtol=0, atol=1.e-7)
        for k in p['candidate_ladder']:
            candidate_logs[k].append(torch.logsumexp(terms[:, :k], dim=1).cpu().numpy())
    output = dict(atom_indices=atoms, center_log_contributions=scores, points=points,
                  center_log_numerator=total,
                  **{f'candidate_log_numerator_{k}':np.asarray(values) for k,values in candidate_logs.items()})
    for k in p['candidate_ladder']:
        distance = scores[:, :1]-scores[:, :k]
        local = domain.ProposalCandidates(atoms[:, :k], distance, distance[:, -1])
        for seed in domain.protocol()['proposal_seeds']:
            result = study.run_adaptive_section5(likelihood, window, FixedCandidates(proposal, local),
                center=center, h=domain.protocol()['h'], draw_ladder=domain.baseline.old.LADDER,
                n_candidates=k, proposal_prefilter_candidates=k, proposal_seed=seed,
                candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
                retain_full_ladder=True, full_information=True, object_ids=rows,
                object_chunk=4, atom_chunk=4096)
            for key in domain.baseline.old.KEYS:
                output[f'k{k}_{seed}_{key}'] = getattr(result, key)
            print(f'FULL_RANK_SAMPLED worker={index} k={k} seed={seed}', flush=True)
    path = ROOT/f'worker_{index}.npz'
    np.savez_compressed(path, **output)
    save(ROOT/f'worker_{index}.json', dict(status='complete', group=group,
        output_sha256=study.file_hash(path), protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        reference_sha256=study.file_hash(PILOT/f'exact_{index}.npz'),
        elapsed_seconds=time.monotonic()-started, exact_center_replay_passed=True))
    del likelihood, proposal
    gc.collect()
    print('FULL_RANK_WORKER_COMPLETE', index, flush=True)


def combine():
    p, original, reports = protocol(), domain.protocol(), []
    for index in range(4):
        receipt = study.read(ROOT/f'worker_{index}.json')
        assert receipt['status'] == 'complete' and receipt['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        study.verify(ROOT/f'worker_{index}.npz', receipt['output_sha256'])
        study.verify(PILOT/f'exact_{index}.npz', receipt['reference_sha256'])
        with np.load(PILOT/f'exact_{index}.npz') as data:
            exact = {name:data[name] for name in data.files}
        score, info = domain.moments(exact['points'], exact['log_selected'])
        j = receipt['group']['focal_row'] % 4
        report = dict(group=receipt['group'], exact_score=score[j].tolist(), exact_information=info[j].tolist(), candidates={})
        with np.load(ROOT/f'worker_{index}.npz') as data:
            for k in p['candidate_ladder']:
                coverage = np.exp(data[f'candidate_log_numerator_{k}'][:,j]-exact['log_numerator'][:,j])
                checks = []
                for seed in original['proposal_seeds']:
                    for rung, draws in [(-2,8192),(-1,16384)]:
                        ds = data[f'k{k}_{seed}_ladder_score'][rung,j]-score[j]
                        di = data[f'k{k}_{seed}_ladder_information'][rung,j]-info[j]
                        good = bool(np.all(abs(ds) <= original['score_atol']+original['moment_rtol']*abs(score[j]))
                            and np.all(abs(di) <= original['information_atol']+original['moment_rtol']*abs(info[j])))
                        checks.append(dict(seed=seed, draws=draws, score_error=ds.tolist(), information_error=di.tolist(), passed=good))
                report['candidates'][str(k)] = dict(minimum_stencil_mass_coverage=float(coverage.min()),
                    maximum_stencil_mass_coverage=float(coverage.max()), checks=checks, passed=all(c['passed'] for c in checks))
        reports.append(report)
    passed = {str(k):all(r['candidates'][str(k)]['passed'] for r in reports) for k in p['candidate_ladder']}
    save(ROOT/'result.json', dict(status='complete', numerical_pilot_pass=passed, reports=reports,
        limitation='Same four focal probes and unchanged numerical gate; not shear calibration, independent acceptance or default promotion.'))
    print('FULL_RANK_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker','combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            combine()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID","cpu")}.json',dict(traceback=traceback.format_exc()))
        raise
