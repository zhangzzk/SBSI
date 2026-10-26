"""Compare exact-centre complement proposals with frozen exhaustive references."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_EXACT_TAIL_ROOT'])
PILOT = Path(os.environ['SBSI_DOMAIN_ROOT'])
RANK = Path(os.environ['SBSI_FULL_RANK_ROOT'])
sys.path.insert(0, str(PILOT))
import run_v39_domain_matched as domain  # noqa: E402
from sbsi.catalogue_sampling import ProposalDraw  # noqa: E402
from v39_exact_tail import build_tail_distribution, draw_tail  # noqa: E402

study, np, torch, save = domain.study, domain.np, domain.torch, domain.save


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    study.verify(ROOT/'v39_exact_tail.py', p['helper_sha256'])
    study.verify(PILOT/'result.json', p['reference_result_sha256'])
    study.verify(RANK/'result.json', p['ranking_result_sha256'])
    study.verify(PILOT/'run_v39_domain_matched.py', domain.protocol()['script_sha256'])
    return p


class ExactTailProposal(domain.FixedCandidates):
    def __init__(self, proposal, local, distributions, rows):
        super().__init__(proposal, local)
        self.distributions, self.rows = distributions, np.asarray(rows)

    def candidates(self, observed, *, n_candidates, **kwargs):
        assert len(observed) == 4 and n_candidates == self.local.indices.shape[1]
        return self.local

    def draw_tilted(self, candidates, observed, *, n_draws, delta, seed,
                    object_ids, temperature=1., **kwargs):
        assert len(observed) == 4 and delta == .1 and temperature == 1.
        np.testing.assert_array_equal(object_ids, self.rows)
        np.testing.assert_array_equal(candidates.indices, self.local.indices)
        draws = [draw_tail(d, n_draws=n_draws, seed=seed, object_id=int(row))
                 for d, row in zip(self.distributions, self.rows)]
        atoms, probabilities = np.array([x[0] for x in draws]), np.array([x[1] for x in draws])
        for row in range(4):
            assert not np.isin(atoms[row], candidates.indices[row]).any()
        return ProposalDraw(atoms, probabilities, np.zeros_like(atoms, bool),
            np.ones_like(atoms, bool), candidates.radius, seed,
            np.full_like(atoms, -1, dtype=np.int32))


def worker(index):
    p = protocol()
    group = study.read(PILOT/'cohort.json')['groups'][index]
    previous = study.read(RANK/f'worker_{index}.json')
    study.verify(RANK/f'worker_{index}.npz', previous['output_sha256'])
    study.verify(PILOT/f'exact_{index}.npz', previous['reference_sha256'])
    with np.load(RANK/f'worker_{index}.npz') as f:
        atoms, scores = f['atom_indices'], f['center_log_contributions']
    with np.load(PILOT/f'exact_{index}.npz') as f:
        reference = {key:f[key] for key in f.files}
    likelihood, proposal = domain.load()
    samples, _ = domain.baseline.observations()
    sample, rows = samples[group['sample']], np.asarray(group['rows'])
    window = study.MockCatalogue(sample.measurements.iloc[rows].reset_index(drop=True),
                                 sample.truth.iloc[rows].reset_index(drop=True))
    original = domain.protocol()
    center, n_atoms = original['center'], original['n_atoms']
    zero = list(map(tuple, reference['points'])).index(tuple(np.round(center, 14)))
    started = time.monotonic()
    # Retain this expensive scan so future diagnostics can reuse it.
    path = ROOT/f'center_log_contributions_{index}.npy'
    logs = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(4, n_atoms))
    total = np.full(4, -np.inf)
    for start in range(0, n_atoms, 16384):
        ids = np.arange(start, min(start+16384, n_atoms))
        values = likelihood.log_importance_weights_tensor(window.measurements, *center,
            atom_indices=ids, proposal_probability=np.ones(len(ids)), object_chunk=4, atom_chunk=4096)
        assert not torch.isnan(values).any() and not torch.isposinf(values).any()
        logs[:, start:start+len(ids)] = values.cpu().numpy()
        total = np.logaddexp(total, torch.logsumexp(values, dim=1).cpu().numpy())
        if start//16384 % 64 == 0 or start+len(ids) == n_atoms:
            print(f'EXACT_TAIL_SCAN worker={index} atoms={start+len(ids)}/{n_atoms} seconds={time.monotonic()-started:.1f}', flush=True)
    logs.flush()
    np.testing.assert_allclose(total, reference['log_numerator'][zero], rtol=0, atol=1.e-7)
    np.testing.assert_allclose(np.take_along_axis(logs, atoms, axis=1), scores, rtol=0, atol=1.e-7)
    center_sha = study.file_hash(path)
    save(ROOT/f'center_{index}.json', dict(status='complete', sha256=center_sha,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), exact_replay_passed=True,
        maximum_log_sum_error=float(np.max(abs(total-reference['log_numerator'][zero])))))
    weights = np.load(PILOT/'weights.npy', mmap_mode='r')
    outputs, diagnostics = {}, {}
    for k in p['candidate_ladder']:
        distance = scores[:, :1]-scores[:, :k]
        local = domain.ProposalCandidates(atoms[:, :k], distance, distance[:, -1])
        distributions = [build_tail_distribution(logs[row], weights, atoms[row, :k], delta=p['delta'])
                         for row in range(4)]
        diagnostics[str(k)] = [dict(log_center_tail_mass=d.log_center_mass,
            prior_tail_mass=d.prior_mass, maximum_cdf_probability_error=d.maximum_cdf_probability_error)
            for d in distributions]
        for seed in original['proposal_seeds']:
            result = study.run_adaptive_section5(likelihood, window,
                ExactTailProposal(proposal, local, distributions, rows),
                center=center, h=original['h'], draw_ladder=domain.baseline.old.LADDER,
                n_candidates=k, proposal_prefilter_candidates=k, proposal_seed=seed,
                candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=p['delta'],
                retain_full_ladder=True, full_information=True, object_ids=rows,
                object_chunk=4, atom_chunk=4096)
            for key in domain.baseline.old.KEYS:
                outputs[f'k{k}_{seed}_{key}'] = getattr(result, key)
            print(f'EXACT_TAIL_SAMPLED worker={index} k={k} seed={seed}', flush=True)
        del distributions
        gc.collect()
    output = ROOT/f'worker_{index}.npz'
    np.savez_compressed(output, **outputs)
    save(ROOT/f'worker_{index}.json', dict(status='complete', group=group,
        output_sha256=study.file_hash(output), center_sha256=center_sha,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        reference_sha256=previous['reference_sha256'], proposal_diagnostics=diagnostics,
        elapsed_seconds=time.monotonic()-started, exact_center_replay_passed=True))
    print('EXACT_TAIL_WORKER_COMPLETE', index, flush=True)


def combine():
    p, original, reports = protocol(), domain.protocol(), []
    baseline = study.read(RANK/'result.json')
    for index in range(4):
        receipt = study.read(ROOT/f'worker_{index}.json')
        assert receipt['status'] == 'complete' and receipt['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        study.verify(ROOT/f'worker_{index}.npz', receipt['output_sha256'])
        study.verify(PILOT/f'exact_{index}.npz', receipt['reference_sha256'])
        with np.load(PILOT/f'exact_{index}.npz') as f:
            score, info = domain.moments(f['points'], f['log_selected'])
        j = receipt['group']['focal_row'] % 4
        report = dict(group=receipt['group'], exact_score=score[j].tolist(),
                      exact_information=info[j].tolist(), candidates={})
        with np.load(ROOT/f'worker_{index}.npz') as data:
            for k in p['candidate_ladder']:
                checks = []
                for seed in original['proposal_seeds']:
                    for rung, draws in [(-2,8192),(-1,16384)]:
                        ds = data[f'k{k}_{seed}_ladder_score'][rung,j]-score[j]
                        di = data[f'k{k}_{seed}_ladder_information'][rung,j]-info[j]
                        good = bool(np.all(abs(ds) <= original['score_atol']+original['moment_rtol']*abs(score[j]))
                            and np.all(abs(di) <= original['information_atol']+original['moment_rtol']*abs(info[j])))
                        checks.append(dict(seed=seed, draws=draws, score_error=ds.tolist(),
                                           information_error=di.tolist(), passed=good))
                old = baseline['reports'][index]['candidates'][str(k)]
                report['candidates'][str(k)] = dict(checks=checks, passed=all(c['passed'] for c in checks),
                    previous_passed_count=sum(c['passed'] for c in old['checks']),
                    exact_candidate_mass_coverage=old['minimum_stencil_mass_coverage'])
        reports.append(report)
    passed = {str(k):all(r['candidates'][str(k)]['passed'] for r in reports) for k in p['candidate_ladder']}
    save(ROOT/'result.json', dict(status='complete', numerical_pilot_pass=passed, reports=reports,
        limitation='Four development probes and same moment gate; no shear calibration, finite-prior convergence or default promotion.'))
    print('EXACT_TAIL_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            combine()
    except Exception:
        save(ROOT/f'failure_{args.stage}_{os.environ.get("SLURM_ARRAY_TASK_ID","cpu")}.json',
             dict(traceback=traceback.format_exc()))
        raise
