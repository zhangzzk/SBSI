"""Exhaustive atom sums for two influential observations and their original groups."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_POPULATION_PROFILE_ROOT'])
SOURCE = Path(os.environ['SBSI_EXPANDED_ROOT'])
sys.path.insert(0, str(SOURCE))
import run_v39_expanded_prior as driver  # noqa: E402

study, np, pd, torch, save = driver.study, driver.np, driver.pd, driver.torch, driver.save
SAMPLES = (('actual', 1958), ('mock', 4460))
COMPONENTS = ('old_nonbright', 'old_bright', 'new_in_domain', 'new_outside_domain')


def worker(index):
    p = study.read(ROOT/'protocol.json')
    assert p['script_sha256'] == study.file_hash(__file__)
    likelihood, proposal = driver.load()
    del proposal
    gc.collect()
    torch.cuda.empty_cache()
    samples, gen = driver.observations()
    n = driver.protocol()['n_atoms']
    chunk = 16384
    blocks = (n+chunk-1)//chunk
    lo, hi = [min(n, chunk*(blocks*j//p['workers'])) for j in (index, index+1)]
    center = gen['center']
    points = [(round(center[0]+delta, 14), round(center[1], 14)) for delta in [-.001, 0., .001]]
    n_old = 20231221
    domain = np.load(study.BASE/'inference_bright_extension_20260925_v1/catalogue_audit_masks.npz')['within_generator_domain']
    zero = likelihood.cache.zero_flow
    groups = {name:samples[name].measurements.iloc[row//4*4:row//4*4+4].reset_index(drop=True) for name,row in SAMPLES}
    sums = np.full((2, 3, 4, 4), -np.inf)
    second = np.full((2, 3, 4), -np.inf)
    maximum = np.full((2, 3, 4), -np.inf)
    started = time.monotonic()
    for start in range(lo, hi, chunk):
        atoms = np.arange(start, min(start+chunk, hi))
        old = atoms < n_old
        bright = zero.r_input_p.to_numpy()[atoms] < 20
        good = np.zeros(len(atoms), bool)
        good[~old] = domain[atoms[~old]-n_old]
        masks = (old & ~bright, old & bright, ~old & good, ~old & ~good)
        for si, (name, _) in enumerate(SAMPLES):
            for pi, point in enumerate(points):
                terms = likelihood.log_importance_weights_tensor(groups[name], *point,
                    atom_indices=atoms, proposal_probability=np.ones(len(atoms)), object_chunk=4, atom_chunk=4096)
                assert not torch.isnan(terms).any() and not torch.isposinf(terms).any()
                for ci, mask in enumerate(masks):
                    if mask.any():
                        value = torch.logsumexp(terms[:, torch.as_tensor(mask, device=terms.device)], dim=1).cpu().numpy()
                        sums[si, pi, ci] = np.logaddexp(sums[si, pi, ci], value)
                second[si, pi] = np.logaddexp(second[si, pi], torch.logsumexp(2*terms, dim=1).cpu().numpy())
                maximum[si, pi] = np.maximum(maximum[si, pi], terms.max(1).values.cpu().numpy())
        if (start-lo)//chunk % 32 == 0 or start+chunk >= hi:
            print(f'EXACT_PROFILE worker={index} atoms={min(start+chunk,hi)-lo}/{hi-lo} seconds={time.monotonic()-started:.1f}', flush=True)
    path = ROOT/f'worker_{index}.npz'
    np.savez(path, log_sums=sums, log_second=second, log_maximum=maximum)
    save(ROOT/f'worker_{index}.json', dict(status='complete', atom_range=[lo, hi], points=points,
        components=COMPONENTS, output_sha256=study.file_hash(path), source_execution=driver.identity(),
        script_sha256=study.file_hash(__file__), elapsed_seconds=time.monotonic()-started))


def combine():
    protocol = study.read(ROOT/'protocol.json')
    parts, receipts = [], []
    end = 0
    for i in range(protocol['workers']):
        rec = study.read(ROOT/f'worker_{i}.json')
        assert rec['status'] == 'complete' and rec['atom_range'][0] == end
        assert rec['script_sha256'] == study.file_hash(__file__)
        study.verify(ROOT/f'worker_{i}.npz', rec['output_sha256'])
        end = rec['atom_range'][1]
        receipts.append(rec)
        with np.load(ROOT/f'worker_{i}.npz') as data:
            parts.append({k:data[k] for k in data.files})
    assert end == 28012032 and all(rec['points'] == receipts[0]['points'] for rec in receipts)
    sums = np.logaddexp.reduce([p['log_sums'] for p in parts])
    second = np.logaddexp.reduce([p['log_second'] for p in parts])
    maximum = np.maximum.reduce([p['log_maximum'] for p in parts])
    audit = study.read(ROOT.parent/'audit/population.json')
    n_old_bright = study.read(SOURCE/'prior_audit.json')['n_old_bright']
    multiplier = (n_old_bright+7780811)/n_old_bright
    log_expanded = np.logaddexp.reduce(sums, axis=2)
    log_original = np.logaddexp(sums[:, :, 0], sums[:, :, 1]+np.log(multiplier))
    log_cleaned = np.logaddexp(sums[:, :, 0], np.logaddexp(sums[:, :, 1], sums[:, :, 2])
                             +np.log(audit['domain_counterfactual_bright_multiplier']))
    norms = {
        'expanded': study.ExactPopulationNormalization.from_payload(study.read(SOURCE/'normalization.json')),
        'original': study.ExactPopulationNormalization.from_payload(study.read(driver.matched.actual_root()/'pass_00/normalization.json'))}
    results = {}
    for si, (name, row) in enumerate(SAMPLES):
        local = row % 4
        result = dict(parent_row=row, original_group_start=row//4*4,
            expanded_center_component_fractions=np.exp(sums[si, 1, :, local]-log_expanded[si, 1, local]).tolist(),
            expanded_center_distinct_atom_ess=float(np.exp(2*log_expanded[si,1,local]-second[si,1,local])),
            expanded_center_max_atom_fraction=float(np.exp(maximum[si,1,local]-log_expanded[si,1,local])))
        for label, values in [('expanded', log_expanded), ('original', log_original), ('domain_matched_numerator_only', log_cleaned)]:
            logs = values[si, :, local]
            if label in norms:
                logs = logs-np.array([norms[label].log_mass(*point) for point in receipts[0]['points']])
            result[label] = dict(log_profile=logs.tolist(), score_g1=float((logs[2]-logs[0])/.002),
                information_11=float(-(logs[2]-2*logs[1]+logs[0])/1.e-6))
        with np.load(SOURCE/f'moments/{name}/worker_{row//5000}.npz') as data:
            j = row % 5000
            result['importance_8k'] = dict(score_g1=float(data['ladder_score'][-2,j,0]), information_11=float(data['ladder_information'][-2,j,0,0]))
            result['importance_16k'] = dict(score_g1=float(data['ladder_score'][-1,j,0]), information_11=float(data['ladder_information'][-1,j,0,0]))
        results[name] = result
    save(ROOT/'result.json', dict(status='complete', samples=results, components=COMPONENTS,
        n_atoms=end, points=receipts[0]['points'],
        limitations=['Exhaustive numerator sums, at the same h=.001 and frozen floating-point density implementation.',
                     'Original/expanded use their existing64-draw selection normalizations; these are not analytic selection integrals.',
                     'Domain-matched profile is numerator-only: no corrected normalization or shear estimator is claimed.',
                     'Two influential observations chosen after inspecting instability; diagnostic, not representative acceptance.']))
    print('EXACT_PROFILE_COMPLETE', results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'worker':
        worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:
        combine()
