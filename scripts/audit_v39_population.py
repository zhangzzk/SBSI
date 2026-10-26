"""Read-only population and saved-curvature audit; no inference weights are fitted."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
ORIGINAL = BASE/'inference_v39_mock20k_originalprior_20260926_v1'
EXPANDED = BASE/'inference_v39_mock500k_distance_20260926_v2'
MATCHED = BASE/'inference_v39_constgold20k_matched_mock_20260927_v2'
RESULT = BASE/'inference_v39_constgold20k_matched_expanded_20260927_v1'
ACTUAL = BASE/'inference_v39_constgold20k_bright20_bfgs_20260927_v1'
EXTENSION = BASE/'inference_bright_extension_20260925_v1'
PARENT = BASE/'inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/input/image_mock_manifest.json'
NAMES = ('magnitude', 'log10_circularized_radius', 'sersic_n', 'axis_ratio')
EDGES = (np.r_[-np.inf, np.arange(10., 30.01, .1), np.inf],
         np.r_[-np.inf, np.arange(-3., 2.01, .025), np.inf],
         np.r_[-np.inf, np.arange(0., 8.01, .05), np.inf],
         np.r_[-np.inf, np.arange(0., 1.001, .01), np.inf])
JOINT = ([-np.inf, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, np.inf],
         [-np.inf, -1, -.5, 0, .25, .5, 1, np.inf],
         [-np.inf, .5, 1, 2, 3, 4, 5, 6, np.inf],
         [-np.inf, .05, .2, .4, .6, .8, 1, np.inf])


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def features(zero):
    e = np.hypot(zero[:, 0], zero[:, 1])
    q = (1-e)/(1+e)
    return np.column_stack([zero[:, 3], np.log10(zero[:, 4]), zero[:, 2], q])


def outside(x):
    major = 10**x[:, 1]/np.sqrt(x[:, 3])
    return ((x[:, 2] <= .5) | (x[:, 2] >= 6) | (x[:, 3] <= .05)
            | (x[:, 3] >= 1) | (major <= .01) | (major >= 10))


class Population:
    def __init__(self):
        self.count, self.mass, self.outside_mass = 0, 0., 0.
        self.sums = np.zeros(4)
        self.shape_sum = np.zeros(2)
        self.hist = [np.zeros(len(edge)-1) for edge in EDGES]
        self.joint = np.zeros(tuple(len(edge)-1 for edge in JOINT))

    def add(self, x, weights, shape, outside_mask=None):
        assert np.isfinite(x).all() and np.isfinite(weights).all() and (weights >= 0).all()
        self.count += len(x)
        self.mass += float(weights.sum())
        self.sums += np.einsum('ni,n->i', x, weights)
        self.shape_sum += np.einsum('ni,n->i', shape, weights)
        self.outside_mass += float(weights[outside(x) if outside_mask is None else outside_mask].sum())
        for i, edge in enumerate(EDGES):
            self.hist[i] += np.histogram(x[:, i], bins=edge, weights=weights)[0]
        self.joint += np.histogramdd(x, bins=JOINT, weights=weights)[0]

    def report(self):
        return dict(count=self.count, mass=self.mass, means=dict(zip(NAMES, (self.sums/self.mass).tolist())),
                    mean_intrinsic_shape=(self.shape_sum/self.mass).tolist(), outside_fraction=self.outside_mass/self.mass)


def difference(a, b):
    return dict(binned_marginal_cdf_max_difference={name: float(np.max(np.abs(np.cumsum(ha/a.mass)-np.cumsum(hb/b.mass))))
                for name, ha, hb in zip(NAMES, a.hist, b.hist)},
                coarse_joint_total_variation=float(.5*np.abs(a.joint/a.mass-b.joint/b.mass).sum()))


def population_audit(output):
    pops = {}

    def add(name, x, w, shape, out=None):
        pops.setdefault(name, Population()).add(x, w, shape, out)
        bright = x[:, 0] < 20
        pops.setdefault(name+'_bright', Population()).add(x[bright], w[bright], shape[bright], None if out is None else out[bright])

    prior_audit = read(RESULT/'prior_audit.json')
    exact_domain = np.load(EXTENSION/'catalogue_audit_masks.npz')['within_generator_domain']
    outside_mass = float((~exact_domain).sum()*prior_audit['bright_atom_weight'])
    factor = prior_audit['bright_mass']/(prior_audit['bright_mass']-outside_mass)
    for name, root in [('original_prior', ORIGINAL), ('expanded_prior', EXPANDED)]:
        receipt = read(root/'support/manifest.json')
        for filename, expected in receipt['output_sha256'].items():
            assert digest(root/'support'/filename) == expected
        z = np.load(root/'support/zero.npy', mmap_mode='r')
        w = np.load(root/'support/weights.npy', mmap_mode='r')
        for lo in range(0, len(z), 1000000):
            hi = min(lo+1000000, len(z))
            x = features(z[lo:hi])
            bad = outside(x)
            if name == 'expanded_prior' and hi > len(z)-len(exact_domain):
                n_old = len(z)-len(exact_domain)
                start = max(lo, n_old)
                bad[start-lo:] = ~exact_domain[start-n_old:hi-n_old]
            add(name, x, w[lo:hi], z[lo:hi, :2], bad)
            if name == 'expanded_prior':
                cleaned = np.array(w[lo:hi])
                cleaned[bad] = 0
                cleaned[x[:, 0] < 20] *= factor
                add('expanded_domain_matched_counterfactual', x, cleaned, z[lo:hi, :2], bad)
        print('POPULATION_PRIOR', name, len(z), flush=True)
    manifest = read(PARENT)
    cases = []
    for rec in manifest['per_case']:
        source = rec['sources']['truth']
        assert digest(source['path']) == source['sha256']
        raw = pd.read_feather(source['path'], columns=['index_input', 'r_input', 'Re_input',
             'sersic_n_input', 'axis_ratio_input', 'position_angle_input'])
        q, major = raw.axis_ratio_input.to_numpy(), raw.Re_input.to_numpy()
        x = np.column_stack([raw.r_input, np.log10(major*np.sqrt(q)), raw.sersic_n_input, q])
        angle = np.deg2rad(raw.position_angle_input.to_numpy())
        shape = ((1-q)/(1+q))[:, None]*np.column_stack([np.cos(2*angle), np.sin(2*angle)])
        bad = ((raw.sersic_n_input <= .5) | (raw.sersic_n_input >= 6) | (q <= .05)
               | (q >= 1) | (major <= .01) | (major >= 10)).to_numpy()
        add('actual_unselected_parent', x, np.ones(len(x)), shape, bad)
        target = raw.index_input.to_numpy() < len(raw)//2
        for label, mask in [('target', target), ('other', ~target)]:
            add('actual_parent_'+label, x[mask], np.ones(mask.sum()), shape[mask], bad[mask])
        cases.append(dict(case=rec['case'], count=len(x), bright=int((x[:, 0] < 20).sum()), outside=int(bad.sum())))
        print('POPULATION_CASE', rec['case'], len(x), flush=True)
    zero = pd.read_parquet(MATCHED/'prepared/flow_zero.parquet').to_numpy()
    x = features(zero)
    add('actual_selected20k_inputs', x, np.ones(len(x)), zero[:, :2])
    ids = np.asarray(read(MATCHED/'mock/manifest.json')['selected_parent_rows'])
    add('matched_mock_retained_inputs', x[ids], np.ones(len(ids)), zero[ids, :2])
    comparisons = {}
    parent = pops['actual_unselected_parent']
    for name, pop in pops.items():
        reference = pops['actual_unselected_parent_bright'] if name.endswith('_bright') else parent
        comparisons[name] = difference(pop, reference)
    arrays = {}
    for name, pop in pops.items():
        arrays[name+'_joint'] = pop.joint
        for i, hist in enumerate(pop.hist):
            arrays[name+f'_marginal_{i}'] = hist
    np.savez_compressed(output/'population_histograms.npz', **arrays)
    save(output/'population.json', dict(status='complete', populations={k:v.report() for k,v in pops.items()},
        comparisons_to_unselected_actual_parent=comparisons, per_case=cases,
        expanded_outside_mass=outside_mass, bright_mass=prior_audit['bright_mass'],
        domain_counterfactual_bright_multiplier=factor,
        counterfactual_only='No inference weights modified; zero mass outside generator domain, preserve bright total',
        domain_rounding='New catalogue mask preserves source dtypes; original features reconstruct q from ellipticity',
        limitations=['Marginals and coarse joint bins do not prove joint population equality.',
                     'Histogram distances are descriptive, not independent-sample significance tests.',
                     'No neighbour-context distribution test in this initial scan.']))


def influence_audit(output):
    result = {}
    zero = pd.read_parquet(MATCHED/'prepared/flow_zero.parquet')
    truth = pd.read_parquet(ACTUAL/'input/truth.parquet')
    retained = np.asarray(read(MATCHED/'mock/manifest.json')['selected_parent_rows'])
    for name in ['mock', 'actual']:
        measured = pd.read_parquet(MATCHED/'mock/evaluation/measurements.parquet' if name == 'mock' else ACTUAL/'input/measurements.parquet')
        parts = []
        for i in range(4):
            path = RESULT/f'moments/{name}/worker_{i}.npz'
            assert digest(path) == read(path.with_suffix('.json'))['output_sha256']
            with np.load(path) as data:
                np.testing.assert_array_equal(data['object_rows'], np.arange(i*5000, (i+1)*5000))
                parts.append({k:data[k] for k in data.files})
        score, info = [np.concatenate([p[k] for p in parts], axis=1) for k in ['ladder_score', 'ladder_information']]
        keep = retained if name == 'mock' else np.arange(20000)
        score, info = score[:, keep], info[:, keep]
        assert np.isfinite(score).all() and np.isfinite(info).all()
        values, vectors = np.linalg.eigh(info[-1].sum(0))
        direction = vectors[:, 0]
        delta = np.einsum('i,nij,j->n', direction, info[-1]-info[-2], direction)
        projected = np.einsum('i,nij,j->n', direction, info[-1], direction)
        mag = measured.measured_mag_auto.to_numpy()[keep]
        diagnostics = {k:np.concatenate([p[k][-1] for p in parts])[keep] for k in ['weight_ess', 'weight_relative_error', 'weight_max_fraction']}
        order = np.argsort(np.abs(delta))[::-1]
        def record(j):
            row = keep[j]
            return dict(row=int(row), case=int(truth.source_case.iloc[row]), input_index=int(truth.source_input_index.iloc[row]),
                measured_magnitude=float(mag[j]), truth=zero.iloc[row].to_dict(),
                projected_information=float(projected[j]), projected_8k_to16k_change=float(delta[j]),
                diagnostics={k:float(v[j]) if np.isfinite(v[j]) else None for k,v in diagnostics.items()})
        bins = {}
        for label, mask in [('mag_lt20', mag < 20), ('mag20_22', (mag >= 20) & (mag < 22)),
                            ('mag22_24', (mag >= 22) & (mag < 24)), ('mag_ge24', mag >= 24)]:
            bins[label] = dict(count=int(mask.sum()), projected_information=float(projected[mask].sum()),
                projected_change=float(delta[mask].sum()), absolute_change=float(np.abs(delta[mask]).sum()))
        result[name] = dict(n=len(keep), eigenvalues=values.tolist(), direction=direction.tolist(),
            projected_change=float(delta.sum()), total_absolute_change=float(np.abs(delta).sum()), magnitude_bins=bins,
            largest_changes=[record(j) for j in order[:20]],
            top_absolute_change_fractions={str(k):float(np.abs(delta[order[:k]]).sum()/np.abs(delta).sum()) for k in [1,5,10,20]},
            diagnostic_only_without_top_change={str(k): np.linalg.eigvalsh(info[-1].sum(0)-info[-1, order[:k]].sum(0)).tolist() for k in [1,5,10]},
            full_draw_ladder_eigenvalues=[np.linalg.eigvalsh(i.sum(0)).tolist() for i in info])
        print('INFLUENCE', name, result[name]['top_absolute_change_fractions'], flush=True)
    save(output/'influence.json', dict(status='complete', samples=result,
        limitation='Rows retained in inference; omission and influence summaries are diagnostics only. Draw ESS is not distinct posterior-atom ESS.'))


def crowding_audit(output):
    """Compare already computed F8 crowding, without rebuilding scene geometry."""
    parent = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_closecrowd_20260924_v1/parent')
    manifest = read(parent/'manifest.json')
    names = ('nbr_close_050', 'nbr_close_100', 'nbr_close_200')
    indices = [manifest['features'].index(name) for name in names]
    edges = np.r_[-np.inf, np.arange(0., 16.001, .025), np.inf]
    joint_edges = [JOINT[0]] + [[-np.inf, .001, .01, .1, .5, 1, 2, 4, 8, np.inf]]*3
    pops = {}

    def add(label, magnitude, close, weight):
        assert np.isfinite(close).all() and (close >= 0).all()
        assert np.isfinite(weight).all() and (weight >= 0).all()
        for suffix, mask in [('', np.ones(len(weight), bool)), ('_bright', magnitude < 20)]:
            key = label+suffix
            if key not in pops:
                pops[key] = dict(mass=0., sum=np.zeros(3), count=0,
                                 hist=np.zeros((3, len(edges)-1)),
                                 joint=np.zeros(tuple(len(e)-1 for e in joint_edges)))
            rec, w, x = pops[key], weight[mask], close[mask]
            rec['mass'] += w.sum()
            rec['count'] += len(w)
            rec['sum'] += np.einsum('ni,n->i', x, w)
            for i in range(3):
                rec['hist'][i] += np.histogram(x[:, i], bins=edges, weights=w)[0]
            rec['joint'] += np.histogramdd(np.column_stack([magnitude[mask], x]), bins=joint_edges, weights=w)[0]

    sources = {str(parent/'manifest.json'): digest(parent/'manifest.json')}
    for case in manifest['cases']:
        path = parent/f'case{case:03d}.npz'
        assert digest(path) == manifest['case_reports'][str(case)]['sha256']
        with np.load(path) as data:
            z = data['context']
            assert len(z) == manifest['case_reports'][str(case)]['truth_parent_rows']
            add('unselected_target_parent', z[:, 3], z[:, indices], np.ones(len(z)))
    exact_domain = np.load(EXTENSION/'catalogue_audit_masks.npz')['within_generator_domain']
    prior = read(RESULT/'prior_audit.json')
    factor = prior['bright_mass']/(prior['bright_mass']-(~exact_domain).sum()*prior['bright_atom_weight'])
    for name, root in [('original_prior', ORIGINAL), ('expanded_prior', EXPANDED)]:
        receipt = read(root/'support/manifest.json')
        sources[str(root/'support/manifest.json')] = digest(root/'support/manifest.json')
        for filename in ('zero.npy', 'weights.npy'):
            assert digest(root/'support'/filename) == receipt['output_sha256'][filename]
        z = np.load(root/'support/zero.npy', mmap_mode='r')
        weights = np.load(root/'support/weights.npy', mmap_mode='r')
        for lo in range(0, len(z), 1000000):
            hi = min(lo+1000000, len(z))
            add(name, z[lo:hi, 3], z[lo:hi, 5:8], weights[lo:hi])
            if name == 'expanded_prior':
                w = np.array(weights[lo:hi])
                start = max(lo, prior['n_old'])
                if hi > start:
                    w[start-lo:] *= exact_domain[start-prior['n_old']:hi-prior['n_old']]
                w[z[lo:hi, 3] < 20] *= factor
                add('domain_matched_counterfactual', z[lo:hi, 3], z[lo:hi, 5:8], w)
        print('CROWDING_PRIOR', name, flush=True)
    reports, arrays = {}, {}
    for name, rec in pops.items():
        reference = pops['unselected_target_parent'+('_bright' if name.endswith('_bright') else '')]
        delta = np.max(np.abs(np.cumsum(rec['hist']/rec['mass'], axis=1)
                              -np.cumsum(reference['hist']/reference['mass'], axis=1)), axis=1)
        reports[name] = dict(count=rec['count'], mass=float(rec['mass']),
            means=dict(zip(names, (rec['sum']/rec['mass']).tolist())),
            binned_marginal_cdf_max_difference=dict(zip(names, delta.tolist())),
            magnitude_crowding_joint_total_variation=float(.5*np.abs(rec['joint']/rec['mass']-reference['joint']/reference['mass']).sum()))
        arrays[name+'_hist'] = rec['hist']
        arrays[name+'_joint'] = rec['joint']
    np.savez_compressed(output/'crowding_histograms.npz', **arrays)
    save(output/'crowding.json', dict(status='complete', populations=reports, sources=sources,
        cases=manifest['cases'], features=names, marginal_edges=edges[1:-1].tolist(),
        limitations=['Unselected target-role inputs only, cases40–119; properties of both roles match in the separate audit.',
                     'Binned descriptive comparisons do not establish full joint equality or shear calibration.',
                     'Counterfactual changes histograms only; no inference weights were modified.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--crowding-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(args.output/'execution.json', dict(script_sha256=digest(__file__), sources={str(p):digest(p) for p in
        [PARENT, RESULT/'comparison.json', RESULT/'execution.json', MATCHED/'mock/manifest.json']}))
    if args.crowding_only:
        crowding_audit(args.output)
    else:
        influence_audit(args.output)
        population_audit(args.output)
    save(args.output/'complete.json', dict(status='complete', seconds=time.monotonic()-started))
