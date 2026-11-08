#!/usr/bin/env python
"""Frozen ConstGold forward-model comparison; run using the training snapshot venv.

Reuse hash-verified measured rows and unchanged C8 probabilities. Regenerate
candidate F8/disk draws with the original case/block seeds and batch dimensions.
The response uses antithetic g1=+/-0.02, independent per-leg measured cuts,
and case-bootstrap uncertainty. No fitting or checkpoint selection occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(root):
    p = json.loads((root / 'protocol.json').read_text())
    for name, expected in p['hashes'].items():
        if digest(name) != expected:
            raise ValueError(f'frozen input changed: {name}')
    return p


def sums(weight, mass, shape, bins):
    rows = np.concatenate(((weight * mass)[:, None], weight[:, None] * shape), axis=1)
    return np.stack([rows[bins == b].sum(0) for b in range(4)])


def evaluate(root, p, chunk):
    import torch
    from scripts import evaluate_smooth_classifier_joint as ev
    from scripts import joint_response_axes as axes
    from plots.constgold_property_bias import load_lookup
    from sbsi.disk_response_transport import disk_response_transport
    from sbsi.flow_guard_response import physical_context_draws
    from sbsi.measurement_model import load_measurement_model
    from sbsi.output_conditioned_response import DiskResponseMoment, predict_scene_draw_responses

    torch.set_num_threads(4)
    cases = tuple(range(40 + 10 * chunk, 50 + 10 * chunk))
    parent, population, baseline = map(Path, (p['parent'], p['population'], p['baseline']))
    manifest = json.loads((parent / 'manifest.json').read_text())
    old_hashes = {r['case']: r['sha256'] for r in
                  json.loads((baseline / 'development_comparison.json').read_text())['row_replay']}
    old_classifier = {r['case']: r for r in json.loads(
        (Path(p['classifier_study']) / f'evaluation_{chunk}.json').read_text())['records']}
    lookup = load_lookup((population / f'rblend/cases{cases[0]}_{cases[-1]}.feather',), cases)
    emulator = DiskResponseMoment(Path(p['emulator']), nthread=4, device='cuda')
    if not emulator.conditioned or emulator.response_coordinate != 'disk_velocity':
        raise ValueError('expected output-conditioned disk velocity emulator')
    bundle = load_measurement_model(p['flow'], device=torch.device('cuda'))
    bundle.model.eval()
    print('FORWARD_START', chunk, torch.cuda.get_device_name(0), flush=True)
    for case in cases:
        output = root / 'cases' / f'case{case:03d}.json'
        if output.exists():
            raise FileExistsError(output)
        started = time.monotonic()
        path = parent / f'case{case:03d}.npz'
        if digest(path) != manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('parent changed')
        with np.load(path, allow_pickle=False) as z:
            ids, context = z['input_index'], z['context'].astype(np.float64)
        rows_path = baseline / f'evaluation/chunk{chunk}/rows/case{case:03d}.npz'
        if digest(rows_path) != old_hashes[case]:
            raise ValueError('baseline measured/model rows changed')
        with np.load(rows_path, allow_pickle=False) as z:
            reference = dict(z)
        qpath = Path(p['c8_probabilities']) / f'case{case:03d}.npz'
        with np.load(qpath, allow_pickle=False) as z:
            probabilities = dict(z)
        if not (np.array_equal(ids, reference['input_index'])
                and np.array_equal(context, reference['context'])
                and np.array_equal(ids, probabilities['input_index'])
                and np.array_equal(ids, lookup[case]['input_index'])
                and np.all(ids[1:] > ids[:-1])):
            raise ValueError('parent/saved/C8/lookup identities differ')
        pairs, pair_metadata = ev.load_candidate_pairs(
            population / 'pairs', case,
            manifest['case_reports'][str(case)].get('eight_input_source_sha256', digest(path)),
            ids, lookup[case]['R_blend'])
        pair_ids = pairs['index_input_p'].to_numpy()
        bins = np.digitize(context[:, 3], [24, 25.5, 27.5])
        names = ('measured', 'old_C8', 'new_C8', 'old_true_flags', 'new_true_flags')
        totals = {name: np.zeros((2, 4, 3)) for name in names}
        replay = []
        for li, leg in enumerate(('plus', 'minus')):
            q = probabilities[f'probability_{leg}']
            if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                raise ValueError('invalid C8 probabilities')
            # Revalidate cached C8 against original sums for all cuts and bins.
            old_shape = reference[f'flow_{leg}'] + reference[f'blend_{leg}']
            replay.append(np.stack([sums(q, reference[f'mass_{leg}'][:, k],
                                           old_shape[:, k], bins) for k in range(3)]))
            values, usable = reference[f'measured_{leg}'], reference[f'usable_{leg}']
            selected = usable & ev.output_cuts(values)[:, 1]
            totals['measured'][li] = sums(selected.astype(float), np.ones(len(ids)),
                                         np.where(selected[:, None], values[:, :2], 0), bins)
            for name, w in (('C8', q), ('true_flags', usable.astype(float))):
                totals[f'old_{name}'][li] = sums(w, reference[f'mass_{leg}'][:, 1], old_shape[:, 1], bins)
        expected = np.asarray(old_classifier[case]['sums']['C8_20260913'])
        np.testing.assert_allclose(replay, expected, rtol=1e-10, atol=1e-6)
        replay_error = float(np.max(np.abs(np.asarray(replay) - expected)))
        frames = axes.shear_frames(context, .02, 1)
        contexts = [bundle.context_tensor(f) for f in frames]
        with torch.no_grad():
            for block, start in enumerate(range(0, len(ids), 1024)):
                stop = min(len(ids), start + 1024)
                generated = physical_context_draws(bundle.model, [x[start:stop] for x in contexts],
                    64, 7301 + 10000019 * case + 1000003 * block)
                lo, hi = np.searchsorted(pair_ids, [ids[start], ids[stop-1]], side='left')
                hi = np.searchsorted(pair_ids, ids[stop-1], side='right')
                for li, (leg, values) in enumerate(zip(('plus', 'minus'), generated)):
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError('nonfinite generated measurements')
                    response, _ = predict_scene_draw_responses(emulator, pairs.iloc[lo:hi],
                                                               ids[start:stop], values.cpu().numpy())
                    shear = torch.tensor([.02 if li == 0 else -.02, 0.], dtype=torch.float64, device='cuda')
                    velocity = torch.as_tensor(response, dtype=torch.float64, device='cuda')[..., None] * shear
                    transformed, logdet = disk_response_transport(values[..., :2], velocity)
                    if not bool(torch.isfinite(transformed).all() and torch.isfinite(logdet).all()):
                        raise ValueError('nonfinite disk transport')
                    mass, flow, blend = (x.cpu().numpy() for x in
                        ev.selected_draw_moments(values, transformed - values[..., :2]))
                    for name, w in (('C8', probabilities[f'probability_{leg}']),
                                    ('true_flags', reference[f'usable_{leg}'].astype(float))):
                        totals[f'new_{name}'][li] += sums(w[start:stop], mass[:, 1],
                            flow[:, 1] + blend[:, 1], bins[start:stop])
                if block % 32 == 0:
                    print(f'FORWARD_PROGRESS case={case} rows={stop}/{len(ids)}', flush=True)
        write(output, dict(case=case, protocol_sha256=digest(root / 'protocol.json'),
            parent_rows=len(ids), saved_rows_sha256=old_hashes[case], c8_replay_max_abs=replay_error,
            pair_metadata=pair_metadata, sums={k: v.tolist() for k, v in totals.items()},
            elapsed_seconds=time.monotonic() - started))
        print(f'FORWARD_CASE_DONE case={case} seconds={time.monotonic()-started:.1f}', flush=True)


def summarize(root, p):
    records = [json.loads((root / 'cases' / f'case{case:03d}.json').read_text()) for case in range(40, 120)]
    if any(r['protocol_sha256'] != digest(root / 'protocol.json') for r in records):
        raise ValueError('mixed protocols')
    arrays = {name: np.stack([r['sums'][name] for r in records], axis=1) for name in records[0]['sums']}
    # Shape is leg, case, magnitude bin, (selected mass, selected e1, selected e2).
    def response(s):
        mean = s[..., 1:] / s[..., :1]
        return (mean[0] - mean[1]) / .04
    weights = np.random.default_rng(20260925).multinomial(80, np.full(80, 1/80), size=10000)
    collapsed = {name: v.sum(2) for name, v in arrays.items()}
    central = {name: response(v.sum(1)) for name, v in collapsed.items()}
    boots = {name: response(np.einsum('bc,lck->lbk', weights, v))[:, 0]
             for name, v in collapsed.items()}
    results, mdraws = {}, {}
    for name in ('old_C8', 'new_C8', 'old_true_flags', 'new_true_flags'):
        m = 100 * (central['measured'][0] / central[name][0] - 1)
        mdraws[name] = 100 * (boots['measured'] / boots[name] - 1)
        # Contributions use the total selected denominator on each leg, so bins sum to R.
        def contributions(a):
            s = a.sum(1)
            v = s[..., 1:] / s[:, :, :1].sum(1)[:, None, :]
            return (v[0] - v[1]) / .04
        by_bin = 100 * (contributions(arrays['measured'])[:, 0]
                       - contributions(arrays[name])[:, 0]) / central[name][0]
        np.testing.assert_allclose(by_bin.sum(), m, atol=1e-9, rtol=0)
        results[name] = dict(m_percent=float(m), case_se_pp=float(mdraws[name].std(ddof=1)),
            ci95_pp=np.quantile(mdraws[name], [.025, .975]).tolist(),
            data_response=central['measured'].tolist(), model_response=central[name].tolist(),
            selected_model=collapsed[name].sum(1)[:, 0].tolist(),
            selected_data=collapsed['measured'].sum(1)[:, 0].tolist(),
            magnitude_bin_bias_contributions_pp=by_bin.tolist(),
            sum_absolute_bin_contributions_pp=float(np.abs(by_bin).sum()),
            point_estimate_within_0p3_percent=bool(abs(m) <= .3))
    expected = p['published_control']
    np.testing.assert_allclose([results['old_C8']['m_percent'], results['old_C8']['case_se_pp']],
                              [expected['m_percent'], expected['case_se_pp']], rtol=0, atol=1e-9)
    for arm in ('C8', 'true_flags'):
        draws = mdraws[f'new_{arm}'] - mdraws[f'old_{arm}']
        results[f'new_minus_old_{arm}'] = dict(
            m_difference_pp=results[f'new_{arm}']['m_percent'] - results[f'old_{arm}']['m_percent'],
            paired_case_se_pp=float(draws.std(ddof=1)), ci95_pp=np.quantile(draws, [.025, .975]).tolist())
    write(root / 'summary.json', dict(protocol_sha256=digest(root / 'protocol.json'), results=results,
        published_control_reproduced=True, independent_acceptance=False,
        limitations=p['limitations']))
    for name, r in results.items():
        print(name, json.dumps(r), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--chunk', type=int, choices=range(8))
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    if args.summary == (args.chunk is not None):
        parser.error('choose one chunk or summary')
    p = verify(args.root)
    if args.summary:
        summarize(args.root, p)
    else:
        evaluate(args.root, p, args.chunk)


if __name__ == '__main__':
    main()
