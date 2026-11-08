"""Joint-axis uncertainty from existing per-scene sums; no model evaluation."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
MODES = ['predicted_usability_generated_selection', 'true_usability_generated_selection',
         'true_usability_true_selection']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def response(x, w):
    pooled = np.einsum('bc,cltq->bltq', w, x)
    means = pooled[..., 1:3]/pooled[..., :1]
    return (means[:, 0]-means[:, 1])/.04


def estimates(sim, model, w):
    rs = np.stack([response(s, w) for s in sim], axis=-1)
    total = [np.concatenate([m[..., :1], m[..., 1:3]+m[..., 3:5]], -1) for m in model]
    rm = np.stack([response(m, w) for m in total], axis=-1)
    # [resample,cut,output component,input shear axis].
    m1 = 100*(rs[..., 0, 0]/rm[..., 0, 0]-1)
    m2 = 100*(rs[..., 1, 1]/rm[..., 1, 1]-1)
    d = rs-rm
    return dict(m1_percent=m1, m2_percent=m2, mean_bias_percent=.5*(m1+m2),
                half_difference_pp=.5*(m1-m2),
                two_axis_rms_percent=np.sqrt(.5*(m1*m1+m2*m2)),
                delta_response=d, trace_half_response=.5*(d[..., 0, 0]+d[..., 1, 1]),
                diagonal_difference_half_response=.5*(d[..., 0, 0]-d[..., 1, 1]),
                symmetric_cross_response=.5*(d[..., 0, 1]+d[..., 1, 0]),
                antisymmetric_cross_response=.5*(d[..., 0, 1]-d[..., 1, 0]))


def entry(v, n):
    jk = v[-n:]
    return dict(value=v[0].tolist(), bootstrap_se=v[1:10001].std(0, ddof=1).tolist(),
                jackknife_se=np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist())


def fixture():
    # Unequal masses: pooled ratio, not the mean of per-scene ratios.
    s = np.zeros((2, 2, 1, 3)); s[..., 0] = np.array([[1, 2], [3, 5]])[..., None]
    s[:, 0, 0, 1] = [1, 6]; s[:, 1, 0, 1] = [.2, 2.5]
    got = response(s, np.ones((1, 2)))[0, 0, 0]
    expected = (7/4-2.7/7)/.04
    np.testing.assert_allclose(got, expected, rtol=1e-14)
    assert abs(got-np.mean([(.9)/.04, (2-.5)/.04])) > 1
    model = np.concatenate([s, np.zeros_like(s[..., 1:])], -1)
    # Give the other diagonal a nonzero response.
    s[..., 2] = s[..., 1]; model[..., 2] = s[..., 2]
    v = estimates([s, s], [model, model], np.ones((1, 2)))
    for k in ['m1_percent', 'm2_percent', 'half_difference_pp', 'mean_bias_percent']:
        np.testing.assert_allclose(v[k], 0, atol=1e-14)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    fixture()
    pins = read(H/'source_pins.json')
    for p, digest in pins.items():
        assert sha(p) == digest, p
    cfg = read(H.parent/'conditional_cut_replication_20261107/manifest.json')
    old = read(H.parent/'conditional_cut_replication_20261107/report.json')
    measured = read(H.parent/'joint_measured_photometry_20261106/full_report.json')
    mc = read(H.parent/'joint_measured_photometry_20261106/manifest.json')
    cases = cfg['cases']; cuts = [v[0] for v in cfg['cuts']]; central = cuts.index('mag25_radius_gt0.60')
    assert set(cases) == set(measured['cases']) and len(set(cases)) == 40
    data = {}; provenance = []; checked = 0
    for axis in [1, 2]:
        for arm in cfg['arms']:
            ss = []; xx = []
            for case in cases:
                root = Path(cfg['original_root']) if case in cfg['pilot_cases'] else ROOT/'conditional_cut_replication_20261107'
                p = root/'constant'/arm/f'axis{axis}'/f'case{case}.npz'; rec = read(p.with_suffix('.json'))
                assert sha(p) == rec['data_sha256'] and rec['checkpoint_sha256'] == cfg['checkpoints'][arm]['sha256']
                with np.load(p) as z:
                    assert list(z['cuts']) == cuts
                    ss.append(z['sim'].sum(-2)); xx.append(z['model'].sum(-2))
                provenance.append(dict(path=str(p), sha256=sha(p)))
            data[axis, arm] = np.array(ss), np.array(xx)
        for arm in mc['arms']:
            ss = []; xx = []
            for case in cases:
                p = ROOT/'joint_measured_photometry_20261106'/f'axis{axis}'/f'case{case}.npz'
                rec = read(p.with_suffix('.json'))
                assert rec['simulation_replay_passed'] and rec['selected_masses_exact']
                assert rec['models'] == mc['arms']
                with np.load(p) as z:
                    assert list(z['cuts']) == cuts
                    ai = list(z['arms']).index(arm)
                    ss.append(z['sim'].sum(-2)); xx.append(z['model'][ai, 0].sum(-2))
                if arm == 'baseline':
                    provenance.append(dict(path=str(p), sha256=sha(p)))
            data[axis, 'measured_'+arm] = np.array(ss), np.array(xx)
        for arm in ['weighted', 'measured_baseline', 'measured_nine_weak']:
            np.testing.assert_allclose(data[axis, arm][0], data[axis, 'control'][0], rtol=1e-12, atol=1e-8)
    out = dict(job=os.environ['SLURM_JOB_ID'], source_sha256=sha(__file__), source_pins=pins,
               definition='m+ = (m1+m2)/2, m- = (m1-m2)/2; descriptive same-scene contrasts, not corrections or causal components.',
               bootstrap_seed=202611071901, bootstrap_draws=10000, files=provenance, cohorts={})
    for cohort in ['original_four', 'additional_thirty_six', 'all_forty']:
        ids = old['cohorts'][cohort]['cases']; ix = [cases.index(c) for c in ids]; n = len(ix)
        w = np.vstack([np.ones(n), np.random.default_rng(out['bootstrap_seed']).multinomial(n, np.ones(n)/n, 10000), 1-np.eye(n)])
        groups = {}
        for mi, mode in enumerate(MODES):
            samples = {}
            for arm in cfg['arms']:
                s = [data[axis, arm][0][ix] for axis in [1, 2]]
                x = [data[axis, arm][1][ix, mi] for axis in [1, 2]]
                samples[arm] = estimates(s, x, w)
                for axis in ['1', '2']:
                    for ci, cut in enumerate(cuts):
                        got = entry(samples[arm]['m'+axis+'_percent'][:, ci], n)
                        expected = old['cohorts'][cohort]['axes'][axis][mode]['cuts'][cut][arm]['m_percent']
                        for k in ['value', 'jackknife_se']:
                            np.testing.assert_allclose(got[k], expected[k], rtol=1e-9, atol=1e-10); checked += 1
            groups[mode] = summarize(samples, cuts, n)
        if cohort == 'all_forty':
            samples = {}
            for arm in mc['arms']:
                s = [data[axis, 'measured_'+arm][0] for axis in [1, 2]]
                x = [data[axis, 'measured_'+arm][1] for axis in [1, 2]]
                samples[arm] = estimates(s, x, w)
                for axis in ['1', '2']:
                    for ci, cut in enumerate(cuts):
                        got = entry(samples[arm]['m'+axis+'_percent'][:, ci], n)
                        expected = measured['axes'][axis]['arms'][arm][cut]['m_percent']
                        for k in ['value', 'jackknife_se']:
                            np.testing.assert_allclose(got[k], expected[k], rtol=1e-9, atol=1e-10); checked += 1
            groups['original_models_measured_photometry'] = summarize(samples, cuts, n)
        out['cohorts'][cohort] = dict(cases=ids, groups=groups)
    out['existing_values_and_errors_replayed'] = checked
    (H/'report.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    for group, d in out['cohorts']['all_forty']['groups'].items():
        print(group)
        for arm, row in d['cuts'][cuts[central]].items():
            print(arm, {k:row[k] for k in ['mean_bias_percent', 'half_difference_pp']})
        print('joint_nine_cut_rms', d['joint_nine_cut_rms'])
    print('DONE_AXIS_AUDIT', checked, flush=True)


def summarize(samples, cuts, n):
    first, second = samples
    contrast = {k:samples[second][k]-samples[first][k] for k in samples[first]}
    rows = {cut:{a:{k:entry(v[:, ci], n) for k, v in values.items()}
                 for a, values in {**samples, 'paired_change':contrast}.items()} for ci, cut in enumerate(cuts)}
    rms = {a:np.sqrt(np.mean(v['two_axis_rms_percent'][:, :9]**2, axis=1)) for a, v in samples.items()}
    return dict(cuts=rows, contrast=second+' minus '+first,
                joint_nine_cut_rms={**{a:entry(v,n) for a,v in rms.items()}, 'paired_change':entry(rms[second]-rms[first],n)})


if __name__ == '__main__':
    main()
