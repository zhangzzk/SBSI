"""Independent small-sum check: direct pooling and leave-one-scene-out errors."""
import hashlib
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response(sums, columns):
    means = sums[..., columns] / sums[..., :1]
    return ((means[0] - means[1]) / .04).T


def metrics(sim, model, axis):
    rs = response(sim, slice(1, 3))
    rf = response(model, slice(1, 3))
    re = response(model, slice(3, 5))
    rm = rf + re
    return dict(simulation_response=rs, model_response=rm,
                model_flow_response=rf, model_emulator_response=re,
                delta_response=rs-rm, m_percent=100*(rs[axis-1]/rm[axis-1]-1),
                selected_count_ratio=model[..., 0]/sim[..., 0])


def main():
    cfg = read(H/'manifest.json')
    ref = read(H/'report.json')
    for path, digest in read(H/'source_pins.json').items():
        assert sha(path) == digest, path
    for arm in cfg['arms']:
        assert sha(cfg['checkpoints'][arm]['path']) == cfg['checkpoints'][arm]['sha256']
    raw = {}
    matched = []
    for item in ref['receipts']:
        path = Path(item['path']); receipt = read(path.with_suffix('.json'))
        assert receipt == item['receipt']
        assert sha(path) == receipt['data_sha256']
        assert receipt['checkpoint_sha256'] == cfg['checkpoints'][item['arm']]['sha256']
        with np.load(path) as z:
            raw[item['axis'], item['arm'], item['case']] = (z['sim'].sum(-2), z['model'].sum(-2))
        if item['arm'] == 'control':
            matched.extend(receipt['measured'])
    assert len(raw) == 160
    assert len({r['receipt']['classifier_sha256'] for r in ref['receipts']}) == 1
    assert len({r['receipt']['emulator_sha256'] for r in ref['receipts']}) == 1
    checked = 0; max_error = 0.; cohorts = {}

    def check(values, saved):
        nonlocal checked, max_error
        v = np.asarray(values); n = len(v)-1
        err = np.sqrt((n-1)/n * np.square(v[1:]-v[1:].mean(0)).sum(0))
        for got, expected in [(v[0], saved['value']), (err, saved['jackknife_se'])]:
            np.testing.assert_allclose(got, expected, rtol=1e-8, atol=2e-11)
            max_error = max(max_error, float(np.max(np.abs(got-expected))))
            checked += np.size(got)

    lines = ['# All target cuts in the frozen-model replication', '',
             'Predicted usability and generated selection. m=100(Rsim/Rmodel−1);',
             'each leg normalized separately, antithetic ±0.02. All numbers are percent;',
             'paired errors are whole-scene jackknife SE in percentage points.', '']
    for cohort, saved_cohort in ref['cohorts'].items():
        cases = saved_cohort['cases']; n = len(cases); cohorts[cohort] = {}
        subsets = [list(range(n))] + [[k for k in range(n) if k != omit] for omit in range(n)]
        lines += ['## '+cohort, '', '| Radius (px) | Magnitude | m1 control → new | Δm1 ± SE | m2 control → new | Δm2 ± SE |',
                  '|---|---|---:|---:|---:|---:|']
        for axis, modes in saved_cohort['axes'].items():
            ai = int(axis); cohorts[cohort][axis] = {}
            for mi, (mode, saved_mode) in enumerate(modes.items()):
                samples = {}; per_case = {}
                for arm in cfg['arms']:
                    s = np.array([raw[ai, arm, c][0] for c in cases])
                    x = np.array([raw[ai, arm, c][1][mi] for c in cases])
                    samples[arm] = [metrics(s[ix].sum(0), x[ix].sum(0), ai) for ix in subsets]
                    per_case[arm] = np.array([metrics(ss, xx, ai)['m_percent'] for ss, xx in zip(s, x)])
                for ci, (cut, row) in enumerate(saved_mode['cuts'].items()):
                    for key in samples['control'][0]:
                        # Response arrays are [component,cut]; masses are [leg,cut].
                        values = {a:np.array([v[key][..., ci] for v in samples[a]]) for a in cfg['arms']}
                        for arm, v in values.items():
                            check(v, row[arm][key])
                        check(values['weighted']-values['control'], row['paired_change'][key])
                    check(abs(np.array([v['m_percent'][ci] for v in samples['weighted']]))-
                          abs(np.array([v['m_percent'][ci] for v in samples['control']])), row['absolute_bias_change'])
                    for arm in cfg['arms']:
                        np.testing.assert_allclose(per_case[arm][:, ci], row[arm]['per_case_m_percent'], rtol=1e-9, atol=1e-10)
                rms = {a:np.sqrt(np.mean([v['m_percent'][:9]**2 for v in samples[a]], axis=1)) for a in cfg['arms']}
                for a, v in rms.items():
                    check(v, saved_mode['nine_cut_rms_m_percent'][a])
                check(rms['weighted']-rms['control'], saved_mode['nine_cut_rms_m_percent']['paired_change'])
                delta = per_case['weighted'][:, 4]-per_case['control'][:, 4]
                prms = {a:np.sqrt(np.mean(v[:, :9]**2, axis=1)) for a, v in per_case.items()}
                cohorts[cohort][axis][mode] = dict(central_downward_cases=int((delta<0).sum()),
                    central_change_range_pp=[float(delta.min()), float(delta.max())],
                    individual_nine_cut_rms_improved=int((prms['weighted']<prms['control']).sum()),
                    case_count=n)
        for ci, (_, radius, mag) in enumerate(cfg['cuts'][:9]):
            cells = [str(radius[1]), str(mag)]
            for axis in ['1', '2']:
                row = next(iter(saved_cohort['axes'][axis].values()))['cuts'][cfg['cuts'][ci][0]]
                cells += [f"{row['control']['m_percent']['value']:+.4f} → {row['weighted']['m_percent']['value']:+.4f}",
                          f"{row['paired_change']['m_percent']['value']:+.4f} ± {row['paired_change']['m_percent']['jackknife_se']:.4f}"]
            lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    audit = dict(statistic_scalars_checked=checked, maximum_absolute_difference=max_error,
                 outputs_verified=len(raw), source_report_sha256=sha(H/'report.json'),
                 source_code_sha256=sha(__file__), cohorts=cohorts,
                 matching_totals_unique_case_axis_legs={k:sum(v[k] for v in matched) for k in matched[0]})
    (H/'completed_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (H/'all_cuts.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
