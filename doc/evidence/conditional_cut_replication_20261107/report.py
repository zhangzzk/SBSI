"""Frozen-pair results on original four, additional thirty-six and all forty scenes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name
MODES = ['predicted_usability_generated_selection', 'true_usability_generated_selection',
         'true_usability_true_selection']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def estimates(sim, model, weights, axis):
    s = np.einsum('bc,clq->blq', weights, sim)
    x = np.einsum('bc,clq->blq', weights, model)
    def difference(numerator, denominator):
        return (numerator[:, 0] / denominator[:, 0, None]
                - numerator[:, 1] / denominator[:, 1, None]) / .04
    assert (s[..., 0] > 0).all() and (x[..., 0] > 0).all()
    rs = difference(s[..., 1:3], s[..., 0])
    flow = difference(x[..., 1:3], x[..., 0])
    emu = difference(x[..., 3:5], x[..., 0])
    model_response = difference(x[..., 1:3] + x[..., 3:5], x[..., 0])
    np.testing.assert_allclose(flow + emu, model_response, rtol=1e-12, atol=1e-12)
    delta = rs - model_response
    return dict(simulation_response=rs, model_response=model_response,
                model_flow_response=flow, model_emulator_response=emu,
                delta_response=delta,
                m_percent=100 * delta[:, axis - 1] / model_response[:, axis - 1],
                selected_count_ratio=x[..., 0] / s[..., 0])


def weights(n, seed):
    return np.vstack([np.ones(n), np.random.default_rng(seed).multinomial(
        n, np.full(n, 1 / n), 10000), np.ones((n, n)) - np.eye(n)])


def entry(values, n):
    jk = values[-n:]
    return dict(value=values[0].tolist(), case_bootstrap_se=values[1:10001].std(0, ddof=1).tolist(),
                jackknife_se=np.sqrt((n-1)/n * ((jk - jk.mean(0))**2).sum(0)).tolist())


def main():
    assert os.environ.get('SLURM_JOB_ID')
    cfg = read(H / 'manifest.json')
    raw = {}; receipts = []
    for axis in [1, 2]:
        raw[axis] = {}
        for arm in cfg['arms']:
            simulations, models = [], []
            for case in cfg['cases']:
                original = case in cfg['pilot_cases']
                root = Path(cfg['original_root']) if original else ROOT
                path = root / 'constant' / arm / f'axis{axis}' / f'case{case}.npz'
                receipt = read(path.with_suffix('.json'))
                assert sha(path) == receipt['data_sha256']
                assert receipt['checkpoint_sha256'] == cfg['checkpoints'][arm]['sha256']
                assert receipt['simulation_replay_passed'] and receipt['true_selection_mass_exact']
                assert receipt['pooled_emulator_roundoff_check_passed'] and receipt['reference_score_replay_passed']
                expected = cfg['reference_evaluator_sha256'] if original else sha(H / 'evaluate.py')
                assert receipt['source_sha256'] == expected
                receipts.append(dict(arm=arm, case=case, axis=axis, path=str(path), receipt=receipt))
                with np.load(path) as z:
                    assert list(z['cuts']) == [v[0] for v in cfg['cuts']]
                    simulations.append(z['sim'].sum(-2)); models.append(z['model'].sum(-2))
            raw[axis][arm] = (np.asarray(simulations), np.asarray(models))
        np.testing.assert_array_equal(raw[axis]['control'][0], raw[axis]['weighted'][0])
    cohorts = {'original_four': cfg['pilot_cases'], 'additional_thirty_six': cfg['replication_cases'],
               'all_forty': cfg['cases']}
    result = dict(definition='m=100(Rsim/Rmodel-1); per-leg normalized antithetic h=0.02. Whole-case paired errors, fixed models/draws/measurements. Additional scenes are replication within existing development data, not independent acceptance.',
                  manifest_sha256=sha(H/'manifest.json'), source_sha256=sha(__file__), checkpoints=cfg['checkpoints'],
                  receipts=receipts, cohorts={})
    lines = ['# Frozen-model replication', '', result['definition'], '',
             'Central cut: FLUX_RADIUS >3px, MAG_AUTO <25. Predicted usability/generated selection. ± values below are paired case jackknife SE in percentage points.', '',
             '| Cohort | Control m1 (%) | New m1 (%) | Change m1 (pp) | Control m2 (%) | New m2 (%) | Change m2 (pp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    original_report = read(H.parent/'conditional_cut_probability_20261107/report.json')
    for cohort, cases in cohorts.items():
        ix = [cfg['cases'].index(case) for case in cases]; n = len(ix)
        w = weights(n, cfg['bootstrap_seed']); output = dict(cases=cases, axes={}); central = []
        for axis in [1, 2]:
            modes = {}
            for mode_index, mode in enumerate(MODES):
                rows = {}; values_by_arm = {a: [] for a in cfg['arms']}
                for ci, cut in enumerate([v[0] for v in cfg['cuts']]):
                    samples = {}; per_case = {}; row = {}
                    for arm in cfg['arms']:
                        sim, model = raw[axis][arm]; samples[arm] = estimates(sim[ix, :, ci], model[ix, mode_index, :, ci], w, axis)
                        per_case[arm] = estimates(sim[ix, :, ci], model[ix, mode_index, :, ci], np.eye(n), axis)['m_percent']
                        row[arm] = {key: entry(value, n) for key, value in samples[arm].items()}
                        row[arm]['per_case_m_percent'] = per_case[arm].tolist()
                        if cut in cfg['target_cuts']: values_by_arm[arm].append(samples[arm]['m_percent'])
                        if cohort == 'original_four':
                            old = original_report['axes'][str(axis)]['arms'][arm][mode][cut]
                            for key, value in row[arm].items():
                                if key == 'per_case_m_percent': continue
                                np.testing.assert_allclose(value['value'], old[key]['value'], rtol=1e-10, atol=1e-12)
                                np.testing.assert_allclose(value['jackknife_se'], old[key]['jackknife_se'], rtol=1e-9, atol=1e-12)
                    row['paired_change'] = {key: entry(samples['weighted'][key]-samples['control'][key], n)
                                            for key in samples['control']}
                    row['absolute_bias_change'] = entry(abs(samples['weighted']['m_percent'])-abs(samples['control']['m_percent']), n)
                    rows[cut] = row
                    if ci == 4 and mode_index == 0:
                        central.extend([f"{row[a]['m_percent']['value']:+.4f}" for a in cfg['arms']])
                        delta = row['paired_change']['m_percent']; central.append(f"{delta['value']:+.4f} ± {delta['jackknife_se']:.4f}")
                rms = {a: np.sqrt(np.square(np.array(v)).mean(0)) for a, v in values_by_arm.items()}
                modes[mode] = dict(cuts=rows, nine_cut_rms_m_percent={**{a:entry(v,n) for a,v in rms.items()},
                                                                  'paired_change':entry(rms['weighted']-rms['control'],n)},
                                   improved_target_cuts=sum(rows[c]['absolute_bias_change']['value']<0 for c in cfg['target_cuts']))
            output['axes'][str(axis)] = modes
        result['cohorts'][cohort] = output
        lines.append('| '+cohort+' | '+' | '.join(central)+' |')
    for destination in [H, ROOT]:
        (destination/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (destination/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    print('DONE_REPLICATION_REPORT',flush=True)
