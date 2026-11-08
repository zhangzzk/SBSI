"""Current-checkpoint measured-input counterfactual with matched scene errors."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
spec = importlib.util.spec_from_file_location('current_measured_axis_math', H.parent/'response_strategy_review_20261107/audit_axes.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    for path, digest in r.read(H/'source_pins.json').items():
        assert r.sha(path) == digest, path
    cfg = r.read(H/'manifest.json'); old = r.read(H.parent/'conditional_cut_replication_20261107/report.json')
    cases = cfg['cases']; cuts = [v[0] for v in cfg['cuts']]; raw = {}; receipts = []
    for axis in [1, 2]:
        ss = []; mm = []; gg = {a:[] for a in cfg['checkpoints']}
        for case in cases:
            p = ROOT/f'axis{axis}'/f'case{case}.npz'; receipt = r.read(p.with_suffix('.json'))
            assert receipt['data_sha256'] == r.sha(p) and receipt['source_sha256'] == r.sha(H/'evaluate.py')
            assert receipt['checkpoint_pair'] == cfg['checkpoints']
            for key in ['pair_conditional_exact','measured_photometry_exact','selected_masses_exact','simulation_replay_passed','half_bank_identity_passed']:
                assert receipt[key], key
            receipts.append(receipt)
            with np.load(p) as z:
                assert list(z['cuts']) == cuts
                s = z['sim'].sum(-2); m = z['model'][list(z['arms']).index('control')].sum(-2)
                ss.append(s); mm.append(m)
            genroot = Path(cfg['original_root']) if case in cfg['pilot_cases'] else ROOT.parent/'conditional_cut_replication_20261107'
            for a in gg:
                p = genroot/'constant'/a/f'axis{axis}'/f'case{case}.npz'; rec = r.read(p.with_suffix('.json'))
                assert rec['data_sha256'] == r.sha(p) and rec['checkpoint_sha256'] == cfg['checkpoints'][a]['sha256']
                with np.load(p) as z:
                    np.testing.assert_allclose(s, z['sim'].sum(-2), rtol=1e-12, atol=1e-8)
                    gg[a].append(z['model'].sum(-2))
        raw[axis] = (np.array(ss), np.array(mm), {a:np.array(x) for a,x in gg.items()})
    result = dict(checkpoints=cfg['checkpoints'], definition='Exact measured photometry and membership, actual usability per leg. The conditional shapes and emulator are shared by both frozen-photometry endpoints. This is a counterfactual, not a causal additive error budget.',
                  source_sha256=r.sha(__file__), receipts=receipts, cohorts={})
    lines = ['# Current-model measured-input test', '', result['definition'], '',
             'Central radius >3 pixels, magnitude <25. Errors are whole-scene jackknife SE (percentage points).', '',
             '| Cohort | Prediction | m1 (%) | m2 (%) | Half component difference (pp) |',
             '|---|---|---:|---:|---:|']
    for cohort in ['original_four','additional_thirty_six','all_forty']:
        ids = old['cohorts'][cohort]['cases']; ix = [cases.index(c) for c in ids]; n = len(ix)
        w = np.vstack([np.ones(n),np.random.default_rng(cfg['bootstrap_seed']).multinomial(n,np.ones(n)/n,10000),1-np.eye(n)])
        sim = [raw[a][0][ix] for a in [1,2]]
        samples = {'measured_shared':r.estimates(sim,[raw[a][1][ix,0] for a in [1,2]],w)}
        halves = [r.estimates(sim,[raw[a][1][ix,b] for a in [1,2]],w) for b in [1,2]]
        for arm in cfg['checkpoints']:
            for mi, mode in enumerate(r.MODES):
                name = arm+'_'+mode
                samples[name] = r.estimates(sim,[raw[a][2][arm][ix,mi] for a in [1,2]],w)
                for axis in ['1','2']:
                    for ci,cut in enumerate(cuts):
                        got = r.entry(samples[name]['m'+axis+'_percent'][:,ci],n)
                        exp = old['cohorts'][cohort]['axes'][axis][mode]['cuts'][cut][arm]['m_percent']
                        for k in ['value','jackknife_se']:
                            np.testing.assert_allclose(got[k],exp[k],rtol=1e-9,atol=1e-10)
        contrasts = {name:{k:samples['measured_shared'][k]-v for k,v in values.items()}
                     for name,values in samples.items() if name!='measured_shared'}
        rows = {}
        for ci,cut in enumerate(cuts):
            rows[cut] = dict(predictions={name:{k:r.entry(v[:,ci],n) for k,v in values.items()} for name,values in samples.items()},
                            measured_minus_generated={name:{k:r.entry(v[:,ci],n) for k,v in values.items()} for name,values in contrasts.items()},
                            measured_half_bank_difference={k:r.entry((halves[0][k]-halves[1][k])[:,ci],n) for k in halves[0]})
        rms = {a:r.entry(np.sqrt(np.mean(v['two_axis_rms_percent'][:,:9]**2,axis=1)),n) for a,v in samples.items()}
        result['cohorts'][cohort] = dict(cases=ids,cuts=rows,joint_nine_cut_rms=rms)
        center = rows['mag25_radius_gt0.60']['predictions']
        for name in ['control_true_usability_generated_selection','weighted_true_usability_generated_selection','measured_shared']:
            values = [f"{center[name][k]['value']:+.4f} ± {center[name][k]['jackknife_se']:.4f}" for k in ['m1_percent','m2_percent','half_difference_pp']]
            lines.append('| '+' | '.join([cohort,name,*values])+' |')
    for dest in [H,ROOT]:
        (dest/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (dest/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    print('DONE_CURRENT_MEASURED_REPORT',flush=True)


if __name__=='__main__':
    main()
