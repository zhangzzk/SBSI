"""Reassess the frozen training pair and paired cut contrasts after remeasurement."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name
REF = H.parent / 'conditional_cut_replication_20261107'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(q):
    g = q[..., 1:3] / q[..., :1]
    if q.shape[-1] == 5:
        g = g + q[..., 3:5] / q[..., :1]
    return np.stack([(g[:, 0] - g[:, 1])/.04, (g[:, 2] - g[:, 3])/.04], axis=-1)
def diag(q):
    return np.stack([q[..., 0, 0], q[..., 1, 1]], -1)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    report = json.loads((H/'report.json').read_text())
    audit = json.loads((H/'completed_audit.json').read_text())
    assert sha(H/'report.json') == audit['report_sha256']
    assert sha(ROOT/'case_sums.npz') == report['case_sums_sha256']
    cfg = json.loads((REF/'manifest.json').read_text())
    with np.load(ROOT/'case_sums.npz') as z:
        old, new, weighted, cases, cuts = [z[k] for k in ['old', 'new', 'model', 'cases', 'cuts']]
    np.testing.assert_array_equal(cases, cfg['cases'])
    control = []
    receipts = []
    for case in cases:
        legs = []
        base = Path(cfg['original_root']) if case in cfg['pilot_cases'] else ROOT.parent/REF.name
        for axis in [1, 2]:
            p = base/'constant/control'/f'axis{axis}/case{case}.npz'
            r = json.loads(p.with_suffix('.json').read_text())
            assert sha(p) == r['data_sha256']
            assert r['checkpoint_sha256'] == cfg['checkpoints']['control']['sha256']
            assert r['source_sha256'] == (cfg['reference_evaluator_sha256'] if case in cfg['pilot_cases'] else sha(REF/'evaluate.py'))
            assert r['case'] == case and r['axis'] == axis and r['arm'] == 'control'
            assert r['emulator_sha256'] == 'cd32781db7edd2cab576a2ef7478d722df2984788bc83a7fe12caa6adf19fb43'
            assert r['classifier_sha256'] == 'ad5c05d2dcd9af9d711b18b532cd681ac5de6df6350bab288cdec94a3019a3c6'
            for gate in ['simulation_replay_passed','true_selection_mass_exact','pooled_emulator_roundoff_check_passed','reference_score_replay_passed']:
                assert r[gate]
            with np.load(p) as z:
                np.testing.assert_array_equal(cuts, z['cuts'][:9])
                np.testing.assert_allclose(old[len(control), (axis-1)*2:axis*2], z['sim'].sum(-2)[:, :9], rtol=1e-12, atol=1e-8)
                legs.append(z['model'].sum(-2)[0, :, :9])
            receipts.append({'path': str(p), 'sha256': r['data_sha256']})
        control.append(np.concatenate(legs))
    n = len(cases)
    weights = np.vstack([np.ones(n), np.random.default_rng(202611061020).multinomial(n, np.full(n, 1/n), 10000), 1-np.eye(n)])
    rs, ro, rw, rc = [matrix(np.einsum('bc,clkd->blkd', weights, x)) for x in [new, old, weighted, np.array(control)]]
    mw, mc, oc = [100*(diag(a)/diag(b)-1) for a, b in [(rs,rw),(rs,rc),(ro,rc)]]
    def entry(x):
        jk = x[-n:]
        return {'value':x[0].tolist(), 'jackknife_se':np.sqrt((n-1)/n*((jk-jk.mean(0))**2).sum(0)).tolist(), 'bootstrap_ci95':np.quantile(x[1:10001],[.025,.975],axis=0).tolist()}
    original = json.loads((REF/'report.json').read_text())
    output = {'definition':'Same40 scenes and independent per-leg selection; m=100*(Rmeasured/Rmodel-1). Paired whole-scene errors. Frozen control vs probability-loss endpoint. Nine correlated cuts are not nine independent replications. Cut contrasts describe threshold sensitivity, not causal attribution to a model component.', 'receipts':receipts, 'source_sha256':sha(__file__), 'cuts':{}, 'contrasts':{}}
    for i, cut in enumerate(cuts):
        for axis in [1,2]:
            ref = original['cohorts']['all_forty']['axes'][str(axis)]['predicted_usability_generated_selection']['cuts'][str(cut)]['control']['m_percent']
            calc = entry(oc[:,i,axis-1])
            np.testing.assert_allclose([calc['value'],calc['jackknife_se']], [ref['value'],ref['jackknife_se']], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(entry(mw[:,i])['value'],report['cuts'][str(cut)]['new_m_percent']['value'],rtol=1e-10,atol=1e-10)
        output['cuts'][str(cut)] = {'control_m_percent':entry(mc[:,i]), 'weighted_m_percent':entry(mw[:,i]), 'weighted_minus_control_pp':entry(mw[:,i]-mc[:,i]), 'weighted_mean_m_percent':entry(mw[:,i].mean(-1)), 'weighted_half_difference_pp':entry((mw[:,i,0]-mw[:,i,1])/2)}
    rms_c, rms_w = [np.sqrt((x*x).mean((1,2))) for x in [mc,mw]]
    output['joint_nine_cut_rms_percent'] = {'control':entry(rms_c),'weighted':entry(rms_w),'weighted_minus_control_pp':entry(rms_w-rms_c)}
    # Predefined adjacent radius contrasts at every magnitude, plus adjacent magnitude contrasts at every radius.
    pairs = [(a,b) for base in range(3) for a,b in [(base,base+3),(base+3,base+6)]] + [(base+a,base+b) for base in [0,3,6] for a,b in [(0,1),(1,2)]]
    for a,b in pairs:
        output['contrasts'][str(cuts[b])+' minus '+str(cuts[a])] = {'delta_weighted_m_pp':entry(mw[:,b]-mw[:,a]), 'delta_measured_Rdiag':entry(diag(rs[:,b])-diag(rs[:,a])), 'delta_model_Rdiag':entry(diag(rw[:,b])-diag(rw[:,a])), 'delta_response_error_diag':entry((diag(rs[:,b])-diag(rw[:,b]))-(diag(rs[:,a])-diag(rw[:,a])))}
    (H/'frozen_comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('CENTRAL', json.dumps(output['cuts']['mag25_radius_gt0.60']),flush=True)
    print('RMS',json.dumps(output['joint_nine_cut_rms_percent']),flush=True)
    print('RADIUS',json.dumps(output['contrasts']['mag25_radius_gt0.64 minus mag25_radius_gt0.60']),flush=True)
    print('DONE_FROZEN_COMPARISON',flush=True)
if __name__ == '__main__':
    main()
