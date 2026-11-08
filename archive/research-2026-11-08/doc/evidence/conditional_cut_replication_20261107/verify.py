"""Replay saved small-sum reductions before any replication evaluation."""
import importlib.util
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('cutrep_reducer_test', H/'report.py')
    r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    cfg = r.read(H/'manifest.json')
    old = r.read(H.parent/'conditional_cut_probability_20261107/report.json')
    cases = cfg['pilot_cases']; n = len(cases)
    w = np.vstack([np.ones(n), np.ones((n,n))-np.eye(n)])
    checked = 0; worst = 0.
    for axis in [1,2]:
        for arm in cfg['arms']:
            sims, models = [], []
            for case in cases:
                path = Path(cfg['original_root'])/'constant'/arm/f'axis{axis}'/f'case{case}.npz'
                receipt = r.read(path.with_suffix('.json'))
                assert r.sha(path) == receipt['data_sha256']
                assert receipt['checkpoint_sha256'] == cfg['checkpoints'][arm]['sha256']
                with np.load(path) as z: sims.append(z['sim'].sum(-2)); models.append(z['model'].sum(-2))
            sim, model = np.array(sims), np.array(models)
            for mode, name in enumerate(r.MODES):
                for ci, cut in enumerate(cfg['cuts']):
                    got = r.estimates(sim[:,:,ci], model[:,mode,:,ci], w, axis)
                    saved = old['axes'][str(axis)]['arms'][arm][name][cut[0]]
                    for key, values in got.items():
                        se = np.sqrt((n-1)/n*((values[1:]-values[1:].mean(0))**2).sum(0))
                        for a,b in [(values[0],saved[key]['value']),(se,saved[key]['jackknife_se'])]:
                            worst=max(worst,float(np.abs(np.asarray(a)-b).max()))
                            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)
                        checked+=1
    # Weighted ratios must be formed after pooling; unequal counts make this test sensitive.
    sim=np.array([[[10, .2, 0],[20,-.4,0]],[[100,4.,0],[40,-1.6,0]]],dtype=float)
    model=np.concatenate([sim,np.zeros((2,2,2))],-1)
    got=r.estimates(sim,model,np.ones((1,2)),1)
    expected=(4.2/110-(-2./60))/.04
    np.testing.assert_allclose(got['simulation_response'][0,0],expected,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(got['m_percent'],[0.])
    assert abs(expected-np.mean([1.,2.]))>.1
    tasks=[cfg['first_new_task']]+sum(cfg['worker_tasks'],[])
    assert sorted(tasks)==list(range(len(cfg['tasks']))) and len(tasks)==72
    assert set(cfg['pilot_cases']).isdisjoint(cfg['replication_cases'])
    assert set(cfg['cases'])==set(cfg['pilot_cases'])|set(cfg['replication_cases'])
    receipt=dict(saved_statistics_checked=checked,maximum_difference=worst,
                 pooled_ratio_fixture=True,all_72_tasks_covered_once=True,
                 cohorts_disjoint=True)
    (H/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('REPLICATION_REDUCER_CHECKS',receipt,flush=True)


if __name__=='__main__':main()
