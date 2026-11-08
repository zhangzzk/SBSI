"""Read-only replay of saved summaries for the two-run review; run on Slurm."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('run summary-array audit through scheduler')

B = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
M = B/'galsbi_pair1m_20261012'
R = B/'galsbi_constgold1m_signsym_20261028/recovery_v1'
S = B/'galsbi_faintg05_20261013/snapshot_d776abb'
mock = json.loads((M/'debug/step2_result_brightw8_fl32.json').read_text())
measured = json.loads((R/'c2/estimate.json').read_text())
report = dict(job=os.environ['SLURM_JOB_ID'], scope='Saved-summary replay and source comparison; no new sampling')

two = lambda x: np.column_stack((x[:, 1]-x[:, 0], x[:, 3]-x[:, 2]))/.004
parts, selection = [], []
for side in ['original', 'twin']:
    with np.load(M/f'debug/pergal_{side}_fl32.npz') as z:
        obs = z['obs']
        ref = pd.read_parquet(M/side/'reference/observations.parquet',
            columns=['measured_flux_radius', 'measured_ngmix_g1', 'measured_ngmix_g2'])
        keep = (ref.measured_flux_radius.to_numpy()[obs] < 100)
        keep &= np.hypot(ref.measured_ngmix_g1, ref.measured_ngmix_g2).to_numpy()[obs] < .99
        parts.append((two(z['L'][keep]), two(z['h1'][keep])-two(z['h2'][keep])))
        selection.append(z['z'])
        if side == 'original':
            line = next(l.split() for l in (M/'debug/faintlong_units.txt').read_text().splitlines()
                        if l.split()[1] == 'L_original_u0394')
            excluded = np.array(list(map(int, line[2:])))
            indices = {int(o): i for i, o in enumerate(obs)}
            k = np.array([indices[int(o)] for o in excluded])
            with np.load(M/'debug/pergal_original_faintdraw.npz') as old:
                np.testing.assert_array_equal(old['obs'], obs)
                for key in ['L', 'h1', 'h2', 'ndraw']:
                    np.testing.assert_array_equal(z[key][k], old[key][k])
            report['mock_failed_long_unit'] = dict(label=line[1], observations=len(k),
                retained_after_cuts=int(keep[k].sum()), policy='earlier faintdraw values retained exactly')

z = np.concatenate(selection)
logz = np.log(z.mean(0))
grad = (logz[[2, 4]]-logz[[1, 3]])/.004
score = np.concatenate([p[0] for p in parts])-grad
delta = np.concatenate([p[1] for p in parts])
info = score.T@score-delta.T@delta/4
g = np.linalg.solve(info, score.sum(0))
target = mock['paired']
np.testing.assert_allclose(g, target['one_step']['g'], rtol=0, atol=1e-12)
np.testing.assert_allclose(info, target['information'], rtol=1e-12, atol=1e-6)
gfinal = g+np.linalg.solve(info, np.array(target['S_B_g1'])
                         +np.array(target['I_B'])@mock['g1']-target['S_B0'])
np.testing.assert_allclose(gfinal, target['step2']['g'], rtol=0, atol=1e-12)
report['mock_replay'] = dict(galaxies=len(score), selection_chunks=len(z),
    one_step=g.tolist(), final_from_saved_bright_sums=gfinal.tolist(),
    max_final_error=float(np.max(np.abs(gfinal-target['step2']['g']))),
    limitation='Replays saved per-galaxy summaries and saved bright sums; does not rerun chains or bootstrap')
del parts, score, delta

with np.load(R/'c2/pergal.npz') as z:
    obs = z['obs']
    assert len(np.unique(obs)) == len(obs) == measured['galaxies']
    assert np.array_equal(np.sort(obs), np.arange(len(obs)))
    info = z['Ii'].sum(0)
    g = np.linalg.solve(info, z['v'].sum(0))
    np.testing.assert_allclose(g, measured['g'], rtol=0, atol=1e-12)
    inv = np.linalg.inv(info)
    err = np.sqrt(np.diag(inv+inv@(z['e'].T@z['e']/4)@inv+inv@z['sel']@inv))
    np.testing.assert_allclose(err, measured['errors']['total'], rtol=0, atol=1e-12)
    report['measurement_replay'] = dict(galaxies=len(obs), all_rows_exactly_once=True,
        g=g.tolist(), total_errors=err.tolist(), max_g_error=float(np.max(np.abs(g-measured['g']))))

same = {}
for rel in ['sbsi/galsbi_prior.py', 'sbsi/galsbi_scene_prior.py', 'sbsi/galsbi_target_coordinates.py',
            'sbsi/mcmc_proposals.py', 'sbsi/pseudo_marginal.py', 'sbsi/catalogue_disk_likelihood.py',
            'sbsi/catalogue_likelihood.py', 'sbsi/catalogue_disk_response.py']:
    a, b = (S/rel).read_bytes(), (R/'source_snapshot'/rel).read_bytes()
    assert a == b
    same[rel] = hashlib.sha256(a).hexdigest()
report['identical_core_files'] = same
files = [M/'debug/step2.py', M/'debug/step2_result_brightw8_fl32.json', M/'debug/rule_one.md',
         M/'debug/one/b_result.txt', M/'debug/fl32.py', M/'debug/faintlong.py',
         R/'estimator_base.py', R/'c1/estimate.json', R/'c2/estimate.json', R/'rule.json',
         R/'source_snapshot/scripts/check_galsbi_local_sampler.py']
report['evidence_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
report['status'] = 'passed'
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
