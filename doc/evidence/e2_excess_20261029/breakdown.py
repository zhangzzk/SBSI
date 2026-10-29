"""Read-only breakdown of the measured 1M c2 (symmetric F8): bands, selection part, R-hat split, cases,
and a model-free mean-e2 check of the image data's mirror symmetry."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/recovery_v1')
est = json.loads((H/'c2/estimate.json').read_text())
z = np.load(H/'c2/pergal.npz')
obs, case, mag, v, Ii = z['obs'], z['case'], z['mag'], z['v'], z['Ii']
dlogz = np.array(est['selection_score_per_galaxy'])
I = Ii.sum(0)
inv = np.linalg.inv(I)
g = inv@v.sum(0)
print('g check', g, est['g'])
n = len(obs)

# R-hat of log Z per galaxy: C1 for all, C2 for mag<21
rh = {}
for stage in ('c1', 'c2'):
    for line in (H/stage/'units.txt').read_text().splitlines():
        lab = line.split()[0]
        for r in json.loads((H/stage/lab/'result.json').read_text())['observations']:
            rh[(stage, r['reference_observation'])] = r['diagnostics']['log_z']['rhat'] or 0
rhat = np.array([rh[('c2', o)] if m < 21 else rh[('c1', o)] for o, m in zip(obs, mag)])

ref = pd.read_parquet(H/'reference/observations.parquet').iloc[obs]
e1, e2 = ref['measured_ngmix_g1'].to_numpy(), ref['measured_ngmix_g2'].to_numpy()

def part(k):
    """g2 contribution of subset k to the final g (I^-1 of the full sample), split into data and selection."""
    vs = v[k].sum(0)
    sel = -k.sum()*dlogz
    return (inv@vs)[1], (inv@(vs-sel))[1], (inv@sel)[1], np.sqrt((inv@Ii[k].sum(0)@inv)[1, 1])

print(f'\nN={n}; selection score per galaxy dlogZ/dg = {dlogz}')
print('contribution to final g2 (sums to total): all | likelihood part | selection part | stat sd')
bands = (('<21', -9, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25), ('25-25.8', 25, 26))
for name, lo, hi in bands + (('ALL', -9, 99),):
    k = (mag >= lo) & (mag < hi)
    a, b, c, s = part(k)
    # model-free: mean measured e, response from <e1>/0.02 (assumes c1=0), naive g2 = <e2>/R
    m1, m2 = e1[k].mean(), e2[k].mean()
    s2 = e2[k].std()/np.sqrt(k.sum())
    R = m1/0.02
    own = np.linalg.solve(Ii[k].sum(0), v[k].sum(0))
    print(f'{name:8s} n={k.sum():7d}  c2 part {a*1e4:+6.2f} (lik {b*1e4:+6.2f}, sel {c*1e4:+6.2f}) +- {s*1e4:5.2f} e-4 | '
          f'band-alone g2 {own[1]*1e4:+6.2f}e-4 | <e1> {m1:+.5f} <e2> {m2*1e4:+6.2f} +- {s2*1e4:4.2f} e-4, '
          f'R~{R:.3f}, <e2>/R {m2/R*1e4:+6.2f} +- {s2/R*1e4:5.2f} e-4')

print('\nR-hat(logZ) split:')
for name, k in (('rhat<1.1', rhat < 1.1), ('rhat>=1.1', rhat >= 1.1)):
    a, b, c, s = part(k)
    own = np.linalg.solve(Ii[k].sum(0), v[k].sum(0))
    print(f'  {name:10s} n={k.sum():7d} c2 part {a*1e4:+6.2f} +- {s*1e4:5.2f} e-4; alone g = {own}')
for name, lo, hi in bands:
    for lab, kk in (('ok', rhat < 1.1), ('stuck', rhat >= 1.1)):
        k = kk & (mag >= lo) & (mag < hi)
        if k.sum() > 100:
            own = np.linalg.solve(Ii[k].sum(0), v[k].sum(0))
            print(f'  {name:8s} {lab:5s} n={k.sum():7d} alone g2 {own[1]*1e4:+7.2f} +- '
                  f'{np.sqrt(np.linalg.inv(Ii[k].sum(0))[1,1])*1e4:6.2f} e-4')

# per source case scatter
u = np.unique(case)
gc = np.array([np.linalg.solve(Ii[case == c].sum(0), v[case == c].sum(0)) for c in u])
sc = np.array([np.sqrt(np.diag(np.linalg.inv(Ii[case == c].sum(0)))) for c in u])
chi = ((gc-g)/sc)**2
print(f'\ncases {len(u)}: chi2/dof g1 {chi[:,0].sum()/(len(u)-1):.2f}, g2 {chi[:,1].sum()/(len(u)-1):.2f}; '
      f'largest |pull| g2 {np.abs((gc[:,1]-g[1])/sc[:,1]).max():.2f}')
# measured e2 vs score correlation: how much of the g2 score is the mean-e2 signal
print(f'\nmeasured e2 mean all {e2.mean()*1e4:+.2f} +- {e2.std()/np.sqrt(n)*1e4:.2f} e-4; e1 mean {e1.mean():+.5f}')
print('corr(v_g2, e2) =', np.corrcoef(v[:, 1], e2)[0, 1])
