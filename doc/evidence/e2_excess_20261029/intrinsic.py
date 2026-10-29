"""Read-only: split the measured 1M e2 excess and the inferred c2 into the galaxies' intrinsic-shape draw and the rest.
Intrinsic shapes are e1/e2_input_rot0 of the ConstGold input catalogues (truth used for diagnosis only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/recovery_v1')
S = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant')
est = json.loads((H/'c2/estimate.json').read_text())
z = np.load(H/'c2/pergal.npz')
obs, mag, v, Ii = z['obs'], z['mag'], z['v'], z['Ii']
I = Ii.sum(0)
inv = np.linalg.inv(I)
g = inv@v.sum(0)
ref = pd.read_parquet(H/'reference/observations.parquet').iloc[obs].reset_index(drop=True)
tru = pd.read_parquet(H/'reference/truth.parquet').iloc[obs].reset_index(drop=True)
parts = []
for c in np.unique(tru.source_case):
    t = pd.read_feather(S/f'case{c}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather',
                        columns=['index_input', 'e1_input_rot0', 'e2_input_rot0', 'axis_ratio_input', 'position_angle_input'])
    t['source_case'] = c
    parts.append(t)
t = pd.concat(parts)
m = tru[['source_case', 'source_input_index']].merge(t, left_on=['source_case', 'source_input_index'],
                                                     right_on=['source_case', 'index_input'], how='left', validate='one_to_one')
assert len(m) == len(obs) and m.e1_input_rot0.notna().all(), 'unmatched rows'
ei = m[['e1_input_rot0', 'e2_input_rot0']].to_numpy()
em = ref[['measured_ngmix_g1', 'measured_ngmix_g2']].to_numpy()
print('intrinsic |e| quantiles', np.quantile(np.hypot(*ei.T), [.1, .5, .9]))
print('per-galaxy corr measured vs intrinsic: e1', np.corrcoef(em[:, 0], ei[:, 0])[0, 1], 'e2', np.corrcoef(em[:, 1], ei[:, 1])[0, 1])
print('bright (<22) corr: e1', np.corrcoef(em[mag < 22, 0], ei[mag < 22, 0])[0, 1], 'e2', np.corrcoef(em[mag < 22, 1], ei[mag < 22, 1])[0, 1])

bands = (('<21', -9, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25), ('25-25.8', 25, 26), ('ALL', -9, 99))
print('\nper band (e-4): mean intrinsic e2 | mean measured e2 | measured-intrinsic e2 (data-side additive, R-free) ; same for e1')
for name, lo, hi in bands:
    k = (mag >= lo) & (mag < hi)
    n = k.sum()
    d = em[k]-ei[k]
    f = lambda x: f'{x.mean()*1e4:+7.2f} +- {x.std()/np.sqrt(n)*1e4:5.2f}'
    print(f'{name:8s} n={n:7d} e2: int {f(ei[k,1])} | meas {f(em[k,1])} | diff {f(d[:,1])} || '
          f'e1: int {f(ei[k,0])} | meas {f(em[k,0])} | diff {f(d[:,0])}')

# how much of the inferred g2 follows the intrinsic draw: regress per-galaxy score on intrinsic shape per band
print('\ninferred g2 split (e-4): total | follows intrinsic e2 draw | remainder +- remainder stat')
tot_int, tot_rem, var_rem = 0., 0., 0.
for name, lo, hi in bands[:-1]:
    k = (mag >= lo) & (mag < hi)
    X = np.column_stack((np.ones(k.sum()), ei[k, 0], ei[k, 1]))
    beta, *_ = np.linalg.lstsq(X, v[k, 1], rcond=None)
    fol = beta[2]*ei[k, 1].sum()
    res = v[k, 1]-X@beta
    rem = v[k, 1].sum()-fol
    tot_int += fol; tot_rem += rem; var_rem += (res**2).sum()
    print(f'{name:8s} slope dS/de2_int {beta[2]:9.1f}: total {v[k,1].sum()*inv[1,1]*1e4:+6.2f} | intrinsic {fol*inv[1,1]*1e4:+6.2f} | '
          f'remainder {rem*inv[1,1]*1e4:+6.2f} +- {np.sqrt((res**2).sum())*inv[1,1]*1e4:5.2f}')
print(f'ALL      total {g[1]*1e4:+6.2f} | intrinsic {tot_int*inv[1,1]*1e4:+6.2f} | remainder {tot_rem*inv[1,1]*1e4:+6.2f} '
      f'+- {np.sqrt(var_rem)*inv[1,1]*1e4:5.2f}  (g1-g2 cross term in I ignored: I12/I22 = {I[0,1]/I[1,1]:.4f})')
# g1 counterpart, for m
X = np.column_stack((np.ones(len(v)), ei))
b1, *_ = np.linalg.lstsq(X, v[:, 0], rcond=None)
print(f'g1: intrinsic-e1 draw contributes {b1[1]*ei[:,0].sum()*inv[0,0]*1e4:+.2f}e-4 to g1 '
      f'(= {100*b1[1]*ei[:,0].sum()*inv[0,0]/0.02:+.2f} percentage points of m)')
