"""Does the GalSBI prior + model predict the measured galaxies we see in ConstGold? Compare measured observables of
the 1M model mock (GalSBI prior through V3.9 model, g=(0.02,0)) with the ConstGold plus leg (same g, images), per
measured-magnitude band. Both samples carry the same measured cuts. Errors: bootstrap over galaxies for the mock and
over cases for ConstGold (100 cases)."""
import numpy as np
import pandas as pd

R = '/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/'
mock = pd.read_parquet(R+'galsbi_pair1m_20261012/original/reference/observations.parquet')
img = pd.read_parquet(R+'galsbi_constgold1m_signsym_20261028/recovery_v1/reference/observations.parquet')
case = pd.read_parquet(R+'galsbi_constgold1m_signsym_20261028/recovery_v1/reference/truth.parquet').source_case.to_numpy()
print('mock columns', list(mock.columns))
EDGES = [-9, 21, 23, 24, 25, 25.8]
NAMES = ['<21', '21-23', '23-24', '24-25', '25-25.8']


def summ(d):
    mag = 30-2.5*np.log10(d['measured_flux_from_mag_auto'].to_numpy())
    r = d['measured_flux_radius'].to_numpy()
    g1, g2 = d['measured_ngmix_g1'].to_numpy(), d['measured_ngmix_g2'].to_numpy()
    b = np.digitize(mag, EDGES)-1
    out = []
    for k in range(5):
        m = b == k
        e = np.hypot(g1[m], g2[m])
        out.append([m.mean(), np.median(r[m]), np.quantile(r[m], .16), np.quantile(r[m], .84),
                    g2[m].std(), np.mean(e), np.mean(e > .7), np.mean(g1[m])])
    return np.array(out)


rng = np.random.default_rng(1)
sm = summ(mock)
em = np.std([summ(mock.iloc[rng.integers(0, len(mock), len(mock))]) for _ in range(20)], 0)
si = summ(img)
uc = np.unique(case)
grp = {c: np.flatnonzero(case == c) for c in uc}
ei = np.std([summ(img.iloc[np.concatenate([grp[c] for c in rng.choice(uc, len(uc))])]) for _ in range(20)], 0)
cols = ['fraction', 'median radius px', 'radius 16%', 'radius 84%', 'std g2', 'mean |e|', 'frac |e|>0.7', 'mean g1']
for j, c in enumerate(cols):
    print(f'\n{c}:  band | mock (GalSBI prior + model) | ConstGold images | images - mock')
    for k in range(5):
        print(f'  {NAMES[k]:8s} | {sm[k, j]:9.4f} +- {em[k, j]:.4f} | {si[k, j]:9.4f} +- {ei[k, j]:.4f} | '
              f'{si[k, j]-sm[k, j]:+9.4f} ({(si[k, j]-sm[k, j])/np.hypot(em[k, j], ei[k, j]):+.1f} sigma)')
