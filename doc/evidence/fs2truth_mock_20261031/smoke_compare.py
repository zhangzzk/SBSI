"""Smoke check: model mock from true ConstGold galaxies vs the ConstGold images, same cases 40-41, plus leg."""
import sys

import numpy as np
import pandas as pd

R = '/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/'
mock = pd.read_parquet(sys.argv[1]+'/plus/reference/observations.parquet')
mt = pd.read_parquet(sys.argv[1]+'/plus/reference/truth.parquet')
img = pd.read_parquet(R+'galsbi_constgold1m_signsym_20261028/recovery_v1/reference/observations.parquet')
it = pd.read_parquet(R+'galsbi_constgold1m_signsym_20261028/recovery_v1/reference/truth.parquet')
img = img[it.source_case.isin([40, 41]).to_numpy()]
print(mock.dtypes.to_dict())
for name, d in (('mock (true galaxies + model)', mock), ('images (1M-run sample)', img)):
    mag = 30-2.5*np.log10(d.measured_flux_from_mag_auto.to_numpy())
    r = d.measured_flux_radius.to_numpy()
    e = np.hypot(d.measured_ngmix_g1, d.measured_ngmix_g2).to_numpy()
    print(f'\n{name}: n={len(d)}')
    for lo, hi in ((-9, 21), (21, 23), (23, 24), (24, 25), (25, 25.8)):
        k = (mag >= lo) & (mag < hi)
        print(f'  mag {lo:5.1f}-{hi:4.1f}: frac {k.mean():.3f}  median radius {np.median(r[k]):.2f}  mean|e| {e[k].mean():.3f}'
              f'  frac|e|>0.7 {np.mean(e[k] > .7):.3f}  mean g1 {d.measured_ngmix_g1.to_numpy()[k].mean():+.4f}')
