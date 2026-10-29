import numpy as np
z = np.load('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/recovery_v1/c2/pergal.npz')
mag, Ii = z['mag'], z['Ii']
I22 = Ii[:, 1, 1].sum()
edges = [(-9, 21), (21, 23), (23, 24), (24, 25), (25, 26)]
add = np.array([2.25, -0.76, 3.12, 3.15, 6.47]); sadd = np.array([1.56, 1.49, 1.52, 1.22, 1.34])   # e-4, both legs, all cases
R = np.array([1.050, 1.108, 0.914, 0.636, 0.465])
w = np.array([Ii[(mag >= a) & (mag < b), 1, 1].sum()/I22 for a, b in edges])
c = w*add/R; sc = w*sadd/R
print('band weights', w.round(3)); print('band contributions e-4', c.round(2))
# band errors are correlated through shared cases; quote the independent-sum and the fully-correlated bound
print(f'image-side additive c2 in inference weighting: {c.sum():+.2f} e-4 +- {np.sqrt((sc**2).sum()):.2f} (indep) / {sc.sum():.2f} (fully corr)')
