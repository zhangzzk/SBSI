"""Model-free: mean measured-minus-intrinsic e2/e1 per measured-magnitude band in the g=0 training catalogue,
same measured cuts as the 1M run; delete-one-case jackknife."""
import numpy as np, pyarrow.feather as pf
p = '/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.0_train.feather'
cols = ['detected', 'measured_ngmix_g1', 'measured_ngmix_g2', 'measured_mag_auto', 'measured_flux_radius',
        'e1_input_rot0_p', 'e2_input_rot0_p', 'gamma1_input_p', 'gamma2_input_p', 'case']
t = pf.read_table(p, columns=cols, memory_map=True).to_pandas()
print('rows', len(t), 'shear values', np.unique(t[['gamma1_input_p', 'gamma2_input_p']].to_numpy().round(6), axis=0)[:5])
v = t[['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_mag_auto', 'measured_flux_radius']].to_numpy(float)
ok = t.detected.to_numpy(bool) & np.isfinite(v).all(1) & (v[:, 3] > 3) & (v[:, 3] < 100) & (v[:, 2] < 25.8) \
    & (v[:, 0] != -1) & ~((v[:, 0] == 0) & (v[:, 1] == 0)) & (np.hypot(v[:, 0], v[:, 1]) < .99)
t, v = t[ok], v[ok]
d = v[:, :2]-t[['e1_input_rot0_p', 'e2_input_rot0_p']].to_numpy()
ei = t[['e1_input_rot0_p', 'e2_input_rot0_p']].to_numpy()
case = t.case.to_numpy(); u, ci = np.unique(case, return_inverse=True)
print('selected', len(t), 'cases', len(u))
for name, lo, hi in (('<21', -9, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25), ('25-25.8', 25, 26), ('ALL', -9, 99)):
    k = (v[:, 2] >= lo) & (v[:, 2] < hi)
    out = []
    for lab, x in (('d e2', d[:, 1]), ('d e1', d[:, 0]), ('int e2', ei[:, 1])):
        n = np.bincount(ci[k], minlength=len(u)); s = np.bincount(ci[k], x[k], minlength=len(u))
        jk = (s.sum()-s)/(n.sum()-n)
        out.append(f'{lab} {s.sum()/n.sum()*1e4:+6.2f}+-{np.sqrt((len(u)-1)/len(u)*((jk-jk.mean())**2).sum())*1e4:4.2f}')
    print(f'{name:8s} n={k.sum():9d} ' + ' | '.join(out))
