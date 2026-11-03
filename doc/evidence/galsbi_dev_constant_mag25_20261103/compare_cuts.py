import numpy as np
R = '/project/ls-gruen/users/zekang.zhang/sbsi_caches/'
H = 0.02
lab = ('<24', '24-25.5', '25.5-27.5', '>=27.5')


def load(study):
    M, S = [], []
    for c in range(1000, 1140):
        z = np.load(f'{R}{study}/sums/case{c}.npz')
        M.append(z['model']); S.append(z['sim'])
    return np.array(M), np.array(S)


def parts(s, m):   # s (..., leg, bin, 3), m (..., leg, bin, 5): per-bin contributions to m and total m
    Ns, Nm = s[..., 0].sum(-1), m[..., 0].sum(-1)
    rs = (s[..., 0, :, 1]/Ns[..., 0, None]-s[..., 1, :, 1]/Ns[..., 1, None])/(2*H)
    em = m[..., 1]+m[..., 3]
    rm = (em[..., 0, :]/Nm[..., 0, None]-em[..., 1, :]/Nm[..., 1, None])/(2*H)
    total = 100*(rs.sum(-1)/rm.sum(-1)-1)
    return 100*(rs-rm)/rm.sum(-1)[..., None], total, s[..., 0, :, 0]/Ns[..., 0, None]


W = np.random.default_rng(20260925).multinomial(140, np.full(140, 1/140), size=10000)
out = {}
for study in ('galsbi_dev_constant_20261103', 'galsbi_dev_constant_mag25_20261103'):
    M, S = load(study)
    for w, name in ((0, 'C8'), (1, 'true flags')):
        s, m = S[:, :, 1], M[:, w, :, 1]
        c, t, f = parts(s.sum(0), m.sum(0))
        bc, bt, _ = parts(np.einsum('bc,c...->b...', W, s), np.einsum('bc,c...->b...', W, m))
        out[study, name] = bt
        print(f'{study} {name}: m {t:+.3f} +- {bt.std(ddof=1):.3f}; contributions',
              ' '.join(f'{l}: {x:+.2f}+-{e:.2f}' for l, x, e in zip(lab, c, bc.std(0, ddof=1))),
              ' sel. fraction', np.round(f, 3))
for name in ('C8', 'true flags'):
    d = out['galsbi_dev_constant_mag25_20261103', name]-out['galsbi_dev_constant_20261103', name]
    print(f'paired change 25.0 - 25.8 ({name}): {d.mean():+.3f} +- {d.std(ddof=1):.3f} pp')
