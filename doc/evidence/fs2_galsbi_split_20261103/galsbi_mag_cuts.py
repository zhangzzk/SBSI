"""GalSBI-dev by-true-magnitude contributions for every saved cut (galsbi_dev_constant_20261103 sums), as compare_cuts.py."""
import numpy as np

R = '/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_20261103/sums/'
H = .02
lab = ('<24', '24-25.5', '25.5-27.5', '>=27.5')
M, S = [], []
for c in range(1000, 1140):
    z = np.load(f'{R}case{c}.npz')
    M.append(z['model'])
    S.append(z['sim'])
    cuts = ('uncut', 'main 25.8 r>0.6', '25.8 r>=0.75')
M, S = np.array(M), np.array(S)


def parts(s, m, flow=False):
    Ns, Nm = s[..., 0].sum(-1), m[..., 0].sum(-1)
    rs = (s[..., 0, :, 1]/Ns[..., 0, None]-s[..., 1, :, 1]/Ns[..., 1, None])/(2*H)
    em = m[..., 1]+(0 if flow else m[..., 3])
    rm = (em[..., 0, :]/Nm[..., 0, None]-em[..., 1, :]/Nm[..., 1, None])/(2*H)
    return 100*(rs-rm)/rm.sum(-1)[..., None], 100*(rs.sum(-1)/rm.sum(-1)-1), 100*(rs/rm-1), s[..., 0, :, 0]/Ns[..., 0, None]


W = np.random.default_rng(20260925).multinomial(140, np.full(140, 1/140), size=10000)
for k in range(3):
    for flow in (False, True):
        s, m = S[:, :, k], M[:, 0, :, k]
        c, t, wb, f = parts(s.sum(0), m.sum(0), flow)
        bc, bt, bw, _ = parts(np.einsum('bc,c...->b...', W, s), np.einsum('bc,c...->b...', W, m), flow)
        print(f"{cuts[k]}{' flow only' if flow else ''}: m {t:+.3f} +- {bt.std(ddof=1):.3f}")
        for l, x, e, w, we, fr in zip(lab, c, bc.std(0, ddof=1), wb, bw.std(0, ddof=1), f):
            print(f'   {l:10s} contribution {x:+6.2f} +- {e:.2f}  within {w:+7.2f} +- {we:.2f}  sel.frac {fr:.3f}')
