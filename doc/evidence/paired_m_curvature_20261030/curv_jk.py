"""Chain-length check of the curvature column: log-mean-exp of a finite chain is biased low by ~var/(2N), which makes
-(L+ + L-)/d^2 too large. Correct with the two chain halves (half-length bias is twice as big):
H_corrected = 2 H_full - (H_half1 + H_half2)/2. Same per-band, case-bootstrap layout as curvature.py."""
import sys

import numpy as np

D = .002
DLOGZ1, D2LOGZ1 = -0.00108144, 0.04986303   # g1 selection terms printed by curvature.py (identical for both legs)
z = np.load(sys.argv[1])
cases = np.arange(40, 140)
bands = (('<21', -9, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25), ('25-25.8', 25, 26), ('ALL', -9, 99))
two = lambda x: (x[:, 1]-x[:, 0])/(2*D)
curv = lambda x: -(x[:, 0]+x[:, 1])/D**2+D2LOGZ1
tab = {}
for leg in ('plus', 'minus'):
    L, h1, h2 = z[f'{leg}_L'], z[f'{leg}_h1'], z[f'{leg}_h2']
    s = two(L)-DLOGZ1
    e = two(h1)-two(h2)
    cols = (s, s*s-e*e/4, curv(L), (curv(h1)+curv(h2))/2)
    idx = np.searchsorted(cases, z[f'{leg}_case'])
    mag = z[f'{leg}_mag']
    t = np.zeros((len(bands), 100, len(cols)))
    for b, (_, lo, hi) in enumerate(bands):
        k = (mag >= lo) & (mag < hi)
        for j, x in enumerate(cols):
            np.add.at(t[b, :, j], idx[k], x[k])
    tab[leg] = t


def stats(k):
    p, m = tab['plus'][:, k].sum(1), tab['minus'][:, k].sum(1)
    S = (p[:, 0]-m[:, 0])/0.04
    I = (p[:, 1]+m[:, 1])/2
    H = (p[:, 2]+m[:, 2])/2
    Hh = (p[:, 3]+m[:, 3])/2
    Hc = 2*H-Hh
    return np.stack((S/I-1, S/H-1, S/Hc-1, Hc/I, H/Hh), 1)


rng = np.random.default_rng(20261030)
est = stats(np.arange(100))
err = np.std([stats(rng.integers(0, 100, 100)) for _ in range(2000)], 0)
print('band     | m one-step OPG   | m curvature raw  | m curvature chain-corrected | corrected H/I | H_full/H_half')
for b, (n, _, _) in enumerate(bands):
    print(f'{n:8s} | ' + ' | '.join(f'{100*est[b, j]:+6.2f}+-{100*err[b, j]:4.2f}%' for j in range(3))
          + f' | {est[b, 3]:.3f}+-{err[b, 3]:.3f} | {est[b, 4]:.3f}+-{err[b, 4]:.3f}')
