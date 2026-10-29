"""Summarise allcases.npz: mean measured-minus-intrinsic e2 (and e1) per band, leg and case set; errors from a
jackknife over cases (galaxies in one image are not independent)."""
import sys

import numpy as np

z = np.load(sys.argv[1])
case, leg, S = z['case'], z['leg'], z['sums']   # S: (catalogue, band, [n, d1, d2, d1^2, d2^2, i1, i2, m1, m2])
BANDS = ('<21', '21-23', '23-24', '24-25', '25-25.8', 'ALL')
S = np.concatenate((S, S.sum(1, keepdims=True)), 1)


def ratio(sel, col, b):
    """sum col / sum n over selected catalogues, with a delete-one-case jackknife error (both legs of a case
    share galaxies and noise, so a case is removed as a whole)."""
    s, c = S[sel, b], case[sel]
    tot = s[:, col].sum()/s[:, 0].sum()
    u = np.unique(c)
    jk = np.array([s[c != k, col].sum()/s[c != k, 0].sum() for k in u])
    return tot, np.sqrt((len(u)-1)/len(u)*((jk-jk.mean())**2).sum())


sets = {'cases 40-139': (case >= 40), 'cases 0-39': (case < 40), 'all 140 cases': case >= 0}
print('mean (measured - intrinsic) e2 [e-4], jackknife over cases; n = galaxies')
for b, name in enumerate(BANDS):
    row = []
    for sname, sm in sets.items():
        for l in (0.02, -0.02):
            sel = sm & (leg == l)
            m, e = ratio(sel, 2, b)
            row.append(f'{sname} {l:+.2f}: {m*1e4:+6.2f}+-{e*1e4:4.2f} (n={int(S[sel, b, 0].sum())})')
    sel = leg != 0
    m, e = ratio(sel, 2, b)
    print(f'{name:8s} | ' + ' | '.join(row) + f' || BOTH LEGS ALL: {m*1e4:+6.2f}+-{e*1e4:4.2f}')

print('\nmean intrinsic e2 [e-4] (shape-noise draw) and mean measured e2, all 140 cases')
for b, name in enumerate(BANDS):
    out = []
    for l in (0.02, -0.02):
        sel = leg == l
        mi, ei = ratio(sel, 6, b)
        mm, em = ratio(sel, 8, b)
        out.append(f'{l:+.2f}: int {mi*1e4:+6.2f}+-{ei*1e4:4.2f} meas {mm*1e4:+6.2f}+-{em*1e4:4.2f}')
    print(f'{name:8s} ' + ' | '.join(out))

print('\nmodel-free e1 [e-4]: (plus - minus)/2 measured-minus-intrinsic = shear response x 0.02; (plus + minus)/2 = additive c1-like')
for b, name in enumerate(BANDS):
    p, ep = ratio(leg == 0.02, 1, b)
    q, eq = ratio(leg == -0.02, 1, b)
    print(f'{name:8s} plus {p*1e4:+7.2f}+-{ep*1e4:4.2f} minus {q*1e4:+7.2f}+-{eq*1e4:4.2f} -> R = {(p-q)/2/0.02:.3f}, '
          f'additive {(p+q)/2*1e4:+6.2f}+-{np.hypot(ep, eq)/2*1e4:4.2f}')
