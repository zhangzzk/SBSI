"""Where does the paired m deficit sit in TRUE galaxy properties? Galaxies are matched across the +/-0.02 legs by
(case, input index); each matched galaxy gives its own score response (s+ - s-)/0.04 and the production score-scatter
information (I+ + I-)/2. Per bin of a true property: m = sum response / sum information - 1, case-bootstrap errors.
Truth from the ConstGold input catalogues (diagnosis only; binning by truth is identical in both legs).
The prior is GalSBI F24 while ConstGold is Flagship-2: if the prior is the cause, m should track properties where
the two populations differ (size, Sersic, neighbours)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

D = .002
DLOGZ1 = -0.00108144
SIMS = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant')


def truth(case):
    t = pd.read_feather(SIMS/f'case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather',
                        columns=['index_input', 'RA_input', 'DEC_input', 'Re_input', 'axis_ratio_input',
                                 'sersic_n_input', 'r_input', 'e1_input_rot0', 'e2_input_rot0'])
    dec0 = np.deg2rad(t.DEC_input.mean())
    xy = np.column_stack((t.RA_input*np.cos(dec0), t.DEC_input))*3600.
    tree = cKDTree(xy)
    bright = t.r_input.to_numpy() < 25.8
    tb = cKDTree(xy[bright])
    n_all = np.array([len(x) for x in tree.query_ball_point(xy, 5.)])-1
    n_br = np.array([len(x) for x in tb.query_ball_point(xy, 5.)])-bright
    return pd.DataFrame(dict(case=case, ident=t.index_input.to_numpy(), r=t.r_input.to_numpy(),
                             size=(t.Re_input*np.sqrt(t.axis_ratio_input)).to_numpy(), n=t.sersic_n_input.to_numpy(),
                             e=np.hypot(t.e1_input_rot0, t.e2_input_rot0).to_numpy(), nb5_all=n_all, nb5_br=n_br))


def legframe(z, leg):
    L, h1, h2 = z[f'{leg}_L'], z[f'{leg}_h1'], z[f'{leg}_h2']
    two = lambda x: (x[:, 1]-x[:, 0])/(2*D)
    s = two(L)-DLOGZ1
    e = two(h1)-two(h2)
    return pd.DataFrame({'case': z[f'{leg}_case'], 'ident': z[f'{leg}_ident'], f's_{leg}': s,
                         f'I_{leg}': s*s-e*e/4, f'mag_{leg}': z[f'{leg}_mag']})


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    p, m = legframe(z, 'plus'), legframe(z, 'minus')
    for d in (p, m):
        assert not d.duplicated(['case', 'ident']).any()
    g = p.merge(m, on=['case', 'ident'], how='inner', validate='one_to_one')
    print(f'matched {len(g)}; dropped plus-only {len(p)-len(g)}, minus-only {len(m)-len(g)}')
    with Pool(int(sys.argv[2])) as pool:
        tr = pd.concat(pool.map(truth, sorted(g.case.unique())))
    g = g.merge(tr, on=['case', 'ident'], how='inner', validate='one_to_one')
    print(f'with truth {len(g)}')
    g['resp'] = (g.s_plus-g.s_minus)/0.04
    g['inf'] = (g.I_plus+g.I_minus)/2
    g['mag'] = (g.mag_plus+g.mag_minus)/2
    cases = np.arange(40, 140)
    rng = np.random.default_rng(20261031)
    boot = [rng.integers(0, 100, 100) for _ in range(1000)]

    def report(sel, key, edges):
        x = g[sel]
        idx = np.searchsorted(cases, x.case.to_numpy())
        b = np.digitize(x[key].to_numpy(), edges)-1
        ok = (b >= 0) & (b < len(edges)-1)
        T = np.zeros((len(edges)-1, 100, 2))
        np.add.at(T, (b[ok], idx[ok], 0), x.resp.to_numpy()[ok])
        np.add.at(T, (b[ok], idx[ok], 1), x.inf.to_numpy()[ok])
        f = lambda k: T[:, k, 0].sum(1)/T[:, k, 1].sum(1)-1
        est, err = f(np.arange(100)), np.std([f(k) for k in boot], 0)
        cnt = np.bincount(b[ok], minlength=len(edges)-1)
        for j in range(len(edges)-1):
            print(f'  {key:8s} [{edges[j]:7.3f},{edges[j+1]:7.3f}) n={cnt[j]:7d}  m = {100*est[j]:+6.2f} +- {100*err[j]:4.2f}%')

    for lab, sel in (('measured mag 21-25', (g.mag >= 21) & (g.mag < 25)), ('measured mag 25-25.8', g.mag >= 25),
                     ('all matched', g.mag > -99)):
        x = g[sel]
        print(f'\n== {lab}: {len(x)} galaxies')
        for key in ('size', 'r', 'n', 'e', 'nb5_all', 'nb5_br'):
            q = np.unique(np.quantile(x[key], np.linspace(0, 1, 6)))
            q[-1] = np.nextafter(q[-1], np.inf)
            report(sel, key, q)
