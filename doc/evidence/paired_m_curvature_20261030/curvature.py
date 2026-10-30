"""Read-only review of the paired 1M measured run: does the one-step estimator's score-scatter (OPG) information
match the likelihood's actual curvature and the true slope between the +0.02 and -0.02 legs?

Per galaxy, from the C1 readings at g0 = 0 (L_i(g0 +- d e_a), L_i(g0) = 0 by construction):
  score     s_ia = [L(+d) - L(-d)]/(2d) - dlogZ_a          (as the production estimator)
  curvature h_ia = -[L(+d) + L(-d)]/d^2 + d2logZ_a          (model Hessian of the selected log-likelihood, diagonal)
Per magnitude band b and leg:
  I_b = sum (s s - E)_11 (production OPG information), H_b = sum h_1,
  S_b = (sum s+_1 - sum s-_1)/0.04 (true response of the score to the injected shear, from the legs).
Paired m with the C1 one-step: S/I - 1 (production), with the model curvature: S/H - 1.
"""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
LEGS = {'plus': ROOT/'galsbi_constgold1m_signsym_20261028/recovery_v1',
        'minus': ROOT/'galsbi_constgold1m_minus_signsym_20261029'}
D = .002
NAMES = ['g1_-0.002', 'g1_0.002', 'g2_-0.002', 'g2_0.002']
lm = lambda x: logsumexp(x)-np.log(x.size)
OUT = Path(sys.argv[1])


def unit(args):
    stage_dir, label = args
    run = stage_dir/label
    obs = json.loads((run/'protocol.json').read_text())['observations']
    out = {}
    with np.load(run/'local_profiles.npz') as z:
        for o in obs:
            r = [z[f'observation_{o}_{n}'] for n in NAMES]
            out[o] = ([lm(x) for x in r], [lm(x[:, :2]) for x in r], [lm(x[:, 2:]) for x in r])
    return out


def load(leg, pool):
    H = LEGS[leg]
    units = [(H/'c1', l.split()[0]) for l in (H/'c1/units.txt').read_text().splitlines()]
    r = {}
    for d in pool.imap_unordered(unit, units, chunksize=1):
        assert not set(d) & set(r)
        r.update(d)
    obs = np.array(sorted(r))
    L, h1, h2 = (np.array([r[o][k] for o in obs]) for k in range(3))
    ref = pd.read_parquet(H/'reference/observations.parquet')
    tru = pd.read_parquet(H/'reference/truth.parquet')
    mag = 30-2.5*np.log10(ref['measured_flux_from_mag_auto'].to_numpy()[obs])
    return dict(obs=obs, L=L, h1=h1, h2=h2, mag=mag, case=tru.source_case.to_numpy()[obs],
                ident=tru.source_input_index.to_numpy()[obs])


if __name__ == '__main__':
    if not OUT.exists():
        with Pool(int(sys.argv[2])) as pool:
            data = {leg: load(leg, pool) for leg in LEGS}
        np.savez(OUT, **{f'{leg}_{k}': v for leg, d in data.items() for k, v in d.items()})
    z = np.load(OUT)
    # the selection the production run used: rule.json normalization (10.01M scenes, 5 shear nodes)
    norm = Path(json.loads((LEGS['minus']/'rule.json').read_text())['normalization'])
    spec = importlib.util.spec_from_file_location('paired_normalization', norm/'study.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    logz = np.log(mod.load_selection(norm).mean(0))
    two = lambda x: np.column_stack((x[:, 1]-x[:, 0], x[:, 3]-x[:, 2]))/(2*D)
    leg = {}
    for name in LEGS:
        L = z[f'{name}_L']
        dlogz = np.array([logz[2]-logz[1], logz[4]-logz[3]])/(2*D)
        recorded = json.loads((LEGS[name]/'c1/estimate.json').read_text())['selection_score_per_galaxy']
        assert np.allclose(dlogz, recorded, rtol=0, atol=1e-12), (dlogz, recorded)
        d2logz = np.array([logz[2]+logz[1]-2*logz[0], logz[4]+logz[3]-2*logz[0]])/D**2
        s = two(L)-dlogz
        e = two(z[f'{name}_h1'])-two(z[f'{name}_h2'])
        Ii = np.einsum('ia,ib->iab', s, s)-np.einsum('ia,ib->iab', e, e)/4
        h = -np.column_stack((L[:, 0]+L[:, 1], L[:, 2]+L[:, 3]))/D**2+d2logz
        g = np.linalg.solve(Ii.sum(0), s.sum(0))
        leg[name] = dict(s=s, I11=Ii[:, 0, 0], h=h, mag=z[f'{name}_mag'], case=z[f'{name}_case'], g=g,
                         dlogz=dlogz, d2logz=d2logz)
        print(f'{name}: {len(L)} galaxies; C1 g reproduced {g}; selection dlogZ {dlogz}, d2logZ {d2logz}')
    cases = np.arange(40, 140)
    bands = (('<21', -9, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25), ('25-25.8', 25, 26), ('ALL', -9, 99))
    # per case and band sums: [sum s1, sum I11, sum h1] per leg
    tab = {}
    for name, d in leg.items():
        idx = np.searchsorted(cases, d['case'])
        t = np.zeros((len(bands), 100, 3))
        for b, (_, lo, hi) in enumerate(bands):
            k = (d['mag'] >= lo) & (d['mag'] < hi)
            for j, x in enumerate((d['s'][:, 0], d['I11'], d['h'][:, 0])):
                np.add.at(t[b, :, j], idx[k], x[k])
        tab[name] = t
    rng = np.random.default_rng(20261030)
    boot = [rng.integers(0, 100, 100) for _ in range(2000)]

    def stats(k):
        p, m = tab['plus'][:, k].sum(1), tab['minus'][:, k].sum(1)    # (band, 3)
        S = (p[:, 0]-m[:, 0])/0.04
        I = (p[:, 1]+m[:, 1])/2
        H = (p[:, 2]+m[:, 2])/2
        return np.stack((S/I-1, S/H-1, H/I, (p[:, 0]/p[:, 1]-m[:, 0]/m[:, 1])/0.04-1,
                         (p[:, 0]/p[:, 2]-m[:, 0]/m[:, 2])/0.04-1), 1), S, I, H

    est, S, I, H = stats(np.arange(100))
    err = np.std([stats(k)[0] for k in boot], 0)
    print('\nfirst-stage (all galaxies scored at g=0), g1 only, case-bootstrap errors')
    print('band     | true slope S | OPG info I | curvature H | H/I | paired m one-step OPG (S/I-1) | with curvature (S/H-1) '
          '| per-leg ratio form OPG | per-leg ratio form curvature')
    for b, (name, _, _) in enumerate(bands):
        print(f'{name:8s} | {S[b]:12.0f} | {I[b]:10.0f} | {H[b]:11.0f} | {est[b,2]:.3f}+-{err[b,2]:.3f} | '
              f'{100*est[b,0]:+6.2f}+-{100*err[b,0]:4.2f}% | {100*est[b,1]:+6.2f}+-{100*err[b,1]:4.2f}% | '
              f'{100*est[b,3]:+6.2f}+-{100*err[b,3]:4.2f}% | {100*est[b,4]:+6.2f}+-{100*err[b,4]:4.2f}%')
