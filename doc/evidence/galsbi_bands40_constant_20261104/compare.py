"""Paired end-to-end comparison: GalSBI-dev with the band-guard flow (A) vs GalSBI-dev (B), 40 constant-shear cases.

usage (CPU, py31): python compare.py
Reads sums/caseNNNN.npz of both study roots (scripts/evaluate_galsbi_constant.py draws; rule.md here). Same
estimator as that script's summary: per leg, mean selected shape = sum(e)/sum(mass); model shape = flow + emulator
(blend) shift; R11 = (plus - minus)/(2 x 0.02); m = 100 (R_sim/R_model - 1). The simulation side is identical in A
and B, so A - B is a pure model change. Errors: case bootstrap over the 40 cases (10,000, seed 20260925).
Check: B rows 0-2 equal the first study's sums and B row 3 equals the mag25 study's row 1 (relative < 1e-10).
Writes compare.json and prints a table with the flow-alone change from check_bands.json beside it.
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
D = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
A, B = D/'galsbi_bands40_constant_20261104', D/'galsbi_dev_constant_grid_20261104'
CASES = [c for chunk in (0, 4, 8, 12) for c in range(1000+10*chunk, 1010+10*chunk)]
H = .02
# flow-alone check rows (check_bands.json) matching each end-to-end cut: (radius_min px, magnitude_max)
FLOW_ROW = {'mag25.8_radius_gt0.60': 'radius_gt_3_magnitude_lt_25.8',
            'mag25.8_radius_ge0.75': 'radius_gt_3.75_magnitude_lt_25.8',
            'mag25_radius_gt0.60': 'radius_gt_3_magnitude_lt_25',
            'radius_gt0.60': 'radius_gt_3',
            'radius_ge0.75': 'radius_gt_3.75',
            'mag25.8': 'magnitude_lt_25.8',
            'mag25': 'magnitude_lt_25',
            'mag25_radius_gt0.56': 'radius_gt_2.8_magnitude_lt_25',
            'mag25_radius_gt0.58': 'radius_gt_2.9_magnitude_lt_25',
            'mag25_radius_gt0.62': 'radius_gt_3.1_magnitude_lt_25',
            'mag25_radius_gt0.64': 'radius_gt_3.2_magnitude_lt_25',
            'mag24.6_radius_gt0.60': 'radius_gt_3_magnitude_lt_24.6',
            'mag25.4_radius_gt0.60': 'radius_gt_3_magnitude_lt_25.4',
            'uncut': 'global'}


def load(root):
    model, sim, cuts = [], [], None
    for case in CASES:
        with np.load(root/f'sums/case{case}.npz') as z:
            model.append(z['model'])
            sim.append(z['sim'])
            cuts = list(z['cuts'])
    return np.array(model), np.array(sim), cuts


def response(sums):   # (..., leg, 3): mass, e1, e2 -> R11
    mean = sums[..., 1:]/sums[..., :1]
    return ((mean[..., 0, :]-mean[..., 1, :])/(2*H))[..., 0]


def views(model, sim, w, k):
    x = model[:, w, :, k, :, :5].sum(-2)                          # case, leg, 5
    m = np.concatenate([x[..., :1], x[..., 1:3]+x[..., 3:5]], -1)
    s = sim[:, :, k, :, :3].sum(-2)
    return m, s


def main():
    (ma, sa, cuts), (mb, sb, cuts_b) = load(A), load(B)
    assert cuts == cuts_b and np.array_equal(sa, sb), 'simulation sides differ'
    # reproduction check of B against the earlier full studies
    worst = 0.
    for case_i, case in enumerate(CASES):
        for study, rows in (('galsbi_dev_constant_20261103', ((0, 0), (1, 1), (2, 2))),
                            ('galsbi_dev_constant_mag25_20261103', ((3, 1),))):
            with np.load(D/study/f'sums/case{case}.npz') as z:
                for new, old in rows:
                    for x_new, x_old in ((mb[case_i][:, :, new, :, :5], z['model'][:, :, old]),
                                         (sb[case_i][:, new, :, :3], z['sim'][:, old])):
                        worst = max(worst, float(np.max(np.abs(x_new-x_old)/np.maximum(np.abs(x_old), 1e-300))))
    print(f'B reproduces the earlier studies: max relative difference {worst:.2e}')
    assert worst < 1e-10
    flow = json.loads((D/'galsbi_guard_strength_20261103/check_bands.json').read_text())
    flow_names = [c['name'] for c in flow['cuts']]
    n = len(CASES)
    W = np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10000).astype(float)
    out = {'cases': CASES, 'reproduction_max_relative_difference': worst, 'cuts': {}}
    for w, weight in enumerate(('C8', 'true flags')):
        print(f'\n{weight}: m [%] +- case SE (40 cases); paired change A - B; flow-alone change bands40 - steps4')
        for k, cut in enumerate(cuts):
            (ma_k, s_k), (mb_k, _) = views(ma, sa, w, k), views(mb, sb, w, k)
            rs, ra, rb = response(s_k.sum(0)), response(ma_k.sum(0)), response(mb_k.sum(0))
            bs, ba, bb = (response(np.einsum('bc,c...->b...', W, x)) for x in (s_k, ma_k, mb_k))
            m_a, m_b = 100*(rs/ra-1), 100*(rs/rb-1)
            boot_a, boot_b = 100*(bs/ba-1), 100*(bs/bb-1)
            row = {'A_band_guard': [float(m_a), float(boot_a.std(ddof=1))],
                   'B_galsbi_dev': [float(m_b), float(boot_b.std(ddof=1))],
                   'A_minus_B': [float(m_a-m_b), float((boot_a-boot_b).std(ddof=1))]}
            j = flow_names.index(FLOW_ROW[cut])
            flow_change = flow['flows']['bands40']['m_eq'][j]-flow['flows']['steps4']['m_eq'][j]
            row['flow_alone_change'] = flow_change
            out['cuts'][f'{weight}/{cut}'] = row
            print(f"  {cut:24s} A {m_a:+.2f} +- {row['A_band_guard'][1]:.2f}   B {m_b:+.2f} +- {row['B_galsbi_dev'][1]:.2f}"
                  f"   A-B {m_a-m_b:+.2f} +- {row['A_minus_B'][1]:.2f}   flow alone {flow_change:+.2f}")
    (S/'compare.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
