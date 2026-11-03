"""Where in measured (FLUX_RADIUS, MAG_AUTO) the GalSBI flows miss the simulated shear response.

usage (CPU, py31): python map_analysis.py
Reads $DATA_DIR/sbsi_caches/galsbi_guard_strength_20261103/map.npz (run_galsbi_guard_strength.py map; validation
cases, hard bins of 0.1 px x 0.25 mag). For a hard cut c, the response is T_c = sum_{bins in c} S / N0_c, with S the
binned e response (design-scaled) and N0_c the zero-leg count; the mean-shape term mu_c x (count response) is
dropped (both factors are ~0 for an isotropic population; checked against the soft-cut check numbers).
m_eq = 100 ((T11 + T22)_sim / (T11 + T22)_flow - 1). Errors: case bootstrap (10,000, seed 20260925).
Writes map_analysis.json and map_analysis.png (also SBSI/plots/galsbi_flow_response_map.png).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = Path(__file__).resolve().parent
MAP = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_guard_strength_20261103/map.npz')
PLOTS = Path('/home/z/Zekang.Zhang/SBSI/plots')
FLOWS = ('nll_only', 'steps4', 'steps40')
LABEL = {'nll_only': 'no guard', 'steps4': 'GalSBI-dev (4-step guard)', 'steps40': '40-step guard'}
COLOUR = {'nll_only': '#444444', 'steps4': '#d1661b', 'steps40': '#1f5fa8'}
THRESHOLDS = np.round(np.arange(2.0, 4.51, 0.1), 2)
MAGS = (25.0, 25.4, 25.8)


def main():
    z = np.load(MAP)
    r_edges, m_edges = z['r_edges'], z['m_edges']
    nc = len(z['cases'])
    # bin i of an axis covers [edges[i-1], edges[i]); bin 0 is below edges[0]
    r_lo = np.concatenate([[-np.inf], r_edges])
    m_hi = np.concatenate([m_edges, [np.inf]])
    diag = {k: z[f'{k}_shape'][..., 0, 0] + z[f'{k}_shape'][..., 1, 1] for k in ('sim',) + FLOWS}   # case, r, m
    n0 = z['sim_count'][:, 0]                                                                        # case, r, m
    W = np.random.default_rng(20260925).multinomial(nc, np.full(nc, 1/nc), size=10_000).astype(float)

    def cut_sum(x, t, mag, w):   # sum over bins with r_lo >= t and m_hi <= mag (bins fully inside the cut)
        sel = (r_lo[:, None] >= t - 1e-9) & (m_hi[None, :] <= mag + 1e-9)
        return np.einsum('bc,crm->b', w, x * sel)

    def m_eq(flow, t, mag):
        point = 100 * (cut_sum(diag['sim'], t, mag, np.ones((1, nc)))/cut_sum(diag[flow], t, mag, np.ones((1, nc))) - 1)
        boot = 100 * (cut_sum(diag['sim'], t, mag, W)/cut_sum(diag[flow], t, mag, W) - 1)
        return float(point[0]), boot

    result = {'curves': {}, 'paired_vs_nll_only': {}}
    for mag in MAGS:
        for flow in FLOWS:
            vals = [m_eq(flow, t, mag) for t in THRESHOLDS]
            result['curves'][f'{flow}_mag{mag}'] = {'threshold': THRESHOLDS.tolist(), 'm_eq': [v[0] for v in vals],
                                                    'se': [float(v[1].std(ddof=1)) for v in vals]}
    # per radius bin, mag < 25.8 and < 25.0: per-galaxy response (sim and flows) and flow-minus-sim share of the
    # main-cut response
    profile = {}
    for mag in (25.0, 25.8):
        msel = m_hi <= mag + 1e-9
        n = n0[..., msel].sum((0, 2))
        main = (r_lo >= 3 - 1e-9)
        total_sim_main = diag['sim'][:, main][..., msel].sum()
        prof = {'r_lo': r_lo.tolist(), 'count': n.tolist(),
                'sim_per_galaxy': (diag['sim'][..., msel].sum((0, 2))/np.maximum(n, 1)).tolist()}
        for flow in FLOWS:
            d = diag[flow][..., msel].sum(2) - diag['sim'][..., msel].sum(2)          # case, r
            prof[f'{flow}_per_galaxy'] = (diag[flow][..., msel].sum((0, 2))/np.maximum(n, 1)).tolist()
            prof[f'{flow}_minus_sim_pct_of_main'] = (100*d.sum(0)/total_sim_main).tolist()
            prof[f'{flow}_minus_sim_pct_of_main_se'] = (100*np.einsum('bc,cr->br', W, d).std(0, ddof=1)/total_sim_main).tolist()
        profile[str(mag)] = prof
    result['profile'] = profile
    (S/'map_analysis.json').write_text(json.dumps(result, indent=1)+'\n')

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for i, mag in enumerate((25.0, 25.8)):
        a = ax[0, i]
        for flow in FLOWS:
            c = result['curves'][f'{flow}_mag{mag}']
            a.errorbar(THRESHOLDS, c['m_eq'], c['se'], color=COLOUR[flow], marker='o', ms=3, capsize=2, label=LABEL[flow])
        a.axhline(0, color='k', lw=.6)
        a.axhspan(-.3, .3, color='g', alpha=.12, lw=0)
        a.axvline(3, color='k', lw=.6, ls=':')
        a.set(xlabel='size threshold: FLUX_RADIUS > t [px] (hard cut)', ylabel='flow response error m_eq [%]',
              title=f'MAG_AUTO < {mag}: error vs size threshold')
        a.legend(fontsize=8)
    prof = profile['25.0']
    rc = np.array(prof['r_lo'][1:]) + .05
    sl = slice(1, len(rc)+1)
    a = ax[1, 0]
    a.plot(rc, prof['sim_per_galaxy'][sl], 'k-', lw=2, label='simulation')
    for flow in FLOWS:
        a.plot(rc, prof[f'{flow}_per_galaxy'][sl], color=COLOUR[flow], label=LABEL[flow])
    a2 = a.twinx()
    a2.fill_between(rc, np.array(prof['count'][sl])/1e3, step='mid', color='0.85', zorder=0)
    a2.set_ylabel('galaxies per 0.1 px [thousands]', color='0.5')
    a.set_zorder(a2.get_zorder()+1)
    a.patch.set_visible(False)
    a.set(xlim=(1.5, 6), ylim=(-1, 2.5), xlabel='FLUX_RADIUS bin [px]', ylabel='response per galaxy (R11 + R22)',
          title='MAG_AUTO < 25.0: response of galaxies in each size bin')
    a.axvline(3, color='k', lw=.6, ls=':')
    a.legend(fontsize=8)
    a = ax[1, 1]
    for flow in FLOWS:
        a.errorbar(rc, prof[f'{flow}_minus_sim_pct_of_main'][sl], prof[f'{flow}_minus_sim_pct_of_main_se'][sl],
                   color=COLOUR[flow], marker='o', ms=3, capsize=2, label=LABEL[flow])
    a.axhline(0, color='k', lw=.6)
    a.axvline(3, color='k', lw=.6, ls=':')
    a.set(xlim=(1.5, 6), xlabel='FLUX_RADIUS bin [px]', ylabel='flow − sim response, % of sim main-cut response',
          title='MAG_AUTO < 25.0: excess flow response per 0.1 px bin')
    a.legend(fontsize=8)
    fig.suptitle('GalSBI flow vs simulation shear response by measured size (validation cases, flow alone)', fontsize=11)
    fig.tight_layout()
    fig.savefig(S/'map_analysis.png', dpi=140)
    (PLOTS/'galsbi_flow_response_map.png').write_bytes((S/'map_analysis.png').read_bytes())
    for mag in MAGS:
        print(f'MAG < {mag}')
        for flow in FLOWS:
            c = result['curves'][f'{flow}_mag{mag}']
            print(f'  {flow:9s}', ' '.join(f'{t:.1f}:{v:+.2f}' for t, v in zip(THRESHOLDS, c['m_eq']) if 2.4 <= t <= 4.1))
    print('per-galaxy response, mag < 25, r bins 2.0-4.5')
    for j, r in enumerate(prof['r_lo']):
        if 2.0 <= r < 4.5:
            print(f"  {r:.1f} n={prof['count'][j]:8.0f} sim {prof['sim_per_galaxy'][j]:.4f} " +
                  ' '.join(f"{f} {prof[f + '_per_galaxy'][j]:.4f} ({prof[f + '_minus_sim_pct_of_main'][j]:+.2f}±{prof[f + '_minus_sim_pct_of_main_se'][j]:.2f})" for f in FLOWS))


if __name__ == '__main__':
    main()
