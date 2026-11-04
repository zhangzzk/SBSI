"""Three GalSBI-dev response figures from existing results (2026-11-04; plotting only, no new model evaluation
except the emulator profiles of galsbi_emulator_check_20261104).

usage (CPU, py31): python plots.py
1. emulator_vs_sims.png: emulator neighbour response vs simulation labels on held-out training-family cases 0-39
   (profiles.npz of galsbi_emulator_check_20261104), by measured size, MAG_AUTO, true r, neighbour true r.
2. flow_vs_sims.png: GalSBI-dev flow (steps4) shape response vs simulation on its 40 held-out training-family cases
   (map.npz of galsbi_guard_strength_20261103), by measured size and MAG_AUTO.
3. joint_constant_shear.png: GalSBI-dev flow + emulator on the constant-shear family, 40 cases
   (galsbi_bands40_constant_20261104 compare.json, split_by_mag.json), by cut and true magnitude.
Errors: case bootstrap (10,000, seed 20260925). Figures copied to SBSI/plots/galsbi_dev_<name>.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = Path(__file__).resolve().parent
EV = S.parent
D = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
PLOTS = Path('/home/z/Zekang.Zhang/SBSI/plots')
SIM, MODEL, FLOW, EMU = 'k', '#d1661b', '#1f5fa8', '#2a9d4b'


def boot_weights(n):
    return np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10_000).astype(float)


def save(fig, name):
    fig.savefig(S/name, dpi=140)
    (PLOTS/f'galsbi_dev_{name}').write_bytes((S/name).read_bytes())
    plt.close(fig)


def emulator_plot():
    z = np.load(D/'galsbi_emulator_check_20261104/profiles.npz')
    props = (('flux_radius', 'measured FLUX_RADIUS [px]'), ('mag_auto', 'measured MAG_AUTO'),
             ('true_r', 'true r of the galaxy'), ('neighbour_true_r', 'true r of the neighbour'))
    W = boot_weights(z['flux_radius'].shape[0])
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharex='col', gridspec_kw={'height_ratios': (1.3, 1)})
    out = {}
    for j, (k, label) in enumerate(props):
        x, e = z[k], z[f'{k}_edges']                       # case, selection, bin, (rows, label, pred)
        lo, hi = np.concatenate([[-np.inf], e]), np.concatenate([e, [np.inf]])
        if k == 'mag_auto':
            hi[-1] = 25.8
        ok = np.isfinite(lo) & np.isfinite(hi) & (x[:, 0, :, 0].sum(0) > 2e4)
        c = (lo+hi)/2
        a = ax[0, j]
        tot = x[:, 0].sum(0)
        a.plot(c[ok], (tot[:, 1]/tot[:, 0])[ok], 'o-', color=SIM, ms=3, label='simulation (label)')
        a.plot(c[ok], (tot[:, 2]/tot[:, 0])[ok], 's-', color=EMU, ms=3, label='emulator prediction')
        a.axhline(0, color='k', lw=.5)
        a.set(ylabel='mean response per pair', title=label)
        a2 = a.twinx()
        a2.bar(c[ok], tot[ok, 0]/1e6, width=(hi-lo)[ok], color='0.88', zorder=0)
        a2.set_ylabel('pairs [millions]', color='0.5')
        a.set_zorder(a2.get_zorder()+1)
        a.patch.set_visible(False)
        if j == 0:
            a.legend(fontsize=8)
        b = ax[1, j]
        out[k] = {}
        for s, (name, col, off) in enumerate((('main cut (> 3 px, < 25.8)', '0.3', 0), ('target (> 3 px, < 25)', MODEL, .3))):
            pt = x[:, s].sum(0)
            d = 100*(pt[:, 1]/pt[:, 2]-1)
            bs = np.einsum('bc,cnq->bnq', W, x[:, s])
            se = np.std(100*(bs[..., 1]/bs[..., 2]-1), axis=0, ddof=1)
            keep = ok & (pt[:, 0] > 2e4)
            w = (hi-lo)[keep]
            b.errorbar(c[keep]+off*w/2, d[keep], se[keep], fmt='o', color=col, ms=3, capsize=2, label=name)
            out[k][name] = {'centre': c[keep].tolist(), 'd_percent': d[keep].tolist(), 'se': se[keep].tolist()}
        b.axhline(0, color='k', lw=.6)
        b.axhspan(-5, 5, color='g', alpha=.08, lw=0)
        b.set(ylim=(-40, 40), xlabel=label, ylabel='sim / emulator − 1 [%]')
        if j == 0:
            b.legend(fontsize=8)
    fig.suptitle('GalSBI-dev emulator: neighbour-shear response vs simulation, held-out training-family cases 0–39 '
                 '(galaxy unsheared, neighbours sheared 0.2; pairs inside > 3 px & MAG_AUTO < 25.8)', fontsize=11)
    fig.tight_layout()
    save(fig, 'emulator_vs_sims.png')
    return out


def flow_plot():
    z = np.load(D/'galsbi_guard_strength_20261103/map.npz')
    r_edges, m_edges = z['r_edges'], z['m_edges']
    r_lo, r_hi = np.concatenate([[-np.inf], r_edges]), np.concatenate([r_edges, [np.inf]])
    m_lo, m_hi = np.concatenate([[-np.inf], m_edges]), np.concatenate([m_edges, [np.inf]])
    sim = z['sim_shape'][..., 0, 0]+z['sim_shape'][..., 1, 1]          # case, r, m
    flow = z['steps4_shape'][..., 0, 0]+z['steps4_shape'][..., 1, 1]
    n0 = z['sim_count'][:, 0]
    W = boot_weights(len(z['cases']))
    fig, ax = plt.subplots(2, 3, figsize=(17, 8), gridspec_kw={'height_ratios': (1.3, 1)})
    out = {}
    panels = (('size', 'measured FLUX_RADIUS [px]', 'MAG_AUTO < 25', m_hi <= 25+1e-9, 1),
              ('mag', 'measured MAG_AUTO', 'FLUX_RADIUS > 3 px', r_lo >= 3-1e-9, 2))
    for j, (key, xlabel, sel_label, sel, axis) in enumerate(panels):
        if axis == 1:
            s_, f_, n_ = (v[:, :, sel].sum(2) for v in (sim, flow, n0))
            c, ok = (r_lo+r_hi)/2, np.isfinite(r_lo) & np.isfinite(r_hi)
        else:
            s_, f_, n_ = (v[:, sel, :].sum(1) for v in (sim, flow, n0))
            c, ok = (m_lo+m_hi)/2, np.isfinite(m_lo) & np.isfinite(m_hi) & (m_hi <= 25.8+1e-9)
        ok = ok & (n_.sum(0) > 2000)
        a = ax[0, j]
        a.plot(c[ok], (s_.sum(0)/n_.sum(0))[ok], '-', color=SIM, lw=2, label='simulation')
        a.plot(c[ok], (f_.sum(0)/n_.sum(0))[ok], '-', color=FLOW, label='GalSBI-dev flow')
        a.axhline(0, color='k', lw=.5)
        if axis == 1:
            a.axvline(3, color='k', lw=.6, ls=':')
        a.set(ylabel='shape response per galaxy (R11 + R22)', title=f'{sel_label}: by {xlabel}')
        a2 = a.twinx()
        a2.bar(c[ok], n_.sum(0)[ok]/1e3, width=np.where(ok, r_hi-r_lo if axis == 1 else m_hi-m_lo, 0)[ok], color='0.88', zorder=0)
        a2.set_ylabel('galaxies per bin [thousands]', color='0.5')
        a.set_zorder(a2.get_zorder()+1)
        a.patch.set_visible(False)
        a.legend(fontsize=8)
        b = ax[1, j]
        diff = (f_-s_).sum(0)/n_.sum(0)
        se = np.std(np.einsum('bc,cn->bn', W, f_-s_)/np.einsum('bc,cn->bn', W, n_), axis=0, ddof=1)
        b.errorbar(c[ok], diff[ok], se[ok], fmt='o', color=FLOW, ms=3, capsize=2)
        b.axhline(0, color='k', lw=.6)
        if axis == 1:
            b.axvline(3, color='k', lw=.6, ls=':')
        b.set(xlabel=xlabel, ylabel='flow − simulation per galaxy')
        out[key] = {'centre': c[ok].tolist(), 'sim': (s_.sum(0)/n_.sum(0))[ok].tolist(),
                    'flow': (f_.sum(0)/n_.sum(0))[ok].tolist(), 'flow_minus_sim': diff[ok].tolist(), 'se': se[ok].tolist()}
        if axis == 1:
            a.set_xlim(1.5, 6)
            b.set_xlim(1.5, 6)

    def m_eq(t, mag):
        keep = (r_lo[:, None] >= t-1e-9) & (m_hi[None, :] <= mag+1e-9)
        ps, pf = (sim*keep).sum((1, 2)), (flow*keep).sum((1, 2))
        return 100*(ps.sum()/pf.sum()-1), np.std(100*(W@ps/(W@pf)-1), ddof=1)
    a = ax[0, 2]
    th = np.round(np.arange(2.5, 4.01, .1), 2)
    out['m_vs_size_threshold'] = {}
    for mag, col in ((25.0, MODEL), (25.8, '0.3')):
        v = np.array([m_eq(t, mag) for t in th])
        a.errorbar(th, v[:, 0], v[:, 1], fmt='o-', ms=3, capsize=2, color=col, label=f'MAG_AUTO < {mag:g}')
        out['m_vs_size_threshold'][str(mag)] = {'threshold': th.tolist(), 'm': v[:, 0].tolist(), 'se': v[:, 1].tolist()}
    a.axhline(0, color='k', lw=.6)
    a.axhspan(-.3, .3, color='g', alpha=.12, lw=0)
    a.axvline(3, color='k', lw=.6, ls=':')
    a.set(xlabel='size cut: FLUX_RADIUS > t [px]', ylabel='m = sim / flow − 1 [%]', title='flow-alone m vs size cut')
    a.legend(fontsize=8)
    a = ax[1, 2]
    mags = np.round(np.arange(24.0, 25.81, .25), 2)
    v = np.array([m_eq(3.0, mg) for mg in mags])
    a.errorbar(mags, v[:, 0], v[:, 1], fmt='o-', ms=3, capsize=2, color=FLOW, label='FLUX_RADIUS > 3 px')
    out['m_vs_mag_limit'] = {'limit': mags.tolist(), 'm': v[:, 0].tolist(), 'se': v[:, 1].tolist()}
    a.axhline(0, color='k', lw=.6)
    a.axhspan(-.3, .3, color='g', alpha=.12, lw=0)
    a.set(xlabel='magnitude cut: MAG_AUTO < limit', ylabel='m = sim / flow − 1 [%]', title='flow-alone m vs magnitude cut')
    a.legend(fontsize=8)
    fig.suptitle('GalSBI-dev flow: own shape response vs simulation, 40 held-out training-family cases '
                 '(galaxy sheared 0.05, neighbours at random angles; hard 0.1 px × 0.25 mag bins)', fontsize=11)
    fig.tight_layout()
    save(fig, 'flow_vs_sims.png')
    return out


def joint_plot(flow_curves):
    comp = json.loads((EV/'galsbi_bands40_constant_20261104/compare.json').read_text())['cuts']
    bymag = json.loads((EV/'galsbi_bands40_constant_20261104/split_by_mag.json').read_text())
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    size_cuts = (('mag25_radius_gt0.56', 2.8), ('mag25_radius_gt0.58', 2.9), ('mag25_radius_gt0.60', 3.0),
                 ('mag25_radius_gt0.62', 3.1), ('mag25_radius_gt0.64', 3.2), ('mag25_radius_ge0.75', 3.75))
    mag_cuts = (('mag24.6_radius_gt0.60', 24.6), ('mag25_radius_gt0.60', 25.0), ('mag25.4_radius_gt0.60', 25.4),
                ('mag25.8_radius_gt0.60', 25.8))
    for a, cuts, xlabel, title, ref in (
            (ax[0, 0], size_cuts, 'size cut: FLUX_RADIUS > t [px]', 'MAG_AUTO < 25: m vs size cut',
             (flow_curves['m_vs_size_threshold']['25.0']['threshold'], flow_curves['m_vs_size_threshold']['25.0']['m'])),
            (ax[0, 1], mag_cuts, 'magnitude cut: MAG_AUTO < limit', 'FLUX_RADIUS > 3 px: m vs magnitude cut',
             (flow_curves['m_vs_mag_limit']['limit'], flow_curves['m_vs_mag_limit']['m']))):
        xs = np.array([v for _, v in cuts])
        for w, col, off in (('C8', MODEL, -.012), ('true flags', '0.3', .012)):
            m = np.array([comp[f'{w}/{k}']['B_galsbi_dev'] for k, _ in cuts])
            a.errorbar(xs+off*(xs.max()-xs.min()), m[:, 0], m[:, 1], fmt='o', ms=5, capsize=3, color=col,
                       label=f'flow + emulator, constant shear ({w})')
        a.plot(*ref, '--', color=FLOW, lw=1, label='flow alone, training family (plot 2)')
        a.axhline(0, color='k', lw=.6)
        a.axhspan(-.3, .3, color='g', alpha=.12, lw=0)
        a.set(xlabel=xlabel, ylabel='m [%]', title=title)
        a.legend(fontsize=8)
    for a, cut, title in ((ax[1, 0], 'mag25_radius_gt0.60', 'target > 3 px & MAG_AUTO < 25'),
                          (ax[1, 1], 'mag25.8_radius_gt0.60', 'main > 3 px & MAG_AUTO < 25.8')):
        labels = ('<24', '24-25.5', '25.5-27.5')
        r = bymag[cut]
        x = np.arange(len(labels))
        a.bar(x-.2, [r[l]['R_sim'][0] for l in labels], .4, yerr=[r[l]['R_sim'][1] for l in labels], color='0.25',
              capsize=3, label='simulation')
        fl = np.array([r[l]['R_model_flow'][0] for l in labels])
        em = np.array([r[l]['R_model_emulator'][0] for l in labels])
        a.bar(x+.2, fl, .4, color=FLOW, label='model: flow (own shear)')
        a.bar(x+.2, em, .4, bottom=fl, color=EMU, label='model: emulator (neighbour shear)')
        for i, l in enumerate(labels):
            a.text(x[i], r[l]['R_sim'][0]+.04, f"m {r[l]['m_in_bin'][0]:+.1f} ± {r[l]['m_in_bin'][1]:.1f}%\n"
                   f"{100*r[l]['fraction'][0]:.0f}% of galaxies", ha='center', fontsize=8)
        a.set_xticks(x, [f'true r {l}' for l in labels])
        a.set(ylabel='shape response R11 (galaxies kept in both legs)', title=f'{title}: by true magnitude (true flags)',
              ylim=(0, 1.45))
        a.legend(fontsize=8, loc='upper right')
    fig.suptitle('GalSBI-dev flow + emulator on the constant-shear family (g1 = ±0.02, 40 cases)', fontsize=11)
    fig.tight_layout()
    save(fig, 'joint_constant_shear.png')


def main():
    out = {'emulator': emulator_plot(), 'flow': flow_plot()}
    joint_plot(out['flow'])
    (S/'plots.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
