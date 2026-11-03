"""Corner contour plots of the FS2 and GalSBI galaxy populations, input (all rendered) and detected.

usage (Slurm CPU, py31): python population_corner.py
Families (20 cases each, the plus leg g1 = +0.02 for the detected sample):
  FS2     ConstGold cases 40-59, saved V3.9 evaluation rows (v36_f8_20260924_v1/evaluation/chunk*/rows); true
          magnitude is LSST-r. The rows hold the rendered galaxies the evaluation used.
  GalSBI  constant-shear cases 1000-1019, parents from galsbi_dev_constant_20261103/parent, measured legs joined
          as the frozen measured_leg does (CrossMatch -> Shapes, one-to-one); true magnitude is LSST-i.
Detected = matched to a detection with a usable measurement (finite, |g| < 1, radius > 0, flux > 0).
Figure 1 (true properties): true magnitude, log10 circularized Re [arcsec], Sersic n, |e|, input vs detected.
Figure 2 (measured properties, detected only): MAG_AUTO, FLUX_RADIUS [px], measured |g|, with the main cut lines.
Contours enclose 39% and 86% of each sample (1 and 2 sigma for a 2D Gaussian); diagonals are normalized densities.
Writes population_corner_true.png, population_corner_measured.png and population_summary.json next to this file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

S = Path(__file__).resolve().parent
C = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
FS2 = C/'v36_f8_20260924_v1/evaluation'
GSIMS = Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_g002/sims')
GPARENT = C/'galsbi_dev_constant_20261103/parent'
PLOTS = Path('/home/z/Zekang.Zhang/SBSI/plots')
TRUE_AXES = (('true magnitude (FS2 r, GalSBI i)', (18., 29.), 80), ('log$_{10}$ $R_e$ [arcsec]', (-1.6, .6), 80),
             ('Sérsic $n$', (.3, 6.5), 31), ('true $|e|$', (0., 1.), 80))
MEASURED_AXES = (('MAG_AUTO', (18., 28.), 80), ('FLUX_RADIUS [px]', (1., 8.), 80), ('measured $|g|$', (0., 1.), 80))
CUTS = {0: 25.8, 1: 3.0}   # measured axis index -> main cut value
LEVELS = (.86, .39)
STYLE = {('FS2', 'input'): ('#1f5fa8', '--'), ('FS2', 'detected'): ('#1f5fa8', '-'),
         ('GalSBI', 'input'): ('#d1661b', '--'), ('GalSBI', 'detected'): ('#d1661b', '-')}


def usable_of(values):
    with np.errstate(invalid='ignore'):
        return np.isfinite(values).all(1) & (np.square(values[:, :2]).sum(1) < 1) & (values[:, 2] > 0) & (values[:, 3] > 0)


def fs2_cases():
    for case in range(40, 60):
        with np.load(FS2/f'chunk{(case-40)//10}/rows/case{case:03d}.npz') as z:
            yield z['context'], z['measured_plus'], z['usable_plus']


def galsbi_cases():
    for case in range(1000, 1020):
        with np.load(GPARENT/f'case{case}.npz') as z:
            ids, context = z['input_index'], z['context']
        base = GSIMS/f'case{case}_0.02/real0/catalogues'
        cross = pd.read_feather(base/'CrossMatch/tile180.0_-0.5_rot0_matched.feather', columns=['id_detec', 'id_input'])
        shape = pd.read_feather(base/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',
                                columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS'])
        joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', how='inner',
                             validate='one_to_one').set_index('id_input')
        values = joined.reindex(ids)[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)
        with np.errstate(over='ignore', invalid='ignore'):
            values[:, 3] = 10**(-0.4*(values[:, 3]-30))
        yield context, values, usable_of(values)


def true_features(context):
    return np.column_stack([context[:, 3], np.log10(context[:, 4]), context[:, 2], np.hypot(context[:, 0], context[:, 1])])


def measured_features(values):
    return np.column_stack([30-2.5*np.log10(values[:, 3]), values[:, 2], np.hypot(values[:, 0], values[:, 1])])


def collect():
    samples, summary = {}, {}
    for family, cases in (('FS2', fs2_cases()), ('GalSBI', galsbi_cases())):
        true_in, true_det, meas = [], [], []
        for context, values, usable in cases:
            x = true_features(context)
            true_in.append(x)
            true_det.append(x[usable])
            meas.append(measured_features(values[usable]))
            print(family, len(context), int(usable.sum()), flush=True)
        n_in, n_det = sum(map(len, true_in)), sum(map(len, true_det))
        samples[family, 'input'] = np.concatenate(true_in)
        samples[family, 'detected'] = np.concatenate(true_det)
        samples[family, 'measured'] = np.concatenate(meas)
        m = samples[family, 'measured']
        main = (m[:, 0] < 25.8) & (m[:, 1] > 3.)
        summary[family] = {'cases': len(true_in), 'input_per_case': n_in/len(true_in),
                           'detected_per_case': n_det/len(true_in), 'detected_fraction': n_det/n_in,
                           'main_cut_per_case': float(main.sum())/len(true_in),
                           'median_true': {'input': np.median(samples[family, 'input'], 0).tolist(),
                                           'detected': np.median(samples[family, 'detected'], 0).tolist()},
                           'median_measured_detected': np.median(m, 0).tolist(),
                           'fraction_flux_radius_below_3px_of_mag_auto_lt_25.8':
                               float(((m[:, 0] < 25.8) & (m[:, 1] <= 3.)).sum()/(m[:, 0] < 25.8).sum())}
    return samples, summary


def levels(h):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat)/flat.sum()
    return sorted({flat[min(np.searchsorted(cum, q), len(flat)-1)] for q in LEVELS})


def corner(axes_spec, sets, path, title, cuts=None):
    k = len(axes_spec)
    fig, ax = plt.subplots(k, k, figsize=(2.6*k, 2.6*k))
    for (key, data) in sets:
        colour, ls = STYLE[key]
        for i in range(k):
            for j in range(k):
                a = ax[i, j]
                if j > i:
                    a.set_visible(False)
                    continue
                if i == j:
                    h, e = np.histogram(data[:, i], axes_spec[i][2], axes_spec[i][1], density=True)
                    a.step(e[:-1], h, where='post', color=colour, ls=ls, lw=1.3)
                    a.set_yticks([])
                else:
                    rx, ry = axes_spec[j][1], axes_spec[i][1]
                    h, ex, ey = np.histogram2d(data[:, j], data[:, i], (axes_spec[j][2], axes_spec[i][2]), (rx, ry))
                    h = gaussian_filter(h, 1.)
                    a.contour(.5*(ex[1:]+ex[:-1]), .5*(ey[1:]+ey[:-1]), h.T, levels(h), colors=colour,
                              linestyles=ls, linewidths=1.2)
    for i in range(k):
        for j in range(i+1):
            a = ax[i, j]
            a.set_xlim(axes_spec[j][1])
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune='both'))
            a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune='both'))
            if i != j:
                a.set_ylim(axes_spec[i][1])
            if cuts:
                if j in cuts:
                    a.axvline(cuts[j], color='k', lw=.7, ls=':')
                if i in cuts and i != j:
                    a.axhline(cuts[i], color='k', lw=.7, ls=':')
            a.set_xlabel(axes_spec[j][0] if i == k-1 else '')
            a.set_ylabel(axes_spec[i][0] if j == 0 and i > 0 else '')
            if i != k-1:
                a.set_xticklabels([])
            if j != 0 or i == 0:
                a.set_yticklabels([])
    handles = [plt.Line2D([], [], color=STYLE[key][0], ls=STYLE[key][1], label=f'{key[0]} {key[1]}') for key, _ in sets]
    note = ax[0, k-1]
    note.set_visible(True)
    note.axis('off')
    note.legend(handles=handles, loc='upper right', fontsize=11, frameon=False)
    note.text(1., .35, title, transform=note.transAxes, ha='right', va='top', fontsize=9)
    fig.subplots_adjust(hspace=.06, wspace=.06)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    samples, summary = collect()
    sets = [(key, samples[key]) for key in (('FS2', 'input'), ('FS2', 'detected'), ('GalSBI', 'input'), ('GalSBI', 'detected'))]
    out_true = S/'population_corner_true.png'
    corner(TRUE_AXES, sets, out_true,
           'True properties, 20 cases each\ninput = all rendered galaxies\ndetected = usable measurement (g1 = +0.02 leg)\n'
           'contours: 39% / 86%')
    sets = [((f, 'detected'), samples[f, 'measured']) for f in ('FS2', 'GalSBI')]
    out_meas = S/'population_corner_measured.png'
    corner(MEASURED_AXES, sets, out_meas,
           'Measured properties of detected galaxies\n(g1 = +0.02 leg, 20 cases each)\ndotted: main cut\n'
           'MAG_AUTO < 25.8, FLUX_RADIUS > 3 px\ncontours: 39% / 86%', CUTS)
    (S/'population_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    for f in (out_true, out_meas):
        (PLOTS/f'fs2_galsbi_{f.name}').write_bytes(f.read_bytes())
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
