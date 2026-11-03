"""Number densities (per arcmin^2) of the FS2 and GalSBI populations, input and detected.

usage (Slurm CPU, py31): python number_density.py
Full scenes of 20 cases each, plus leg (g1 = +0.02): FS2 ConstGold cases 40-59 (lsst_sims_fs2_25876_constant),
GalSBI constant cases 1000-1019 (galsbi_resim/constant_g002). Both are one 18181^2 px image at 0.2"; the area is
the galaxy placement box from the input RA/Dec extent (spherical). Stages per case:
  input        all rendered galaxies (gals_info)
  sextractor   all SExtractor detections
  usable       detections matched one-to-one to an input galaxy (CrossMatch -> Shapes) with a usable measurement
               (finite, |g| < 1, radius > 0, flux > 0)
  mag25.8      usable and MAG_AUTO < 25.8
  main         usable, MAG_AUTO < 25.8 and FLUX_RADIUS > 3 px
Errors are the case-to-case standard deviation / sqrt(20). Writes number_density.json and number_density.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
PLOTS = Path('/home/z/Zekang.Zhang/SBSI/plots')
FAMILIES = {'FS2': (Path('/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant'), range(40, 60)),
            'GalSBI': (Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_g002/sims'), range(1000, 1020))}
TILE = 'tile180.0_-0.5'
STAGES = ('input', 'sextractor', 'usable', 'mag25.8', 'main')
MAG_EDGES = np.arange(18., 29.51, .25)
RADIUS_EDGES = np.arange(1., 8.01, .1)
COLOUR = {'FS2': '#1f5fa8', 'GalSBI': '#d1661b'}


def one_case(root, case):
    base = root/f'case{case}_0.02/real0/catalogues'
    gals = pd.read_feather(base/f'input/gals_info_{TILE}.feather', columns=['index_input', 'RA_input', 'DEC_input', 'r_input'])
    ra, dec = np.radians(gals['RA_input']), np.radians(gals['DEC_input'])
    area = (ra.max()-ra.min())*(np.sin(dec.max())-np.sin(dec.min()))*(180*60/np.pi)**2
    n_sex = len(pd.read_feather(base/f'SExtractor/{TILE}_bandr_rot0.feather', columns=['NUMBER']))
    cross = pd.read_feather(base/f'CrossMatch/{TILE}_rot0_matched.feather', columns=['id_detec', 'id_input'])
    shape = pd.read_feather(base/f'Shapes/shape_catalogue_detect_position_all_{TILE}.feather',
                            columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS'])
    joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', how='inner', validate='one_to_one')
    joined = joined.merge(gals[['index_input', 'r_input']], left_on='id_input', right_on='index_input', how='inner',
                          validate='one_to_one')
    g = joined[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float)
    mag, radius = joined['MAG_AUTO'].to_numpy(float), joined['FLUX_RADIUS'].to_numpy(float)
    with np.errstate(invalid='ignore'):
        usable = np.isfinite(g).all(1) & (np.square(g).sum(1) < 1) & (radius > 0) & np.isfinite(mag) & np.isfinite(radius)
        bright = usable & (mag < 25.8)
        main = bright & (radius > 3.)
    counts = np.array([len(gals), n_sex, usable.sum(), bright.sum(), main.sum()])/area
    hists = {'true_mag_input': np.histogram(gals['r_input'], MAG_EDGES)[0]/area,
             'true_mag_usable': np.histogram(joined['r_input'][usable], MAG_EDGES)[0]/area,
             'true_mag_main': np.histogram(joined['r_input'][main], MAG_EDGES)[0]/area,
             'mag_auto_usable': np.histogram(mag[usable], MAG_EDGES)[0]/area,
             'flux_radius_mag25.8': np.histogram(radius[bright], RADIUS_EDGES)[0]/area}
    return area, counts, hists


def main():
    result, curves = {}, {}
    for family, (root, cases) in FAMILIES.items():
        areas, counts, hists = [], [], []
        for case in cases:
            a, c, h = one_case(root, case)
            areas.append(a)
            counts.append(c)
            hists.append(h)
            print(family, case, round(a, 1), np.round(c, 2), flush=True)
        counts = np.array(counts)
        mean, se = counts.mean(0), counts.std(0, ddof=1)/np.sqrt(len(counts))
        result[family] = {'area_arcmin2': float(np.mean(areas)),
                          'per_arcmin2': {s: [float(m), float(e)] for s, m, e in zip(STAGES, mean, se)},
                          'fraction_of_input': {s: float(m/mean[0]) for s, m in zip(STAGES, mean)}}
        curves[family] = {k: np.mean([h[k] for h in hists], 0) for k in hists[0]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    mc = .5*(MAG_EDGES[1:]+MAG_EDGES[:-1])
    rc = .5*(RADIUS_EDGES[1:]+RADIUS_EDGES[:-1])
    for family, c in curves.items():
        col = COLOUR[family]
        ax[0].step(mc, c['true_mag_input']/.25, where='mid', color=col, ls='--', label=f'{family} input')
        ax[0].step(mc, c['true_mag_usable']/.25, where='mid', color=col, label=f'{family} detected')
        ax[0].step(mc, c['true_mag_main']/.25, where='mid', color=col, ls=':', label=f'{family} main cut')
        ax[1].step(mc, c['mag_auto_usable']/.25, where='mid', color=col, label=f'{family} detected')
        ax[2].step(rc, c['flux_radius_mag25.8']/.1, where='mid', color=col, label=f'{family} detected, MAG_AUTO < 25.8')
    ax[0].set(yscale='log', ylim=(1e-2, 300), xlabel='true magnitude (FS2 r, GalSBI i)', ylabel='galaxies / arcmin$^2$ / mag')
    ax[1].set(yscale='log', ylim=(1e-2, 300), xlabel='MAG_AUTO', ylabel='galaxies / arcmin$^2$ / mag')
    ax[1].axvline(25.8, color='k', lw=.7, ls=':')
    ax[2].set(xlabel='FLUX_RADIUS [px]', ylabel='galaxies / arcmin$^2$ / px')
    ax[2].axvline(3., color='k', lw=.7, ls=':')
    for a in ax:
        a.legend(fontsize=8, frameon=False)
    fig.suptitle('Number density, 20 cases each (g1 = +0.02 leg), same 18181$^2$ px image at 0.2"', fontsize=11)
    fig.tight_layout()
    out = S/'number_density.png'
    fig.savefig(out, dpi=150)
    (PLOTS/'fs2_galsbi_number_density.png').write_bytes(out.read_bytes())
    (S/'number_density.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
