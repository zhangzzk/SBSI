"""FFT size GalSim needs to draw each GalSBI scene galaxy with the MultiBand_ImSim renderer.

The renderer draws an untruncated Sersic (n clipped to 0.3-6.2, q clipped at 0.05, hlr = Re*sqrt(q), then sheared to
axis ratio q and by the constant shear), convolved with the trunc=4.5*FWHM Moffat PSF, with method 'auto' (pixel
included) on a stamp GalSim sizes itself. drawFFT refuses FFTs above gsparams.maximum_fft_size (default 8192) and the
whole tile then fails (the galsbi-resim renderer branch retries such galaxies at 32768). Re above 100 arcsec would send a
galaxy to the renderer's bulge+disk branch, which scenes do not support, so max_Re is reported too. Galaxies are ranked by Re/stepk(n, hlr=1), which orders their drawn extent; the top --top per
scene get GalSim's exact size (same steps as GSObject.drawFFT_makeKImage). Needs galsim: run in the sims1 env.
"""
import argparse
import json
from pathlib import Path

import galsim
import numpy as np
import pandas as pd

SERSIC_N_MIN, SERSIC_N_MAX, Q_MIN = 0.3, 6.2, 0.05


def fft_size(prof, scale):
    n = galsim.Image.good_fft_size(prof.getGoodImageSize(scale))
    n = max(n, prof.gsparams.minimum_fft_size)
    dk = 2*np.pi/(n*scale)
    return n if n*dk/2 > prof.maxk else int(np.ceil(prof.maxk/dk))*2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=Path, required=True)
    ap.add_argument('--seeing', type=float, default=0.71)
    ap.add_argument('--beta', type=float, default=2.458)
    ap.add_argument('--scale', type=float, default=0.2)
    ap.add_argument('--shear', type=float, default=0.02)
    ap.add_argument('--top', type=int, default=3000)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    psf = galsim.Moffat(beta=a.beta, fwhm=a.seeing, trunc=4.5*a.seeing)
    pix = galsim.Pixel(a.scale)
    limit = galsim.GSParams().maximum_fft_size
    rows, worst = [], []
    for path in sorted(a.scenes.glob('scene*.feather')):
        s = pd.read_feather(path, columns=['Re', 'axis_ratio', 'position_angle', 'sersic_n', 'r'])
        n = s['sersic_n'].clip(SERSIC_N_MIN, SERSIC_N_MAX).to_numpy()
        unit = {v: galsim.Sersic(n=v, half_light_radius=1.0).stepk for v in np.unique(n)}
        proxy = s['Re'].to_numpy()/np.array([unit[v] for v in n])
        top = np.argsort(proxy)[::-1][:a.top]
        sizes = []
        for i in top:
            q = max(float(s['axis_ratio'].iat[i]), Q_MIN)
            gal = galsim.Sersic(n=float(n[i]), half_light_radius=float(s['Re'].iat[i])*q**0.5, trunc=0)
            gal = gal.shear(q=q, beta=float(s['position_angle'].iat[i])*galsim.degrees).shear(g1=a.shear, g2=0.0)
            sizes.append(fft_size(galsim.Convolve(gal, psf, pix), a.scale))
        sizes = np.array(sizes)
        case = int(path.stem.removeprefix('scene'))
        over = top[sizes > limit]
        rows.append(dict(case=case, max_Re=float(s['Re'].max()), n_checked=len(top), n_over_limit=int(len(over)),
                         max_fft=int(sizes.max()),
                         max_fft_mem_gb=float(24*sizes.max()**2/1024**3),
                         smallest_checked_fft=int(sizes.min()), monotone_top=bool(sizes[0] == sizes.max())))
        for i, size in zip(top[:5], sizes[:5]):
            worst.append(dict(case=case, row=int(i), fft=int(size), Re=float(s['Re'].iat[i]),
                              q=float(s['axis_ratio'].iat[i]), n=float(n[i]), mag=float(s['r'].iat[i])))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(limit=limit, cases=rows, largest=sorted(worst, key=lambda w: -w['fft'])[:20])
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    for w in report['largest']:
        print(json.dumps(w))


if __name__ == '__main__':
    main()
