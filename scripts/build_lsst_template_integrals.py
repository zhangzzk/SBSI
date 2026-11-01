"""Tabulate GalSBI template integrals for the Rubin LSST r and i throughputs.

GalSBI's table magnitude calculator looks fluxes up at the nearest (z, E(B-V)) node. The SBSI prior always uses
E(B-V) = 0, so only that column is needed; the grid keeps E(B-V) = (0, 0.001) and GalSBI's production redshift grid
(0 to 7.462 in steps of 0.001). HSC_i2 is recomputed the same way and compared with GalSBI's shipped table to check
units and conventions. Run through the scheduler.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    from galsbi.ucat import filters_util, sed_templates_util
    import galsbi.ucat.sed_templates_util as stu

    ap = argparse.ArgumentParser()
    ap.add_argument('--throughputs', type=Path, required=True, help='dir with Rubin total_r.dat, total_i.dat (nm)')
    ap.add_argument('--hsc-filters', type=Path, required=True)
    ap.add_argument('--hsc-integrals', type=Path, required=True)
    ap.add_argument('--templates', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    collection = a.out / 'LSST_filters_collection.h5'
    with h5py.File(collection, 'w') as f:
        for band in ('r', 'i'):
            lam_nm, amp = np.loadtxt(a.throughputs / f'total_{band}.dat', comments='#', unpack=True)
            f[f'LSST_{band}/lam'] = lam_nm * 10.0  # Angstrom, as in GalSBI filter collections
            f[f'LSST_{band}/amp'] = amp
        with h5py.File(a.hsc_filters, 'r') as h:
            for name in ('HSC_i2', 'HSC_r2'):
                f[f'{name}/lam'] = np.array(h[name]['lam'])
                f[f'{name}/amp'] = np.array(h[name]['amp'])

    names = ['LSST_r', 'LSST_i', 'HSC_r2', 'HSC_i2']
    # Same unit scalings as galsbi get_magnitude_calculator_direct.
    filters = filters_util.load_filters(str(collection), filter_names=names, lam_scale=1e-4)
    templates = sed_templates_util.load_template_spectra(str(a.templates), lam_scale=1e-4, amp_scale=1e4)

    def grid(z_max, z_stepsize, excess_b_v_max, excess_b_v_stepsize):
        z = np.arange(0, z_max + z_stepsize, step=z_stepsize)  # as galsbi get_redshift_extinction_grid
        e = np.array([0.0, excess_b_v_stepsize])
        return z, e, np.repeat(z, len(e)), np.tile(e, len(z))

    stu.get_redshift_extinction_grid = grid  # only the E(B-V) = 0 column is ever used by SBSI
    integrals, e_grid, z_grid = sed_templates_util.get_template_integrals(
        templates, filters, filter_names=names, ids_templates=range(templates.n_templates))
    table = a.out / 'LSST_template_integrals_ebv0.h5'
    sed_templates_util.store_sed_integrals(str(table), integrals, e_grid, z_grid)

    with h5py.File(a.hsc_integrals, 'r') as h:
        z_ref = np.array(h['z'])
        report = dict(z_grid_matches=bool(np.array_equal(z_ref, z_grid)))
        for name in ('HSC_r2', 'HSC_i2'):
            worst = 0.0
            for t in range(templates.n_templates):
                ref = np.array(h[f'integrals/{name}/template_{t}'])[:, 0]
                new = integrals[name][t][:, 0]
                scale = np.max(np.abs(ref))
                worst = max(worst, float(np.max(np.abs(new - ref)) / scale))
            report[f'{name}_max_abs_diff_over_max'] = worst
    for band in ('r', 'i'):
        new = np.array([integrals[f'LSST_{band}'][t][:, 0] for t in range(templates.n_templates)])
        hsc = np.array([integrals[f'HSC_{band}2'][t][:, 0] for t in range(templates.n_templates)])
        ok = (hsc > 0) & (new > 0)
        dm = -2.5 * np.log10(new[ok] / hsc[ok])
        report[f'LSST_{band}_minus_HSC_{band}2_single_template_mag'] = dict(
            median=float(np.median(dm)), p1=float(np.percentile(dm, 1)), p99=float(np.percentile(dm, 99)))
    report.update(throughputs=str(a.throughputs), templates=str(a.templates), filters=str(collection),
                  table=str(table))
    (a.out / 'LSST_template_integrals_ebv0.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
