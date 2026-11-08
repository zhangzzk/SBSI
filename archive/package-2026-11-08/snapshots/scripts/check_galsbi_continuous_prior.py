"""Generate an isolated GalSBI catalogue and audit a continuous native prior.

Run through the scheduler. GalSBI is an optional installed dependency of this
diagnostic, not a new SBSI dependency. No F8/C8/emulator or measured cuts run.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, betaprime, chi2_contingency, kstest, ks_2samp, norm

from sbsi.galsbi_prior import ExactMemo, GalSBIContinuousPrior


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def build_prior(model_index=0, grid_size=8193):
    import PyCosmo
    from galsbi import load
    from galsbi.ucat.galaxy_sampling_util import intp_z_m_cut
    from galsbi.ucat.galaxy_population_models.galaxy_luminosity_function import maximum_redshift
    from galsbi.ucat.plugins.sample_galaxies_photo import get_magnitude_calculator_table

    config = importlib.import_module('galsbi.configs.config_Fischbacher+24_intrinsic')
    p = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    row = load.load_abc_posterior('Fischbacher+24')[model_index]
    p.update({k: row[k].item() for k in row.dtype.names})
    p.update(filters=['r', 'i'], extinction_map_file_name=None, catalog_precision=np.float64)
    par = SimpleNamespace(**p)
    cosmo = PyCosmo.build()
    cosmo.set(h=p['h'], omega_m=p['omega_m'])
    mag = get_magnitude_calculator_table(['B', 'r', 'i'], par)
    # Exact per-redshift memory: MH moves mostly keep z. Neighbour proposals
    # add fresh z, so this memory restarts sooner to keep insertions cheap.
    luminosity_distance = ExactMemo(lambda z: cosmo.background.dist_lum_a(a=1/(1+z)), limit=500_000)

    def photometry(z, M_B, c):
        zero = np.zeros(len(z))
        absolute = mag(redshifts=zero, excess_b_v=zero, coeffs=c, filter_names=['B', 'i'])
        coeff = c * 10 ** (.4*(absolute['B']-M_B))[:, None]
        distance = luminosity_distance(z)
        coeff *= ((1e-5/distance)**2/(1+z))[:, None]
        apparent = mag(redshifts=z, excess_b_v=zero, coeffs=coeff, filter_names=['r', 'i'])
        return dict(m_r=apparent['r'], m_i=apparent['i'],
                    M_i=M_B+absolute['i']-absolute['B'])

    @ExactMemo
    def volume(z):
        transverse = cosmo.background.dist_trans_a(a=1/(1+z))
        expansion = np.sqrt(cosmo.params.omega_m*(1+z)**3 + cosmo.params.omega_l)
        return cosmo.params.c/cosmo.params.H0 * transverse**2/expansion

    cap = intp_z_m_cut(cosmo, mag, par)
    zmax = {}
    for t in p['galaxy_types']:
        zmax[t] = maximum_redshift(
            cap, p['lum_fct_m_max'], p['lum_fct_z_max'], p['lum_fct_parametrization'],
            p[f'lum_fct_z_const_{t}'], p[f'lum_fct_alpha_{t}'],
            (p[f'lum_fct_m_star_{t}_slope'], p[f'lum_fct_m_star_{t}_intcpt']), p['seed_ngal'])
    prior = GalSBIContinuousPrior(p, photometry, volume, cap, zmax, grid_size=grid_size)
    return prior, cosmo, config


def generate(args):
    import galsbi
    from galsbi import GalSBI

    mask = np.zeros(12*args.nside**2, dtype=bool)
    mask[0] = True
    model = GalSBI('Fischbacher+24', verbosity='info')
    model(mode='intrinsic', model_index=args.model_index,
          file_name=str(args.output/'reference'), healpix_map=mask,
          nside_sampling=args.nside, seed=args.reference_seed,
          filters=['r', 'i'], extinction_map_file_name=None, catalog_precision=np.float64)
    report = dict(galsbi_version=galsbi.__version__, model='Fischbacher+24',
                  model_index=args.model_index, seed=args.reference_seed,
                  nside=args.nside, healpix_pixel=0, extinction=0,
                  catalog_precision='float64', filters=['HSC_r2', 'HSC_i2'],
                  counts={k: len(v) for k, v in model.load_catalogs(model_index=args.model_index).items()})
    save(args.output/'reference_generation.json', report)
    print('REFERENCE_COMPLETE', report, flush=True)


def load_reference(args):
    from cosmic_toolbox import arraytools as at

    root = args.output
    report = json.loads((root/'reference_generation.json').read_text())
    expected = dict(model='Fischbacher+24', model_index=args.model_index,
                    seed=args.reference_seed, nside=args.nside, extinction=0,
                    catalog_precision='float64', filters=['HSC_r2', 'HSC_i2'])
    if any(report[key] != value for key, value in expected.items()):
        raise ValueError('comparison arguments do not match the reference generation manifest')
    paths = [root/f'reference_{args.model_index}_{band}_ucat.gal.cat' for band in ('r', 'i')]
    r, i = [at.load_hdf(str(p)) for p in paths]
    if not np.array_equal(r['id'], i['id']) or len(np.unique(r['id'])) != len(r):
        raise ValueError('reference band identifiers must match uniquely')
    if np.any(r['excess_b_v'] != 0):
        raise ValueError('this pilot requires zero foreground extinction')
    c = r['template_coeffs_abs'].astype(float)
    c /= c.sum(axis=1, keepdims=True)
    d = dict(type=r['galaxy_type'].astype(int), z=r['z'], M_B=r['abs_mag_lumfun'], c=c,
             R_kpc=r['int_r50_phys'], n=r['sersic_n'],
             a=np.hypot(r['int_e1'], r['int_e2']),
             psi=np.mod(np.arctan2(r['int_e2'], r['int_e1']), 2*np.pi),
             m_r=r['int_mag'], m_i=i['int_mag'], M_i=i['abs_mag'],
             radius_arcsec=r['int_r50_arcsec'])
    return d, paths


def add_derived(data, cosmo):
    distance = cosmo.background.dist_ang_a(a=1/(1+data['z']))
    data['radius_arcsec'] = data['R_kpc']/(1e3*distance)*180/np.pi*3600
    data['e1_distortion'] = data['a']*np.cos(data['psi'])
    data['e2_distortion'] = data['a']*np.sin(data['psi'])
    data['color_r_i'] = data['m_r']-data['m_i']
    data['log_radius_arcsec'] = np.log(data['radius_arcsec'])


def morphology_pit(data, prior):
    result = {k: np.empty(len(data['z'])) for k in ('size', 'sersic', 'shape', 'phase')}
    for index, t in enumerate(prior.types):
        use = data['type'] == index
        z = data['z'][use]
        mean, sigma = prior.size_parameters(z, data['M_i'][use], t)
        result['size'][use] = norm.cdf((np.log(data['R_kpc'][use])-mean)/sigma)
        a, b = prior.sersic_parameters(z, t)
        lo = betaprime.cdf(prior.p['sersic_n_min'], a, b)
        hi = betaprime.cdf(prior.p['sersic_n_max'], a, b)
        result['sersic'][use] = (betaprime.cdf(data['n'][use], a, b)-lo)/(hi-lo)
        a, b = prior.shape_parameters(t)
        result['shape'][use] = beta.cdf(data['a'][use]/prior.p['ell_beta_emax'], a, b)
        result['phase'][use] = data['psi'][use]/(2*np.pi)
    return result


def holm(tests):
    ordered = sorted(tests, key=lambda k: tests[k]['p_value'])
    maximum = 0.
    for rank, key in enumerate(ordered):
        maximum = max(maximum, min(1., (len(ordered)-rank)*tests[key]['p_value']))
        tests[key]['holm_p_value'] = maximum


def two_sample_checks(reference, continuous):
    checks = {}
    names = ('z', 'M_B', 'm_r', 'm_i', 'color_r_i', 'log_radius_arcsec', 'n',
             'a', 'e1_distortion', 'e2_distortion')
    for name in names:
        r = ks_2samp(reference[name], continuous[name], method='asymp')
        checks[name] = dict(statistic=float(r.statistic), p_value=float(r.pvalue))
    for index in range(5):
        r = ks_2samp(reference['c'][:, index], continuous['c'][:, index], method='asymp')
        checks[f'sed_{index}'] = dict(statistic=float(r.statistic), p_value=float(r.pvalue))
    for x, y in (('m_r', 'log_radius_arcsec'), ('m_r', 'color_r_i'), ('z', 'n'),
                 ('m_r', 'a'), ('m_r', 'z')):
        edges = []
        for key in (x, y):
            edges.append(np.r_[-np.inf, np.unique(np.quantile(
                np.r_[reference[key], continuous[key]], np.arange(1, 6)/6)), np.inf])
        hist = np.array([np.histogram2d(d[x], d[y], bins=edges)[0].ravel()
                         for d in (reference, continuous)])
        expected = hist.sum(axis=0)[None, :] * hist.sum(axis=1)[:, None]/hist.sum()
        good = (expected >= 5).all(axis=0)
        table = hist[:, good]
        rare = hist[:, ~good].sum(axis=1)
        if rare.sum():
            table = np.column_stack((table, rare))
        result = chi2_contingency(table, correction=False)
        checks[f'joint_{x}_{y}'] = dict(statistic=float(result.statistic),
                                       p_value=float(result.pvalue), cells=table.shape[1])
    counts = np.array([np.bincount(d['type'], minlength=2) for d in (reference, continuous)])
    result = chi2_contingency(counts, correction=False)
    checks['type_fraction'] = dict(statistic=float(result.statistic), p_value=float(result.pvalue),
                                   counts=counts.tolist())
    holm(checks)
    return checks


def plot_comparison(root, reference, continuous, pits, model_index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    columns = [('m_r', 'True HSC r magnitude'), ('color_r_i', 'True r − i'),
               ('z', 'Redshift'), ('log_radius_arcsec', 'ln native angular half-light radius'),
               ('n', 'Sérsic index'), ('a', 'Native distortion amplitude')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (name, label) in zip(axes.flat, columns):
        merged = np.r_[reference[name], continuous[name]]
        bounds = np.quantile(merged, [.001, .999])
        edges = np.linspace(*bounds, 46)
        for data, color, title in ((reference, '#245EA6', 'GalSBI catalogue'),
                                    (continuous, '#D56C24', 'SBSI continuous prior')):
            values = data[name]
            # Normalize by the whole sample, including tails outside the displayed range.
            ax.hist(values, bins=edges, weights=np.ones(len(values))/len(values)/np.diff(edges)[0],
                    histtype='step', color=color, label=title, linewidth=1.4)
        ax.set(xlabel=label, ylabel='Density')
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f'Isolated F24 galaxies · posterior row {model_index} · N={len(reference["z"]):,} / '
                 f'{len(continuous["z"]):,}\nCommon intrinsic i-band support; no measurement cuts')
    fig.savefig(root/'population_comparison.png', dpi=180)
    fig.savefig(root/'population_comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for ax, (name, values) in zip(axes, pits.items()):
        ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype='step', color='#245EA6')
        ax.axhline(1, color='0.4', linestyle='--')
        ax.set(xlabel=f'{name} conditional CDF', ylabel='Density')
    fig.suptitle('GalSBI catalogue evaluated with SBSI analytic morphology distributions')
    fig.savefig(root/'conditional_density_checks.png', dpi=180)
    plt.close(fig)


def compare(args):
    import galsbi
    import sbsi.galsbi_prior as implementation

    prior, cosmo, config = build_prior(args.model_index)
    print('PRIOR_READY', prior.parent_rate, prior.z_max, flush=True)
    reference, reference_paths = load_reference(args)
    saved_radius = reference['radius_arcsec'].copy()
    mapped = prior.photometry(reference['z'], reference['M_B'], reference['c'])
    replay = {key: float(np.max(np.abs(mapped[key]-reference[key]))) for key in mapped}
    add_derived(reference, cosmo)
    replay['radius_arcsec'] = float(np.max(np.abs(saved_radius-reference['radius_arcsec'])))
    print('POINTWISE_REPLAY', replay, flush=True)
    continuous, norm_estimate = prior.sample(args.proposals, args.sample_seed)
    add_derived(continuous, cosmo)
    print('CONTINUOUS_COMPLETE', norm_estimate, flush=True)
    log_ref = prior.log_intensity(reference)
    log_sample = prior.log_intensity(continuous)
    pits = morphology_pit(reference, prior)
    pit_checks = {}
    for name, values in pits.items():
        r = kstest(values, 'uniform')
        pit_checks[name] = dict(statistic=float(r.statistic), p_value=float(r.pvalue))
    holm(pit_checks)
    integrals = {}
    for t in prior.types:
        a, b = prior.shape_parameters(t)
        integrals[f'{t}_shape'] = quad(lambda x: beta.pdf(x, a, b), 0, 1, epsabs=1e-9)[0]
        for z in (.1, 1., 3.):
            a, b = prior.sersic_parameters(np.array([z]), t)
            a = float(a[0])
            lo, hi = prior.p['sersic_n_min'], prior.p['sersic_n_max']
            mass = betaprime.cdf(hi, a, b)-betaprime.cdf(lo, a, b)
            integrals[f'{t}_sersic_z{z}'] = quad(lambda x: betaprime.pdf(x, a, b)/mass,
                                               lo, hi, epsabs=1e-9)[0]
    checks = two_sample_checks(reference, continuous)
    area = 4*np.pi/(12*args.nside**2)
    expected = norm_estimate['surface_density_sr']*area
    count_sd = np.sqrt(expected + (norm_estimate['surface_density_mc_se_sr']*area)**2)
    count_z = (len(reference['z'])-expected)/count_sd
    grid = prior.grid_convergence()
    gates = dict(photometric_replay=max(replay.values()) < 1e-8,
                 all_reference_densities_finite=bool(np.isfinite(log_ref).all()),
                 all_sample_densities_finite=bool(np.isfinite(log_sample).all()),
                 morphology_normalization=max(abs(v-1) for v in integrals.values()) < 1e-7,
                 luminosity_grid=max(abs(v) for v in grid.values()) < 5e-4,
                 magnitude_inversion=norm_estimate['maximum_magnitude_inverse_cdf_error'] < 1e-5,
                 distribution_checks=min(v['holm_p_value'] for v in checks.values()) >= .01,
                 morphology_pit=min(v['holm_p_value'] for v in pit_checks.values()) >= .01,
                 catalogue_count=abs(count_z) < 4)
    gates = {key: bool(value) for key, value in gates.items()}
    np.savez_compressed(args.output/'continuous_catalogue.npz', **continuous)
    np.savez_compressed(args.output/'density_evaluations.npz',
                        reference_log_intensity=log_ref, continuous_log_intensity=log_sample,
                        reference_log_prob=prior.log_prob(reference, surface_density=norm_estimate['surface_density_sr']),
                        **{f'pit_{k}': v for k, v in pits.items()})
    plot_comparison(args.output, reference, continuous, pits, args.model_index)
    source_paths = [Path(__file__), Path(implementation.__file__), Path(config.__file__)]
    package_root = Path(galsbi.__file__).parent
    source_paths += [package_root/rel for rel in (
        'galsbi.py', 'load.py', 'ucat/config/common.py', 'ucat/magnitude_calculator.py',
        'ucat/filters_util.py',
        'ucat/plugins/sample_galaxies_photo.py', 'ucat/plugins/sample_galaxies_morph.py',
        'ucat/galaxy_sampling_util.py', 'ucat/galaxy_population_models/galaxy_luminosity_function.py',
        'ucat/galaxy_population_models/galaxy_sed.py', 'ucat/galaxy_population_models/galaxy_size.py',
        'ucat/galaxy_population_models/galaxy_shape.py', 'ucat/galaxy_population_models/galaxy_light_profile.py')]
    from cosmo_torrent import data_path
    source_paths += [Path(data_path('Fischbacher+24_abc_posterior'))/'abc_posterior.h5',
                     Path(prior.p['templates_int_tables_file_name'])]
    sources = {str(path): digest(path) for path in source_paths+reference_paths}
    prefixes = ('lum_fct_', 'sersic_', 'ell_', 'logr50_', 'template_coeff_')
    snapshot = {}
    for key, value in prior.p.items():
        if key.startswith(prefixes) or key in ('h', 'omega_m', 'reference_band', 'galaxy_types',
                                               'gals_mag_min', 'gals_mag_max', 'seed_ngal',
                                               'ellipticity_sampling_method'):
            snapshot[key] = np.asarray(value).tolist() if isinstance(value, (np.ndarray, np.generic)) else value
    save(args.output/'parameters.json', snapshot)
    result = dict(status='passed_pilot_checks' if all(gates.values()) else 'diagnostic_failure',
                  scope='isolated native-coordinate prior; not shear calibration',
                  model='Fischbacher+24', model_index=args.model_index, galsbi_version=galsbi.__version__,
                  reference_seed=args.reference_seed, continuous_seed=args.sample_seed,
                  reference_count=len(reference['z']), continuous_count=len(continuous['z']),
                  normalization=norm_estimate, pointwise_replay_max_abs=replay,
                  luminosity_grid_relative_changes=grid, conditional_density_integrals=integrals,
                  distribution_tests=checks, morphology_pit=pit_checks,
                  reference_pit_correlation=np.corrcoef(np.array(list(pits.values()))).tolist(),
                  expected_reference_count=expected, reference_count_z_score=count_z,
                  gates=gates, sources=sources,
                  limitations=['Native latent density, not marginal five-property density.',
                               'Photometry is shared with GalSBI and uses its nearest-grid tables.',
                               'Continuous LF sampling approximates the intended law; reference uses discrete grids.',
                               'Surface-density normalizer has reported Monte Carlo uncertainty.',
                               'No neighbours, image measurements, selection response or shear inference.',
                               'Native angular radius has not yet been mapped to SBSI circularized radius.',
                               'Distributional checks at this sample size do not establish 0.3% accuracy.'])
    save(args.output/'result.json', result)
    print('COMPARISON_COMPLETE', result['status'], gates, flush=True)
    if not all(gates.values()):
        raise SystemExit('comparison diagnostics failed; see result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate', 'compare', 'all'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--nside', type=int, default=512)
    parser.add_argument('--reference-seed', type=int, default=2026093011)
    parser.add_argument('--sample-seed', type=int, default=2026093012)
    parser.add_argument('--proposals', type=int, default=131072)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('generate', 'all'):
        generate(args)
    if args.stage in ('compare', 'all'):
        compare(args)


if __name__ == '__main__':
    main()
