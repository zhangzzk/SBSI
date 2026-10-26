"""Audit uniform GalSBI neighbour scenes; run on one scheduled CPU.

The independent reference uses the earlier GalSBI-generated galaxy marks and
count, assigns fresh uniform positions, and extracts disjoint circular patches.
It tests the requested unclustered model, not the original HEALPix geometry.
Reference rows are not bootstrapped. No image measurements or shear inference.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import kstest, poisson

from check_galsbi_continuous_prior import build_prior, digest, holm, load_reference, save, two_sample_checks, add_derived
from sbsi.galsbi_scene_prior import ARCSEC2_PER_SR, GalSBISceneBatch, GalSBIScenePrior
from sbsi.v39_inference import physical_close_crowding


def reference_patches(reference, targets, *, area_arcsec2, aperture, seed):
    """Thin a Poisson catalogue into nonoverlapping uniform-position patches.

    Partition its area into equal square cells; each circle fits inside one
    cell. Each catalogue row is placed once, independently and uniformly.
    Excluding the cell corners gives independent Poisson circular patches.
    Targets are independent of this catalogue and shared with the direct arm.
    """
    n = int(area_arcsec2//(2*aperture)**2)
    if not 1 <= n <= len(targets['z']):
        raise ValueError('reference area requires more target scenes or a larger aperture')
    side = np.sqrt(area_arcsec2/n)
    rng = np.random.default_rng(seed)
    owners = rng.integers(n, size=len(reference['z']))
    positions = rng.uniform(-side/2, side/2, size=(len(owners), 2))
    keep = np.square(positions).sum(1) <= aperture**2
    rows = np.flatnonzero(keep)
    # GalSBI stores types in blocks. Randomize list order so the reference has
    # the same ordered-iid mark measure as the scene density, not type sorting.
    rows = rng.permutation(rows)
    rows = rows[np.argsort(owners[rows], kind='stable')]
    counts = np.bincount(owners[rows], minlength=n)
    scenes = GalSBISceneBatch({k: v[:n] for k, v in targets.items()},
                             {k: v[rows] for k, v in reference.items()},
                             np.r_[0, np.cumsum(counts)], positions[rows])
    return scenes, dict(n_patches=n, square_cell_side_arcsec=side,
                       retained_neighbours=len(rows), discarded_cell_corner_rows=int((~keep).sum()),
                       input_rows=len(reference['z']), each_catalogue_row_used_at_most_once=True,
                       original_sky_positions_used=False)


def save_scenes(path, scenes, prior, sigma):
    np.savez_compressed(path, indptr=scenes.indptr, offset_arcsec=scenes.offset_arcsec,
                        counts=scenes.counts, crowding=prior.v39_crowding(scenes),
                        log_kernel=prior.log_kernel(scenes),
                        log_prob_estimate=prior.log_prob(scenes, surface_density=sigma),
                        **{f'target_{k}': v for k, v in scenes.targets.items()},
                        **{f'neighbour_{k}': v for k, v in scenes.neighbours.items()})


def plot(root, direct, reference, prior, expected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    max_count = max(direct.counts.max(), reference.counts.max())
    bins = np.arange(max_count+2)-.5
    for scene, label, color in ((direct, 'Continuous scenes', '#245EA6'),
                                (reference, 'Uniformly placed GalSBI marks', '#D56C24')):
        axes[0].hist(scene.counts, bins=bins, density=True, histtype='step', label=label, color=color)
        u = np.square(scene.offset_arcsec).sum(1)/prior.aperture_arcsec**2
        axes[1].hist(u, bins=np.linspace(0, 1, 26), density=True, histtype='step', color=color)
        # Same targets in these two environment arms.
        c = prior.v39_crowding(scene)[:len(reference.counts), 2]
        axes[2].hist(c, bins=np.linspace(0, 7, 40),
                     weights=np.ones(len(c))/len(c)/(7/39), histtype='step', color=color)
    k = np.arange(max_count+1)
    axes[0].plot(k, poisson.pmf(k, expected), color='0.3', linestyle='--', label='Poisson prediction')
    axes[0].set(xlabel='Neighbours within 10 arcsec', ylabel='Probability')
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(1, color='0.3', linestyle='--')
    axes[1].set(xlabel='Squared separation / aperture²', ylabel='Density')
    axes[2].set(xlabel='F8/C8 crowding at 2 arcsec', ylabel='Density')
    fig.suptitle('Fixed F24 population · uniform positions · all neighbours · no measurement cuts')
    fig.savefig(root/'neighbour_comparison.png', dpi=180)
    fig.savefig(root/'neighbour_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2026093021)
    parser.add_argument('--reference-position-seed', type=int, default=2026093022)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.reference_root/'result.json').read_text())
    generation = json.loads((args.reference_root/'reference_generation.json').read_text())
    ref_args = SimpleNamespace(output=args.reference_root, model_index=previous['model_index'],
                               reference_seed=previous['reference_seed'], nside=generation['nside'])
    reference, reference_paths = load_reference(ref_args)
    # Reference population identity must not drift while reusing its normalizer.
    checked = {}
    for name, expected_hash in previous['sources'].items():
        if '/site-packages/galsbi/' in name or '/cosmo-torrent/' in name or Path(name) in reference_paths:
            actual = digest(name)
            if actual != expected_hash:
                raise ValueError(f'reference source changed: {name}')
            checked[name] = actual
    galaxy, cosmo, _ = build_prior(previous['model_index'])
    prior = GalSBIScenePrior(galaxy)
    sigma = previous['normalization']['surface_density_sr']
    sigma_se = previous['normalization']['surface_density_mc_se_sr']
    expected = sigma*prior.area_sr
    direct, sampling = prior.sample(args.scenes, args.seed)
    print('SCENES_SAMPLED', sampling, flush=True)
    for data in (direct.targets, direct.neighbours, reference):
        add_derived(data, cosmo)
    area_arcsec2 = 4*np.pi/(12*generation['nside']**2)*ARCSEC2_PER_SR
    ref_scenes, ref_metadata = reference_patches(reference, direct.targets,
        area_arcsec2=area_arcsec2, aperture=prior.aperture_arcsec, seed=args.reference_position_seed)
    mu_se = np.sqrt(expected/args.scenes+(sigma_se*prior.area_sr)**2)
    count_z = (direct.counts.mean()-expected)/mu_se
    fano = direct.counts.var(ddof=1)/direct.counts.mean()
    spatial = {}
    for label, scene in (('continuous', direct), ('reference', ref_scenes)):
        u = np.square(scene.offset_arcsec).sum(1)/prior.aperture_arcsec**2
        phase = np.mod(np.arctan2(scene.offset_arcsec[:, 1], scene.offset_arcsec[:, 0]), 2*np.pi)/(2*np.pi)
        for key, value in (('radial_area', u), ('azimuth', phase)):
            test = kstest(value, 'uniform')
            spatial[f'{label}_{key}'] = dict(statistic=float(test.statistic), p_value=float(test.pvalue))
    holm(spatial)
    mark_checks = two_sample_checks(ref_scenes.neighbours, direct.neighbours)
    # Replay independent production code for the first 64 native neighbour lists.
    close = prior.v39_crowding(direct)
    max_replay = 0.
    for j in range(min(64, args.scenes)):
        lo, hi = direct.indptr[j:j+2]
        xy = np.vstack((np.zeros(2), direct.offset_arcsec[lo:hi]))
        dec = xy[:, 1]/3600
        frame = pd.DataFrame(dict(RA=xy[:, 0]/3600/np.cos(np.deg2rad(dec.mean())), DEC=dec,
                                  r=np.r_[direct.targets['m_r'][j], direct.neighbours['m_r'][lo:hi]]))
        replay = physical_close_crowding(frame, np.array([0]))[0]
        max_replay = max(max_replay, float(np.max(np.abs(replay-close[j]))))
    nref = len(ref_scenes.counts)
    delta = close[:nref]-prior.v39_crowding(ref_scenes)
    paired = dict(mean_difference=delta.mean(0).tolist(),
                  standard_error=(delta.std(0, ddof=1)/np.sqrt(nref)).tolist(),
                  scales_arcsec=[.5, 1., 2.], n_paired_targets=nref)
    paired_z = np.array(paired['mean_difference'])/np.array(paired['standard_error'])
    ref_count_z = ((ref_scenes.counts.mean()-expected)
                   /np.sqrt(expected/nref+(sigma_se*prior.area_sr)**2))
    finite = bool(np.isfinite(prior.log_kernel(direct)).all()
                  and np.isfinite(prior.log_kernel(ref_scenes)).all())
    gates = dict(all_scene_densities_finite=finite, count_mean=abs(count_z) < 5,
                 count_dispersion=abs(fano-1) < 5*np.sqrt(2/(args.scenes-1)),
                 reference_count_mean=abs(ref_count_z) < 5,
                 uniform_positions=min(x['holm_p_value'] for x in spatial.values()) >= .01,
                 reference_mark_distributions=min(x['holm_p_value'] for x in mark_checks.values()) >= .01,
                 production_crowding_replay=max_replay < 1e-10,
                 paired_crowding_means=bool((np.abs(paired_z) < 4).all()))
    gates = {k: bool(v) for k, v in gates.items()}
    save_scenes(args.output/'continuous_scenes.npz', direct, prior, sigma)
    save_scenes(args.output/'reference_patches.npz', ref_scenes, prior, sigma)
    plot(args.output, direct, ref_scenes, prior, expected)
    paths = [Path(__file__).resolve(), Path('sbsi/galsbi_scene_prior.py').resolve(),
             Path('sbsi/galsbi_prior.py').resolve(), Path('sbsi/v39_inference.py').resolve(),
             Path('scripts/check_galsbi_continuous_prior.py').resolve()]
    result = dict(status='passed_pilot_checks' if all(gates.values()) else 'diagnostic_failure',
                  model='Fischbacher+24', model_index=previous['model_index'],
                  seed=args.seed, reference_position_seed=args.reference_position_seed,
                  source_reference=str(args.reference_root.resolve()),
                  surface_density_sr=sigma, surface_density_mc_se_sr=sigma_se,
                  sampling=sampling, reference_patches=ref_metadata,
                  expected_neighbours=expected, mean_neighbours=float(direct.counts.mean()),
                  count_mean_z=float(count_z), count_fano=float(fano),
                  reference_mean_neighbours=float(ref_scenes.counts.mean()),
                  reference_count_mean_z=float(ref_count_z),
                  spatial_tests=spatial, mark_tests=mark_checks, paired_crowding=paired,
                  production_crowding_max_abs_error=max_replay, gates=gates,
                  sources={str(p): digest(p) for p in paths}, reference_hashes_checked=checked,
                  limitations=['Independent patches, not a shared sky realization.',
                               'Reference galaxy marks/count are from GalSBI; positions reassigned uniformly.',
                               'Same finite intrinsic population support as isolated pilot; no measured cuts.',
                               'Absolute log_prob uses an uncertain normalizer; fixed-population log_kernel does not.',
                               'No clustering, native-to-SBSI radius mapping, emulator prediction or MCMC.',
                               'No shear or sub-percent accuracy validation.'])
    save(args.output/'result.json', result)
    print('NEIGHBOUR_CHECK_COMPLETE', result['status'], gates, flush=True)
    if not all(gates.values()):
        raise SystemExit('neighbour diagnostics failed; see result.json')


if __name__ == '__main__':
    main()
