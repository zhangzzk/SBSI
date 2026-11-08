"""Read-only CPU audit: nuisance factorization and saved-particle prior sensitivity.

No likelihood model is loaded, no inference population is redrawn, and no
production setting is changed. Execute with imports pinned to this checkout.
The saved-particle sensitivity is descriptive, not a convergence certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import log_expit, logsumexp
from scipy.stats import beta, betaprime, norm

from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.particle_diagnostics import combine_populations
from sbsi.scene_exchange import numerical_over_analytic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    source = Path(__file__).resolve().parents[2]
    files = [Path(__file__), *[source/p for p in (
        'sbsi/galsbi_prior.py', 'sbsi/galsbi_scene_prior.py', 'sbsi/galsbi_summary.py',
        'sbsi/galsbi_target_coordinates.py', 'sbsi/scene_exchange.py',
        'sbsi/particle_diagnostics.py', 'scripts/check_galsbi_continuous_prior.py')]]
    source_hashes = {str(path): digest(path) for path in files}
    galaxy, cosmo, _ = build_prior()
    distance = lambda z: cosmo.background.dist_ang_a(a=1/(1+z))
    coordinates = GalSBITargetCoordinates(galaxy, distance)
    prior = GalSBIScenePrior(galaxy)
    p = galaxy.p
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73,
                      moffat_beta=2.224, pixel_rms=.312)
    scene, _ = prior.sample(128, seed=2026102198)
    x = coordinates.encode(scene.targets)
    native, jac = coordinates.decode(x, scene.targets['type'])

    # Derive the integrand H_t independently of the log_intensity assembly.
    def core(data, kind):
        z, magnitude, c, radius = (data[k] for k in ('z', 'M_B', 'c', 'R_kpc'))
        ms, phi, alpha = galaxy.luminosity_parameters(z, kind)
        logx = .4*np.log(10)*(ms-magnitude)
        lum = np.log(.4*np.log(10)*phi*galaxy.volume(z))+(alpha+1)*logx-np.exp(logx)
        ac = galaxy.sed_alpha(z, kind)
        from scipy.special import gammaln
        sed = gammaln(ac.sum(1))-gammaln(ac).sum(1)+((ac-1)*np.log(c)).sum(1)
        mean, sigma = galaxy.size_parameters(z, data['M_i'], kind)
        size = norm.logpdf(np.log(radius), mean, sigma)  # R * p(R)
        aa, bb = galaxy.sersic_parameters(z, kind)
        sersic = betaprime.logpdf(data['n'], aa, bb)-galaxy.sersic_log_mass(z, kind)
        return lum+sed+size+sersic

    log_jn = np.log(p['sersic_n_max']-p['sersic_n_min'])+log_expit(x[:, 7])+log_expit(-x[:, 7])
    log_ja = np.log(p['ell_beta_emax'])+log_expit(x[:, 8])+log_expit(-x[:, 8])
    common = x[:, 0]+np.log(native['c']).sum(1)+log_jn+log_ja-np.log(2*np.pi)
    errors = []
    eta = x[:, 6]-.25*np.log1p(-np.where(native['a'] > 1e-7, native['a'], 0.)**2)
    shape_changed = x.copy()
    shape_changed[:, 8] += .7
    temporary, _ = coordinates.decode(shape_changed, native['type'])
    shape_changed[:, 6] = eta+.25*np.log1p(-temporary['a']**2)
    changed_native, _ = coordinates.decode(shape_changed, native['type'])
    shape_free_errors = []
    for kind_index, kind in enumerate(galaxy.types):
        typed = native | {'type': np.full(len(x), kind_index)}
        lp = galaxy.log_intensity(typed)+jac
        aa, bb = galaxy.shape_parameters(kind)
        assembled = core(native, kind)+common+beta.logpdf(native['a'], aa, bb, scale=p['ell_beta_emax'])
        valid = np.isfinite(lp)
        errors.append(float(np.max(np.abs(lp[valid]-assembled[valid]))))
        shape_free_errors.append(float(np.max(np.abs(core(native, kind)-core(changed_native, kind)))))

    rng = np.random.default_rng(2026102199)
    perturbed = x.copy()
    perturbed[:, 0] += rng.uniform(-.1, .1, len(x))
    perturbed[:, 0] = np.minimum(perturbed[:, 0], np.log(min(galaxy.z_max.values()))-.01)
    perturbed[:, 2:6] += rng.normal(0., .3, (len(x), 4))
    moved_native, _ = coordinates.decode(perturbed, native['type'])
    moved_scene = GalSBISceneBatch(moved_native, scene.neighbours, scene.indptr, scene.offset_arcsec)
    original_features, original_ptr, original_pairs = scene_model_inputs(
        scene, prior, distance, conditions, response_radius_arcsec=10., response_neighbours=20)
    moved_features, moved_ptr, moved_pairs = scene_model_inputs(
        moved_scene, prior, distance, conditions, response_radius_arcsec=10., response_neighbours=20)
    assert np.array_equal(original_ptr, moved_ptr)
    feature_error = float(np.max(np.abs(original_features.to_numpy()-moved_features.to_numpy())))
    pair_error = float(np.max(np.abs(original_pairs-moved_pairs)))
    first, _ = prior.sample_given_targets(native, 2026102200)
    second, _ = prior.sample_given_targets(moved_native, 2026102200)
    independence = all(np.array_equal(first.neighbours[key], second.neighbours[key]) for key in first.neighbours)
    independence &= np.array_equal(first.indptr, second.indptr) and np.array_equal(first.offset_arcsec, second.offset_arcsec)
    assert max(errors+shape_free_errors+[feature_error, pair_error]) < 1e-10
    assert independence
    print('FACTOR_AND_INPUT_CHECKS', errors, shape_free_errors, feature_error, pair_error, flush=True)

    ratios, scores, logzs, run_receipts, prior_errors, type_changes = [], [], [], [], [], []
    for number in range(8):
        directory = args.runs_root/f'dev_v5_{number}'
        result = json.loads((directory/'result.json').read_text())
        protocol = json.loads((directory/'protocol.json').read_text())
        assert result['status'] == 'completed'
        assert protocol['seed'] == 2026102170+number
        with np.load(directory/'particles.npz') as saved:
            array = saved['coordinates']
            shape = array.shape[:-1]
            xx = array.reshape(-1, 10)
            score = saved['scores'].copy()
            lp_saved = saved['log_prior'].reshape(-1)
        data, jac = coordinates.decode(xx, np.zeros(len(xx), int))
        terms, corrections = [], []
        for index in range(len(galaxy.types)):
            typed = data | {'type': np.full(len(xx), index)}
            terms.append(galaxy.log_intensity(typed))
            corrections.append(numerical_over_analytic(galaxy, typed))
        terms, corrections = np.column_stack(terms), np.column_stack(corrections)
        lp = logsumexp(terms, axis=1)
        lq = logsumexp(terms+corrections, axis=1)
        assert np.isfinite(lp).all() and np.isfinite(lq).all()
        ratios.append(np.exp(lq-lp).reshape(shape))
        scores.append(score)
        logzs.append(result['log_normalizer'])
        prior_errors.append(float(np.max(np.abs(lp+jac-lp_saved))))
        type_changes.append(np.max(np.abs(np.exp(terms+corrections-lq[:, None])-np.exp(terms-lp[:, None]))))
        run_receipts.append(dict(path=str(directory), seed=protocol['seed'],
            sha256={name: digest(directory/name) for name in ('protocol.json','result.json','particles.npz')}))
        print('READ_POPULATION', number, flush=True)
    ratios, scores, logzs = np.stack(ratios), np.stack(scores), np.array(logzs)
    assert max(prior_errors) < 1e-9
    observations = []
    for obs in range(ratios.shape[1]):
        rr, ss, zz = ratios[:, obs], scores[:, obs], logzs[:, obs]
        old = combine_populations(zz, ss.mean(1))
        adjusted = combine_populations(zz+np.log(rr.mean(1)), (rr[..., None]*ss).mean(1)/rr.mean(1)[:, None])
        observations.append(dict(observation_index=result['observations'][obs],
            particle_count=int(rr.size), log_q_over_p_quantiles=np.quantile(np.log(rr), [0,.01,.5,.99,1]).tolist(),
            run_weighted_intensity_mass_ratio=float(old['weights']@rr.mean(1)),
            analytic_score=old['mean'].tolist(), numeric_prior_score=adjusted['mean'].tolist(),
            score_shift=(adjusted['mean']-old['mean']).tolist(), analytic_score_mcse=old['mcse'].tolist(),
            absolute_shift_over_mcse=(np.abs(adjusted['mean']-old['mean'])/old['mcse']).tolist()))
    report = dict(status='passed', cpu_only=True, likelihood_models_loaded=False,
        source_sha256=source_hashes, input_receipts=run_receipts,
        factorization=dict(scene_count=128, maximum_log_density_error_by_type=errors,
            shape_free_core_max_error_by_type=shape_free_errors,
            nuisance_feature_max_error=feature_error, nuisance_response_pair_max_error=pair_error,
            sample_given_targets_environment_bitwise_independent=bool(independence)),
        saved_particle_sensitivity=dict(populations=8, target_particles=int(ratios.size),
            maximum_prior_replay_error=max(prior_errors),
            maximum_conditional_type_probability_change=float(max(type_changes)),
            observations=observations), elapsed_seconds=time.monotonic()-start,
        limitations=['The ratio uses numerical/analytic target INTENSITIES; omitted global constants cancel from score means.',
            'All eight predeclared v5 populations included with normalizer weighting; no production sampler was rerun.',
            'Sensitivity is conditional on represented saved states; unvisited modes and missing support are not bounded.',
            'Scores are the saved F64 diagnostic observable, evaluated under the F32 production posterior.',
            'Factorization identity does not supply or validate a numerical marginal-prior integrator.'])
    assert source_hashes == {str(path): digest(path) for path in files}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('BRIDGE_AUDIT_READY', json.dumps(observations), flush=True)


if __name__ == '__main__':
    main()
