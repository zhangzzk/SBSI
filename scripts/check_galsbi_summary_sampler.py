"""Small fixed-summary independence-MH pilot; run under Slurm.

No fitted summary prior, tuning on injected shear, finite-atom reuse, or
catalogue shear estimate. A failed mixing diagnostic is a scientific result,
not grounds for increasing the run automatically.
"""
import argparse
from pathlib import Path
import time

import arviz as az
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from check_galsbi_continuous_prior import build_prior, digest, save
from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.catalogue_likelihood import predict_detection_probability
from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.galsbi_summary import (SummaryBatch, SummaryLikelihood, compress_scenes,
                                native_shapes, scene_model_inputs)
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.summary_sampling import prior_metropolis
from sbsi.v39_inference import V39CatalogueCache


def geometry_replay():
    """Independent SVD check against the installed renderer's actual matrix."""
    from ufig.rendering_util import compute_ellip_matrices
    a = np.array([0., 1e-8, .1, .5, .9])
    phase = np.array([0., .3, 1.1, 2.8, 5.7])
    radius = np.array([.3, .5, .7, 1., 2.])
    native = dict(a=a, psi=phase, z=np.ones(len(a)), R_kpc=radius*1e6/(180/np.pi*3600))
    mapped = native_shapes(native, lambda z: np.full(len(z), 1000.))
    matrix = compute_ellip_matrices(radius, a*np.cos(phase), a*np.sin(phase))
    matrix[:, 0, 1] *= -1  # draw_photon_gal applies this minus sign.
    axes = np.linalg.svd(matrix, compute_uv=False)
    expected = np.column_stack((axes[:, 0], np.sqrt(axes.prod(1))))
    actual = np.column_stack((mapped['semimajor'], mapped['circularized']))
    return float(np.max(np.abs(expected-actual)))


def prepared(prior, angular_distance, conditions, model, *, n, seed):
    scenes, report = prior.sample(n, seed)
    zero, ptr, pairs = scene_model_inputs(scenes, prior, angular_distance, conditions,
        response_radius_arcsec=10., response_neighbours=20)
    response = CatalogueDiskResponse(ptr, pairs, model)
    print('SCENES_READY', seed, report, 'response_pairs', len(pairs), flush=True)
    return zero, response, report


def mock_panel(zero, response, flow, detector, *, seed, shear, retain_population=False):
    context = zero.copy()
    context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
        zero.e1_input_p.to_numpy(), zero.e2_input_p.to_numpy(), *shear)
    rng = np.random.default_rng(seed)
    probability = predict_detection_probability(detector, context)
    usable = np.flatnonzero(rng.random(len(zero)) < probability)
    torch.manual_seed(seed+1)
    draws, gap = flow.sample_with_shape_log_gap(context.iloc[usable], batch_size=4096)
    draws, gap = draws[:, 0], gap[:, 0]
    keep = np.flatnonzero((draws[:, 2] > 3.) & (draws[:, 3] > 10**(.4*(30-25.8))))
    draws, gap, ids = draws[keep], gap[keep], usable[keep]
    b, _ = response.predict(ids, draws[:, 2], draws[:, 3])
    device = flow.device
    transformed, gap, _ = transport_with_log_gap(torch.as_tensor(draws[:, :2], device=device),
        torch.as_tensor(gap, device=device), torch.as_tensor(b[:, None]*np.array(shear), device=device))
    draws[:, :2] = transformed.cpu().numpy()
    frame = pd.DataFrame(draws, columns=flow.target_transform.target_names)
    frame[LOG_GAP_COLUMN] = gap.cpu().numpy()
    if retain_population:
        return frame.reset_index(drop=True), dict(n_inputs=len(zero), n_usable=len(usable),
            n_retained=len(frame), selection='fresh C8 Bernoulli and measured radius>3 pixels/mag<25.8',
            panel_selection='all retained observations in generation order'), ids
    magnitude = 30-2.5*np.log10(draws[:, 3])
    indices, strata = [], []
    for lo, hi in ((-np.inf, 20.), (20., 22.), (22., 24.), (24., 25.8)):
        rows = np.flatnonzero((magnitude >= lo) & (magnitude < hi))
        strata.append(dict(lower=None if not np.isfinite(lo) else lo, upper=hi, available=len(rows),
                           selected_row=int(rows[0]) if len(rows) else None))
        if len(rows):
            indices.append(rows[0])
    return frame.iloc[indices].reset_index(drop=True), dict(n_inputs=len(zero), n_usable=len(usable),
        n_retained=len(frame), strata=strata, selection='fresh C8 Bernoulli and measured radius>3 pixels/mag<25.8',
        panel_selection='first retained observation in each measured-magnitude stratum'), ids[indices]


def diagnostics(values):
    if np.ptp(values) == 0:
        return dict(rhat=None, ess_bulk=None, constant=True)
    rhat, ess = float(az.rhat(values, method='rank')), float(az.ess(values, method='bulk'))
    return dict(rhat=rhat if np.isfinite(rhat) else None,
                ess_bulk=ess if np.isfinite(ess) else None, constant=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--mock-inputs', type=int, default=8192)
    parser.add_argument('--draws-per-chain', type=int, default=2048)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026093041)
    args = parser.parse_args()
    if (args.mock_inputs < 1 or args.draws_per_chain < 32 or args.chains < 2
            or (args.output/'result.json').exists()):
        raise ValueError('fresh output and positive pilot dimensions required')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    sources = [Path(__file__), Path('sbsi/galsbi_summary.py'), Path('sbsi/summary_sampling.py'),
        Path('sbsi/galsbi_prior.py'), Path('sbsi/galsbi_scene_prior.py'), Path('scripts/check_galsbi_continuous_prior.py'),
        Path('sbsi/catalogue_disk_likelihood.py'), Path('sbsi/catalogue_likelihood.py'),
        Path('sbsi/catalogue_disk_response.py'), Path('sbsi/measurement_model.py'), Path('configs/models_v3_9_like.json')]
    protocol = dict(seed=args.seed, mock_inputs=args.mock_inputs, chains=args.chains,
        draws_per_chain=args.draws_per_chain, burnin=args.draws_per_chain//2, shear=[.02, 0.],
        density_batch_size=4096, response_pairing=dict(radius_arcsec=10., nearest_k=20, truth_cuts=None),
        conditions=conditions, mapping='installed UFig zero-flexion rendered ellipse',
        proposal='fresh iid GalSBI scene compressed at each observed radius/flux',
        prior='fixed F24 row0, native i14-28 support, uniform neighbours',
        code_gates=dict(renderer_geometry_max_error=1e-12, log_kernel_replay_max_error=1e-6),
        exploration_gates=dict(max_rank_rhat=1.01, min_bulk_ess=400, min_prior_importance_ess_per_chain=100),
        hypothesis='compression preserves likelihood; fresh-prior MH may fail to explore narrow bright posteriors',
        no_shear_estimate=True, implementation_sha256={str(p.resolve()): digest(p) for p in sources})
    save(args.output/'protocol.json', protocol)
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    likelihood = SummaryLikelihood(flow, detector)
    geometry_error = geometry_replay()
    if geometry_error > protocol['code_gates']['renderer_geometry_max_error']:
        raise AssertionError('native geometry disagrees with installed UFig renderer')
    print('GEOMETRY_REPLAY', geometry_error, flush=True)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)
    def angular_distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))
    mock_zero, mock_response, mock_sampling = prepared(prior, angular_distance, conditions, model,
        n=args.mock_inputs, seed=args.seed)
    observed, retention, generating_ids = mock_panel(mock_zero, mock_response, flow, detector,
        seed=args.seed+1, shear=protocol['shear'])
    observed.to_parquet(args.output/'observations.parquet', index=False)
    save(args.output/'retention.json', retention)
    print('MOCK_PANEL', retention, flush=True)
    # These generating inputs are diagnostic records only, never chain starts.
    mock_zero.iloc[generating_ids].to_parquet(args.output/'generating_features.parquet', index=False)
    del mock_zero, mock_response
    n = args.chains*args.draws_per_chain
    zero, response, sampling = prepared(prior, angular_distance, conditions, model, n=n, seed=args.seed+10)
    zero.to_parquet(args.output/'proposal_features.parquet', index=False)
    np.savez_compressed(args.output/'response_pairs.npz', indptr=response.indptr, features=response.features)
    results, replay_errors = [], []
    g0 = protocol['shear']
    for index in range(len(observed)):
        obs = observed.iloc[[index]]
        summary = compress_scenes(zero, response, obs.iloc[0, 2:4].to_numpy())
        kernel = likelihood.log_kernel(summary, obs, *g0)
        # Independent old full-pair path, exactly the same 4096-sized density calls.
        cache = V39CatalogueCache(zero, np.ones(n), {}, detector=detector)
        full = DiskCatalogueLikelihood(flow, cache, disk_response=response)
        repeated = obs.iloc[np.zeros(n, dtype=int)].reset_index(drop=True)
        original_density = full.conditional_log_likelihood(repeated, *g0,
            atom_indices=np.arange(n), object_chunk=4096)
        # Match classifier batching as well as flow batching. Record the
        # original all-at-once classifier sensitivity instead of loosening
        # the replay threshold or attributing it to summary compression.
        view = cache.get(*g0)
        probability = detector.predict_proba(view.flow, batch_size=4096)
        with np.errstate(divide='ignore'):
            original = original_density+np.log(probability.astype(float))
            classifier_batch_difference = float(np.max(np.abs(
                np.log(view.detection_probability)-np.log(probability.astype(float)))))
        replay = float(np.max(np.abs(original-kernel)))
        small = SummaryBatch(summary.values[:128], summary.auxiliary[:128])
        partial = likelihood.log_kernel(small, obs, *g0)
        replay_partial = float(np.max(np.abs(partial-kernel[:128])))
        reversed_summary = SummaryBatch(summary.values[::-1], summary.auxiliary[::-1])
        reverse = likelihood.log_kernel(reversed_summary, obs, *g0)[::-1]
        replay_reverse = float(np.max(np.abs(reverse-kernel)))
        replay_errors.extend([replay, replay_partial, replay_reverse])
        if (not np.isfinite(replay_errors).all()
                or max(replay_errors) > protocol['code_gates']['log_kernel_replay_max_error']):
            save(args.output/'replay_failure.json', dict(observation=index, errors=replay_errors))
            raise AssertionError('summary/full/batch replay failed; do not run MH')
        logk = kernel.reshape(args.chains, args.draws_per_chain)
        rng = np.random.default_rng(args.seed+100+index)
        history, accepted = prior_metropolis(logk, rng.uniform(np.finfo(float).tiny, 1.,
            (args.chains, args.draws_per_chain-1)))
        burn = protocol['burnin']
        global_rows = history+np.arange(args.chains)[:, None]*args.draws_per_chain
        chain = summary.values[global_rows]
        checks = {name: diagnostics(chain[:, burn:, col]) for col, name in
                  ((0, 'e1'), (1, 'e2'), (3, 'm_r'), (4, 'Re_circularized'), (5, 'crowding_050'), (8, 'b'))}
        logw = logk-logsumexp(logk, axis=1, keepdims=True)
        weights = np.exp(logw)
        importance_ess = 1/np.square(weights).sum(1)
        profile = []
        ratios_saved = []
        for g1 in (.018, .022):
            shifted = likelihood.log_kernel(summary, obs, g1, 0.)
            ratios = (shifted-kernel)[global_rows[:, burn:]]
            logratio = logsumexp(ratios, axis=1)-np.log(ratios.shape[1])
            profile.append(dict(g1=g1, log_numerator_ratio_per_chain=logratio.tolist(),
                                no_population_selection_normalization=True))
            ratios_saved.append(ratios)
        explorations = [v['rhat'] is not None and v['rhat'] < 1.01 and v['ess_bulk'] > 400
                        for v in checks.values() if not v['constant']]
        mixing = bool(explorations and all(explorations) and np.all(importance_ess > 100))
        magnitude = float(30-2.5*np.log10(obs.iloc[0, 3]))
        result = dict(observation=index, measured_magnitude=magnitude,
            measured_radius_arcsec=float(obs.iloc[0, 2]*.2),
            code_replay_max_abs_error=max(replay, replay_partial, replay_reverse),
            classifier_8192_vs_4096_max_log_probability_difference=classifier_batch_difference,
            acceptance_after_burnin=accepted[:, burn:].mean(1).tolist(),
            distinct_retained_states=[len(np.unique(h[burn:])) for h in history],
            prior_importance_ess_per_chain=importance_ess.tolist(),
            prior_max_weight_per_chain=weights.max(1).tolist(), diagnostics=checks,
            b_prior_zero_fraction=float(np.mean(summary.values[:, 8] == 0)),
            b_prior_quantiles=np.quantile(summary.values[:, 8], [0, .1, .5, .9, 1]).tolist(),
            local_numerator_profiles=profile, exploration_checks_pass=mixing)
        np.savez_compressed(args.output/f'observation_{index}.npz', summaries=summary.values,
            auxiliary=summary.auxiliary, log_kernel=logk, history=history, accepted=accepted,
            posterior_log_ratios=np.stack(ratios_saved))
        results.append(result)
        save(args.output/'progress.json', results)
        print('OBSERVATION_COMPLETE', index, 'mag', magnitude, 'acceptance', result['acceptance_after_burnin'],
              'importance_ESS', importance_ess, 'exploration_pass', mixing, flush=True)
    import ufig.rendering_util
    renderer = Path(ufig.rendering_util.__file__)
    result = dict(status='completed_diagnostic', implementation_checks_pass=True,
        all_exploration_checks_pass=bool(results and all(x['exploration_checks_pass'] for x in results)),
        geometry_max_abs_error=geometry_error, log_kernel_replay_max_abs_error=max(replay_errors, default=0.),
        retention=retention, mock_sampling=mock_sampling, proposal_sampling=sampling, observations=results,
        elapsed_seconds=time.monotonic()-started,
        renderer_source=dict(path=str(renderer), sha256=digest(renderer)),
        protocol_sha256=digest(args.output/'protocol.json'),
        limitations=['Small magnitude-stratified stress panel, not representative shear validation.',
                     'Prior-proposal MH is correct for the implemented generative prior but can mix very poorly.',
                     'Native prior inversion/photometry has its previously audited grid approximation.',
                     'The induced summary prior is sampleable, not an ordinary evaluable smooth density.',
                     'Tree response can have atoms, including exact b=0; it is not smoothed.',
                     'Numerator ratios omit the separate population measurement-selection normalization.',
                     'Fixed F24 population and uniform neighbours differ from the FS2 actual-data population.',
                     'No 0.3% claim, production sampler change or automatic larger run.'])
    save(args.output/'result.json', result)
    print('SUMMARY_PILOT_COMPLETE', result['all_exploration_checks_pass'], result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
