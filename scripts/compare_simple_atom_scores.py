"""Compare simpler exact-set scores on six frozen centre-sum diagnostics.

Run under Slurm. No production settings change; proposal draws remain fixed.
"""
import dataclasses
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
RUN = BASE/'inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1'
MAPS = BASE/'inference_atom_maps_20260925_v1'
EXT = BASE/'inference_bright_extension_20260925_v1'
OUT = BASE/'inference_simple_scores_20260925_v2'
sys.path.insert(0, str(MAPS/'runtime'))
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable  # noqa: E402
from sbsi.disk_inference_store import file_hash, subset_weights, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUT/Path(__file__).name)
    ref = read(RUN/'ten_k_bright20_16631250/result.json')
    for name, expected in ref['implementation_sha256'].items():
        assert file_hash(MAPS/'runtime'/name) == expected
    assert file_hash(RUN/'proposal20m_floor50_v1/coordinates.npz') == ref['proposal_cache_sha256']
    baseline = ProposalCoordinateTable.load(RUN/'proposal20m_floor50_v1')
    baseline = dataclasses.replace(baseline, fractional_targets=tuple(ref['proposal_fractional_targets']))
    subset = read(RUN/'prior_subset20m_bright20/manifest.json')
    weights0 = subset_weights(subset)
    prob0 = np.load(RUN/'disk_assembled20m/probability.npy', mmap_mode='r')[0]
    prepared = read(RUN/'disk_assembled20m/manifest.json')
    rawparts = []
    for i, receipt in enumerate(prepared['receipts']):
        path = RUN/f'disk_prepared20m/shard_{i:02d}/dispersion.npy'
        assert file_hash(path) == receipt['output_sha256']['dispersion.npy']
        rawparts.append(np.load(path))
    raw0 = np.concatenate(rawparts)
    del rawparts
    assert raw0.shape == baseline.values.shape
    observations = pd.read_parquet(RUN/'input/measurements.parquet')
    comparison = read(EXT/'comparison/summary.json')
    prior_receipt = read(EXT/'prior/manifest.json')
    reports = []
    for prior in ('baseline', 'expanded'):
        if prior == 'baseline':
            coords, weights, prob, rawstd = baseline, weights0, prob0, raw0
        else:
            roots = [EXT/f'evaluation/shard_{i:02d}' for i in range(8)]
            for root in roots:
                receipt = read(root/'manifest.json')
                assert receipt['prior_manifest_sha256'] == file_hash(EXT/'prior/manifest.json')
                for name in ('values.npy', 'dispersion.npy', 'probability_zero.npy'):
                    assert file_hash(root/name) == receipt['output_sha256'][name]
            floors = np.array(comparison['frozen_floor'])
            flux_floor = comparison['frozen_fractional_flux_floor']
            newvalues = np.concatenate([np.load(r/'values.npy') for r in roots])
            newraw = np.concatenate([np.load(r/'dispersion.npy') for r in roots])
            newfloor = np.broadcast_to(floors, newraw.shape).copy()
            newfloor[:, 3] = np.where(np.abs(newvalues[:, 3]) > 0,
                                     np.abs(newvalues[:, 3])*flux_floor, floors[3])
            coords = dataclasses.replace(baseline, values=np.concatenate([baseline.values, newvalues]),
                dispersion=np.concatenate([baseline.dispersion, np.maximum(newraw, newfloor)]))
            object.__setattr__(coords, '_metric_center', baseline._metric_center.copy())
            object.__setattr__(coords, '_metric_scale', baseline._metric_scale.copy())
            np.testing.assert_array_equal(coords.standardize(baseline.values[::10000]),
                                          baseline.standardize(baseline.values[::10000]))
            oldweights = weights0.copy()
            bright = np.load(RUN/'disk_assembled20m/zero.npy', mmap_mode='r')[:, 3] < 20
            oldweights[bright] *= prior_receipt['old_bright_weight_multiplier']
            weights = np.r_[oldweights, np.full(len(newvalues), prior_receipt['weight_per_bright_atom'])]
            prob = np.concatenate([prob0]+[np.load(r/'probability_zero.npy') for r in roots])
            rawstd = np.concatenate([raw0, newraw])
            del newvalues, newraw, newfloor
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-9, rtol=0)
        proposal = DefensiveLocalProposal(coords, weights, local_base_weights=prob)
        for reference in comparison['rows']:
            row = reference['row']
            path = (MAPS/f'results_16696310/row_{row}_log_terms.npy' if prior == 'baseline'
                    else EXT/f'comparison/row_{row}_log_terms.npy')
            terms = np.load(path, mmap_mode='r')
            total = logsumexp(terms)
            np.testing.assert_allclose(total, reference[f'{prior}_log_numerator'], atol=1e-10, rtol=0)
            mass = np.exp(terms-total)
            observed = observations.iloc[[row]].reset_index(drop=True)
            raw = observed[list(coords.target_names)].to_numpy(float)
            candidates = proposal.candidates(observed, n_candidates=8192,
                prefilter_candidates=131072, torch_device='cuda')
            standardized = proposal._torch_standardized['cuda']
            query = torch.as_tensor(coords.standardize(raw), dtype=standardized.dtype, device='cuda')
            distances = torch.cdist(query, standardized)[0]
            pref = torch.topk(distances, 131072, largest=False, sorted=True).indices
            ids = proposal.active_indices[pref.cpu().numpy()]
            # Shared-scale kernel: the same fixed metric used by location retrieval.
            d2 = distances[pref].double().square().cpu().numpy()
            logbase = np.log(proposal.local_base_weights[ids])
            selected_std = rawstd[ids]
            valid = np.isfinite(selected_std).all(1) & (selected_std > 0).all(1)
            rawscore = np.full(len(ids), -np.inf)
            with np.errstate(over='ignore', invalid='ignore'):
                rawscore[valid] = (logbase[valid] - np.log(selected_std[valid]).sum(1)
                    - .5*np.square((raw-coords.values[ids[valid]])/selected_std[valid]).sum(1))
            rawscore[~np.isfinite(rawscore)] = -np.inf
            ranks = dict(gaussian_floored=candidates.indices[0], nearest_location=ids[:8192],
                shared_kernel=ids[np.argsort(-(logbase-.5*d2), kind='stable')[:8192]],
                gaussian_raw=ids[np.argsort(-rawscore, kind='stable')[:8192]])
            if prior == 'baseline':
                saved = np.load(MAPS/f'results_16696310/row_{row}_plot_data.npz')
                saved_exact, saved_draws = saved['exact_atoms'], saved['draw_atoms']
            else:
                saved = np.load(EXT/f'comparison/row_{row}_draws.npz')
                saved_exact, saved_draws = saved['exact'], saved['draws']
            np.testing.assert_array_equal(ranks['gaussian_floored'][:1024], saved_exact)
            q = proposal._tilted_proxy('cuda').mixture(raw, delta=.1)[0].cpu().numpy()
            assert np.isfinite(q).all() and (q > 0).all()
            np.testing.assert_allclose(q.sum(), 1, atol=1e-12, rtol=0)
            # This gives exact repeated-draw variance, beyond the 16-seed diagnostic.
            second = np.square(mass)/q
            total_second = second.sum()
            samples = []
            for seed in range(8701, 8717):
                draw = proposal.draw_tilted(candidates, observed, n_draws=16384, delta=.1,
                    temperature=1., seed=seed, object_ids=np.array([row]), device='cuda')
                drawids = draw.indices[0]
                np.testing.assert_allclose(draw.probability[0], q[drawids], rtol=1e-10)
                if seed == 8701:
                    np.testing.assert_array_equal(drawids, saved_draws)
                samples.append(drawids)
            np.savez_compressed(OUT/f'{prior}_{row}_selection.npz', **ranks,
                                draws=np.array(samples), probability=q[np.array(samples)])
            metrics = []
            for name, rank in ranks.items():
                for k, m in ((1024, 16384), (8192, 8192)):
                    exact = rank[:k]
                    assert len(np.unique(exact)) == k
                    exact_mass = float(mass[exact].sum())
                    # Subtraction loses precision when exact atoms dominate the
                    # second moment; directly sum a copy with those entries zeroed.
                    remainder_second = second.copy()
                    remainder_second[exact] = 0
                    variance = (remainder_second.sum()-(1-exact_mass)**2)/m
                    assert variance >= -1e-12
                    ratios = []
                    for drawids in samples:
                        drawids = drawids[:m]
                        outside = ~np.isin(drawids, exact)
                        ratios.append(float(exact_mass+(mass[drawids[outside]]/q[drawids[outside]]).sum()/m))
                    if name == 'gaussian_floored' and k == 1024:
                        np.testing.assert_allclose(ratios,
                            [r['likelihood_ratio'] for r in reference[f'{prior}_trials']], rtol=1e-10)
                    metrics.append(dict(score=name, k=k, m=m, exact_mass=exact_mass,
                        relative_rmse_16seeds=float(np.sqrt(np.mean((np.array(ratios)-1)**2))),
                        relative_sd_all_draws=float(np.sqrt(max(variance, 0))), ratios=ratios,
                        dominant_captured=bool(int(np.argmax(mass)) in exact)))
            report = dict(prior=prior, row=row, n_atoms=len(mass), metrics=metrics,
                prefilter_mass=float(mass[ids].sum()), raw_score_invalid=int((~np.isfinite(rawscore)).sum()),
                total_second_moment=float(total_second), terms_sha256=file_hash(path))
            reports.append(report)
            write_json(OUT/f'{prior}_{row}.json', report)
            print(json.dumps(dict(prior=prior, row=row, metrics=metrics)), flush=True)
            del q, second, remainder_second, mass, distances, pref
        del proposal, coords, rawstd
        torch.cuda.empty_cache()
    write_json(OUT/'summary.json', dict(status='complete', reports=reports,
        seeds=list(range(8701, 8717)), metric_center=baseline._metric_center.tolist(),
        metric_scale=baseline._metric_scale.tolist(), script_sha256=file_hash(__file__),
        runtime_sha256=ref['implementation_sha256'],
        limitations=['Three already inspected observations on two priors, not independent validation.',
                     'Only exact-set ranking changes; random proposal and all model evaluations stay frozen.',
                     'Centre likelihood numerators only, no shear estimate or selected normalization.',
                     'Raw cached means/scatter are at zero shear, not transported to the inference centre.',
                     'Nearest and shared-kernel scores use population scales, not measured noise covariance.']))


if __name__ == '__main__':
    main()
