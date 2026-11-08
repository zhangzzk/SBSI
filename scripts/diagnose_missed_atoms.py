"""Trace actual posterior-heavy misses through the frozen production proposal."""
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
RUN = BASE/'inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1'
MAPS = BASE/'inference_atom_maps_20260925_v1'
OUT = BASE/'inference_missed_atom_audit_20260925_v1'
sys.path.insert(0, str(MAPS/'runtime'))
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable  # noqa: E402
from sbsi.crowding import FLOW_FEATURES  # noqa: E402
from sbsi.disk_inference_store import file_hash, subset_weights, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    reference = read(RUN/'ten_k_bright20_16631250/result.json')
    for name, expected in reference['implementation_sha256'].items():
        assert file_hash(MAPS/'runtime'/name) == expected
    assert file_hash(RUN/'proposal20m_floor50_v1/coordinates.npz') == reference['proposal_cache_sha256']
    coords = ProposalCoordinateTable.load(RUN/'proposal20m_floor50_v1')
    coords = dataclasses.replace(coords, fractional_targets=tuple(reference['proposal_fractional_targets']))
    subset = read(RUN/'prior_subset20m_bright20/manifest.json')
    prepared = read(RUN/'disk_assembled20m/manifest.json')
    zero = np.load(RUN/'disk_assembled20m/zero.npy', mmap_mode='r')
    probability = np.load(RUN/'disk_assembled20m/probability.npy', mmap_mode='r')[0]
    weights = subset_weights(subset)
    proposal = DefensiveLocalProposal(coords, weights, local_base_weights=probability)
    measurements = pd.read_parquet(RUN/'input/measurements.parquet')
    offsets = np.r_[0, np.cumsum([s['n_rows'] for s in subset['shards']])]
    proxy = proposal._tilted_proxy('cuda')
    reports = []
    for row in (142230, 3563, 409188):
        previous = read(MAPS/f'results_16696310/row_{row}_report.json')
        terms_path = MAPS/f'results_16696310/row_{row}_log_terms.npy'
        assert file_hash(terms_path) == previous['terms_sha256']
        terms = np.load(terms_path, mmap_mode='r')
        mass = np.exp(terms-previous['log_numerator'])
        saved = np.load(MAPS/f'results_16696310/row_{row}_plot_data.npz')
        observed = measurements.iloc[[row]].reset_index(drop=True)
        raw = observed[list(coords.target_names)].to_numpy(float)
        candidates = proposal.candidates(observed, n_candidates=1024,
            prefilter_candidates=131072, torch_device='cuda')
        np.testing.assert_array_equal(candidates.indices[0], saved['exact_atoms'])
        drawn = proposal.draw_tilted(candidates, observed, n_draws=16384,
            delta=.1, temperature=1., seed=8701, object_ids=np.array([row]), device='cuda')
        np.testing.assert_array_equal(drawn.indices[0], saved['draw_atoms'])
        exact, draws = candidates.indices[0], drawn.indices[0]
        reached = np.union1d(exact, draws)
        np.testing.assert_allclose(mass[reached].sum(), previous['posterior_reached'], rtol=1e-10)
        mixture = proxy.mixture(raw, delta=.1)[0]
        np.testing.assert_allclose(mixture[torch.as_tensor(draws, device='cuda')].cpu().numpy(),
                                   drawn.probability[0], rtol=1e-10)
        standardized = proposal._torch_standardized['cuda']
        query = torch.as_tensor(coords.standardize(raw), dtype=standardized.dtype, device='cuda')
        distances = torch.cdist(query, standardized)[0]
        pref = torch.topk(distances, 131072, largest=False, sorted=True).indices
        values, dispersion, base, _ = proposal._torch_reranking_tables['cuda']
        target = torch.as_tensor(raw, dtype=torch.float64, device='cuda')
        scores = base[pref] - torch.log(dispersion[pref]).sum(1) - .5*torch.square((target-values[pref])/dispersion[pref]).sum(1)
        # Check independently reconstructed shortlist scores produce the actual exact set.
        np.testing.assert_array_equal(pref[torch.topk(scores, 1024, sorted=True).indices].cpu().numpy(), exact)
        head = np.argpartition(mass, -64)[-64:]
        head = head[np.argsort(mass[head])[::-1]]
        head_missed = head[~np.isin(head, reached)][:10]
        selected = np.unique(np.r_[head[:5], head_missed])
        selected = selected[np.argsort(mass[selected])[::-1]]
        atoms = []
        for atom in selected:
            q = float(mixture[int(atom)].item())
            shard = int(np.searchsorted(offsets, atom, side='right')-1)
            rawdisp_path = RUN/f'disk_prepared20m/shard_{shard:02d}/dispersion.npy'
            assert file_hash(rawdisp_path) == prepared['receipts'][shard]['output_sha256']['dispersion.npy']
            rawdisp = np.load(rawdisp_path, mmap_mode='r')[atom-offsets[shard]]
            pref_position = torch.nonzero(pref == int(atom)).reshape(-1)
            rank = None if not len(pref_position) else int((scores > scores[pref_position[0]]).sum().item())+1
            residual = (raw[0]-coords.values[atom])/coords.dispersion[atom]
            atoms.append(dict(atom=int(atom), posterior_mass=float(mass[atom]),
                in_exact=bool(atom in exact), times_drawn=int((draws == atom).sum()),
                location_rank=int((distances < distances[int(atom)]).sum().item())+1,
                proxy_rank_in_prefilter=rank,
                proposal_q=q, expected_draw_count=16384*q,
                probability_missed_in_16384_draws=float(np.exp(16384*np.log1p(-q))),
                posterior_to_proposal_mass_ratio=float(mass[atom]/q),
                proposal_probability_rank=int((mixture > mixture[int(atom)]).sum().item())+1,
                truth=dict(zip(FLOW_FEATURES, zero[atom].tolist())),
                flow_mean=coords.values[atom].tolist(), raw_flow_std=rawdisp.tolist(),
                floored_proxy_std=coords.dispersion[atom].tolist(),
                residual_in_proxy_std=residual.tolist(),
                mean_measured_radius_arcsec=float(coords.values[atom, 2]*.2),
                mean_flux_as_magnitude=float(30-2.5*np.log10(coords.values[atom, 3])) if coords.values[atom, 3] > 0 else None))
        report = dict(row=row, observed=raw[0].tolist(), observed_mag=previous['observed_mag'],
            observation_truth=previous['observation_truth'], posterior_reached=previous['posterior_reached'],
            posterior_missed=float(1-mass[reached].sum()),
            missed_top64_mass=float(mass[head[~np.isin(head, reached)]].sum()),
            exact_proxy_mass=float(mixture[torch.as_tensor(exact, device='cuda')].sum().item()),
            actual_draws_inside_exact=int(np.isin(draws, exact).sum()), atoms=atoms,
            median_exact_truth=dict(zip(FLOW_FEATURES, np.median(zero[exact], axis=0).tolist())),
            median_exact_floored_sigma=np.median(coords.dispersion[exact], axis=0).tolist())
        write_json(OUT/f'row_{row}.json', report)
        reports.append(report)
        print(json.dumps(dict(row=row, missed_mass=report['posterior_missed'], atoms=atoms[:3]), indent=2), flush=True)
        del mixture, distances, pref, scores
    write_json(OUT/'summary.json', dict(status='complete', reports=reports,
        scope='Original 20.23m prior, K1024 M16384 seed8701, matching the annotated atom maps',
        runtime_sha256=reference['implementation_sha256'], script_sha256=file_hash(__file__),
        note='These are posterior-important examples from three inspected observations, not population-typical frequency estimates.'))


if __name__ == '__main__':
    main()
