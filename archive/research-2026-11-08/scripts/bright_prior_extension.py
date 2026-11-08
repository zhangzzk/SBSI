"""Audited bright-support experiment against the frozen latest inference runtime.

Run every stage under Slurm. This is an experimental prior, not a default change.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
RUN = BASE / 'inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1'
MAPS = BASE / 'inference_atom_maps_20260925_v1'
ROOT = BASE / 'inference_bright_extension_20260925_v1'
CAT = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims/FS2/FS2_25876/FS2_r_20_IA_Griffith_UDF_i.feather')
ROWS = (142230, 3563, 409188)
SHARDS = 8
sys.path.insert(0, str(MAPS / 'runtime'))

from sbsi.disk_inference_store import file_hash, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def verify(root, hashes):
    for name, expected in hashes.items():
        if file_hash(Path(root) / name) != expected:
            raise ValueError(f'Hash mismatch: {root}/{name}')


def build():
    from sbsi.coordinates import ellipticity_from_axis_ratio_angle
    from sbsi.crowding import FLOW_FEATURES
    from sbsi.forward_catalogue import rescale_emulator_pairs
    from sbsi.output_conditioned_response import TRUTH_FEATURES

    output = ROOT / 'prior'
    output.mkdir(exist_ok=False)
    audit = read(ROOT / 'catalogue_audit.json')
    downloaded = read(ROOT / 'download.json')
    # The downloaded catalogue is immutable; pin its current digest in the receipt.
    catalogue_hash = file_hash(CAT)
    assert catalogue_hash == next(d['sha256'] for d in downloaded if d['path'] == str(CAT))
    print(f'CATALOGUE {catalogue_hash}', flush=True)
    cat = pd.read_feather(CAT)
    assert len(cat) == audit['new_rows'] and not cat.duplicated(['halo_id', 'galaxy_id']).any()
    source = RUN / 'prior/shard_00'
    original = read(source / 'manifest.json')
    verify(source, original['output_sha256'])
    context = pd.read_parquet(source / 'flow_zero.parquet')
    ptr = np.load(source / 'pair_indptr.npy', mmap_mode='r')
    pairs = np.load(source / 'pair_features.npy', mmap_mode='r')
    secondaries = np.load(source / 'pair_secondary.npy', mmap_mode='r')
    c = original['conditions']
    beta, fwhm = c['moffat_beta'], c['psf_fwhm']
    psf = fwhm / 2 * np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    rms = c['pixel_rms'] * (psf/c['pixel_size'])**2 * np.pi
    e = np.hypot(context.e1_input_p, context.e2_input_p)
    old_re = context.circularized_Re_input_p.to_numpy() / np.sqrt((1-e)/(1+e))
    subset = read(RUN / 'prior_subset20m_bright20/manifest.json')
    old_bright, bright_mass = 0, 0.
    for shard in subset['shards']:
        p = Path(shard['root'])
        mask = pd.read_parquet(p/'flow_zero.parquet', columns=['r_input_p']).r_input_p.to_numpy() < 20
        w = np.load(p/'prior_weight.npy')
        old_bright += int(mask.sum())
        bright_mass += float(w[mask].sum())
    new_weight = bright_mass/(old_bright+len(cat))
    records = []
    for shard, indices in enumerate(np.array_split(np.arange(len(cat)), SHARDS)):
        out = output / f'shard_{shard:02d}'
        out.mkdir()
        new = cat.iloc[indices]
        n = len(new)
        # Independent environments from ten full randomly placed scenes. The old
        # generator samples properties independently of position, justifying this
        # product construction. Never substitute a zero-neighbour environment.
        rng = np.random.default_rng(np.random.SeedSequence([20260925, shard]))
        donors = rng.integers(len(context), size=n)
        zero = context.iloc[donors].copy().reset_index(drop=True)
        zero.index.name = 'primary_row'
        angle = new.pa_gal.to_numpy(float) + 90.
        q = new.q2d_gal.to_numpy(float)
        re = new.Re_arcsec.to_numpy(float)/np.sqrt(q)
        e1, e2 = ellipticity_from_axis_ratio_angle(q, angle)
        zero['e1_input_p'], zero['e2_input_p'] = e1, e2
        zero['r_input_p'] = new.lsst_r_mag.to_numpy(float)
        zero['sersic_n_input_p'] = new.sersic_n.to_numpy(float)
        zero['circularized_Re_input_p'] = new.Re_arcsec.to_numpy(float)
        assert tuple(zero.columns) == FLOW_FEATURES and np.isfinite(zero).all().all()
        zero.to_parquet(out/'flow_zero.parquet')
        ids = 139936000 + indices
        np.save(out/'source_atom_ids.npy', ids)
        np.save(out/'context_donor.npy', donors)
        np.save(out/'prior_weight.npy', np.full(n, new_weight))
        new.to_parquet(out/'source_rows.parquet', index=False)
        counts = ptr[donors+1]-ptr[donors]
        newptr = np.r_[0, np.cumsum(counts)]
        np.save(out/'pair_indptr.npy', newptr)
        arr = np.lib.format.open_memmap(out/'pair_features.npy', mode='w+', dtype='float64', shape=(int(newptr[-1]), 7))
        sec = np.lib.format.open_memmap(out/'pair_secondary.npy', mode='w+', dtype='int64', shape=(int(newptr[-1]),))
        for lo in range(0, n, 16384):
            hi = min(n, lo+16384)
            count = counts[lo:hi]
            parent = np.repeat(np.arange(lo, hi), count)
            edge = (np.repeat(ptr[donors[lo:hi]], count) + np.arange(count.sum())
                    - np.repeat(np.cumsum(count)-count, count))
            data = pairs[edge].copy()
            distance = data[:, 6]*np.hypot(old_re[donors[parent]], psf)
            data[:, 0] = re[parent]/np.hypot(re[parent], psf)
            data[:, 2] = zero.r_input_p.to_numpy()[parent] + 2.5*np.log10(rms)
            data[:, 4] = zero.sersic_n_input_p.to_numpy()[parent]
            data[:, 6] = distance/np.hypot(re[parent], psf)
            assert np.isfinite(data).all() and (distance < 10.0000001).all()
            # Verify the replacement against the canonical physical rescaling,
            # including the radius convention and magnitude normalization.
            if lo == 0:
                k = min(1000, len(data))
                d = data[:k]
                rs = psf*d[:, 1]/np.sqrt(1-d[:, 1]**2)
                raw = pd.DataFrame(dict(Re_input_p=re[parent[:k]], Re_input_s=rs,
                    r_input_p=zero.r_input_p.to_numpy()[parent[:k]],
                    r_input_s=d[:, 3]-2.5*np.log10(rms),
                    sersic_n_input_p=d[:, 4], sersic_n_input_s=d[:, 5], distance=distance[:k]))
                expected = rescale_emulator_pairs(raw, c)[list(TRUTH_FEATURES)].to_numpy()
                np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-10)
            arr[newptr[lo]:newptr[hi]] = data
            sec[newptr[lo]:newptr[hi]] = secondaries[edge]
        arr.flush()
        sec.flush()
        # Re-open the serialized files: NumPy scalar dimensions can otherwise
        # produce a non-literal NPY header in some installed NumPy versions.
        assert np.load(out/'pair_features.npy', mmap_mode='r').shape == (int(newptr[-1]), 7)
        assert np.load(out/'pair_secondary.npy', mmap_mode='r').shape == (int(newptr[-1]),)
        names = ('flow_zero.parquet', 'source_atom_ids.npy', 'context_donor.npy',
                 'prior_weight.npy', 'source_rows.parquet', 'pair_indptr.npy',
                 'pair_features.npy', 'pair_secondary.npy')
        record = dict(root=str(out), n_rows=n, source_start=int(indices[0]),
                      source_stop=int(indices[-1])+1, n_pairs=int(newptr[-1]),
                      output_sha256={name: file_hash(out/name) for name in names})
        write_json(out/'manifest.json', record)
        records.append(record)
        print(f'BUILT {shard} rows={n} pairs={newptr[-1]}', flush=True)
    write_json(output/'manifest.json', dict(status='complete',
        format='bright_context_enrichment_experiment_v1', n_new=len(cat),
        n_old=subset['n_rows'], n_total=subset['n_rows']+len(cat),
        old_bright=old_bright, bright_mass=bright_mass, weight_per_bright_atom=new_weight,
        old_bright_weight_multiplier=old_bright/(old_bright+len(cat)),
        source_catalogue=str(CAT), source_sha256=catalogue_hash, download=downloaded,
        source_audit=audit, baseline_subset_sha256=file_hash(RUN/'prior_subset20m_bright20/manifest.json'),
        context_source=str(source), context_manifest_sha256=file_hash(source/'manifest.json'),
        context_cases=original['cases'], context_seed=20260925,
        construction='all new rows, independent empirical neighbour environments; retain old atoms; preserve bright mass',
        overlap_policy='retain morphology realizations with overlapping physical IDs; do not call them independent galaxies',
        truth_cuts=None, conditions=c, shards=records, script_sha256=file_hash(__file__)))


def evaluate(shard):
    import torch
    from sbsi.catalogue_disk_cache import DiskCatalogueModelCache
    from sbsi.catalogue_disk_response import CatalogueDiskResponse
    from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
    from sbsi.catalogue_likelihood import CatalogueSelection, OutputCut
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import V36_LIKE, load_detection_classifier, load_disk_response
    from scripts import run_disk_inference as driver

    torch.set_num_threads(4)
    start = time.monotonic()
    receipt = read(ROOT/'prior/manifest.json')
    record = receipt['shards'][shard]
    source = Path(record['root'])
    verify(source, record['output_sha256'])
    reference = read(RUN/'ten_k_bright20_16631250/result.json')
    verify(MAPS/'runtime', reference['implementation_sha256'])
    actual = dict(measurement=file_hash(V36_LIKE.flow_checkpoints[0]),
                  emulator=file_hash(V36_LIKE.emulator_model), metadata=file_hash(V36_LIKE.emulator_metadata),
                  detector=file_hash(V36_LIKE.detection_classifiers[0]))
    assert actual == reference['preparation_identity']['models']
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V36_LIKE, device='cuda')
    emulator = load_disk_response(V36_LIKE, device='cuda', nthread=4)
    zero = pd.read_parquet(source/'flow_zero.parquet')
    weights = np.load(source/'prior_weight.npy')
    ptr = np.load(source/'pair_indptr.npy', mmap_mode='r')
    pairs = np.load(source/'pair_features.npy', mmap_mode='r')
    cache = DiskCatalogueModelCache(SimpleNamespace(galaxies=zero, weights=weights),
        detector=detector, conditions=receipt['conditions'], zero_flow=zero,
        pair_indptr=ptr, pair_features=pairs,
        pair_secondary=np.load(source/'pair_secondary.npy', mmap_mode='r'),
        source_atom_ids=np.load(source/'source_atom_ids.npy'))
    response = CatalogueDiskResponse(ptr, pairs, emulator)
    config = read(driver.CONFIG)
    selection = CatalogueSelection(OutputCut.from_specs(flow.target_transform.target_names,
        specs=config['measured_selection']['bounds']))
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response, selection=selection)
    out = ROOT/f'evaluation/shard_{shard:02d}'
    out.mkdir(parents=True, exist_ok=False)
    n = len(zero)
    values, dispersion = np.empty((n, 4)), np.empty((n, 4))
    driver.seed_stream(8201, 20+shard)
    for lo in range(0, n, 8192):
        hi = min(n, lo+8192)
        draws = np.asarray(flow.sample(zero.iloc[lo:hi], n_samples=128, batch_size=8192, qmc=True))
        values[lo:hi], dispersion[lo:hi] = draws.mean(1), draws.std(1)
        if lo % (8192*16) == 0:
            print(f'COORDINATES shard={shard} rows={hi}/{n} seconds={time.monotonic()-start:.1f}', flush=True)
    assert np.isfinite(values).all() and np.isfinite(dispersion).all()
    np.save(out/'values.npy', values)
    np.save(out/'dispersion.npy', dispersion)
    center = reference['preparation_identity']['center']
    np.save(out/'probability_zero.npy', cache.get(0., 0.).detection_probability)
    np.save(out/'probability_center.npy', cache.get(*center).detection_probability)
    observations = pd.read_parquet(RUN/'input/measurements.parquet')
    flow.compile_log_prob(mode=None, dynamic=True)
    for row in ROWS:
        observed = observations.iloc[[row]].reset_index(drop=True)
        terms = np.lib.format.open_memmap(out/f'row_{row}_log_terms.npy', mode='w+', dtype='float64', shape=(n,))
        for lo in range(0, n, 65536):
            atoms = np.arange(lo, min(n, lo+65536))
            block = likelihood.log_importance_weights_tensor(observed, *center,
                atom_indices=atoms, proposal_probability=np.ones(len(atoms)), object_chunk=16)
            terms[lo:lo+len(atoms)] = block[0].cpu().numpy()
        assert not np.isnan(terms).any() and not np.isposinf(terms).any()
        terms.flush()
        print(f'EXACT shard={shard} row={row} seconds={time.monotonic()-start:.1f}', flush=True)
    write_json(out/'manifest.json', dict(status='complete', shard=shard, n_rows=n,
        prior_manifest_sha256=file_hash(ROOT/'prior/manifest.json'), models=actual,
        runtime=reference['implementation_sha256'], seconds=time.monotonic()-start,
        script_sha256=file_hash(__file__),
        output_sha256={p.name: file_hash(p) for p in out.glob('*.npy')}))


def compare(n_candidates=1024, n_draws=16384, output=None):
    import torch
    from scipy.special import logsumexp
    from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
    from sbsi.disk_inference_store import subset_weights

    torch.set_num_threads(4)
    receipt = read(ROOT/'prior/manifest.json')
    reference = read(RUN/'ten_k_bright20_16631250/result.json')
    baseline = ProposalCoordinateTable.load(RUN/'proposal20m_floor50_v1')
    baseline = dataclasses.replace(baseline, fractional_targets=tuple(reference['proposal_fractional_targets']))
    roots = [ROOT/f'evaluation/shard_{i:02d}' for i in range(SHARDS)]
    for root in roots:
        r = read(root/'manifest.json')
        assert r['prior_manifest_sha256'] == file_hash(ROOT/'prior/manifest.json')
        verify(root, r['output_sha256'])
    # Freeze the baseline median floors and location metric for the primary
    # comparison: merely oversampling bright atoms must not retune the proposal.
    floors = baseline.dispersion.min(0)
    flux_floor = np.min(baseline.dispersion[:, 3]/np.abs(baseline.values[:, 3]))
    v = np.concatenate([baseline.values]+[np.load(r/'values.npy') for r in roots])
    newdisp = []
    for root in roots:
        values = np.load(root/'values.npy')
        raw = np.load(root/'dispersion.npy')
        floor = np.broadcast_to(floors, raw.shape).copy()
        floor[:, 3] = np.where(np.abs(values[:, 3]) > 0, np.abs(values[:, 3])*flux_floor, floors[3])
        newdisp.append(np.maximum(raw, floor))
    coords = dataclasses.replace(baseline, values=v,
        dispersion=np.concatenate([baseline.dispersion]+newdisp),
        metadata=dict(experiment='bright extension; frozen baseline median floors and metric',
                      baseline=baseline.metadata))
    # __post_init__ re-estimates the fractional metric IQR from all rows even
    # when center/scale are supplied. Freeze that derived metric too: otherwise
    # oversampling bright rows silently changes the location prefilter.
    object.__setattr__(coords, '_metric_center', baseline._metric_center.copy())
    object.__setattr__(coords, '_metric_scale', baseline._metric_scale.copy())
    probe = baseline.values[::10000]
    np.testing.assert_array_equal(coords.standardize(probe), baseline.standardize(probe))
    subset = read(RUN/'prior_subset20m_bright20/manifest.json')
    oldweights = subset_weights(subset)
    baselineweights = oldweights.copy()
    oldzero = np.load(RUN/'disk_assembled20m/zero.npy', mmap_mode='r')
    bright = oldzero[:, 3] < 20
    assert int(bright.sum()) == receipt['old_bright']
    oldweights[bright] *= receipt['old_bright_weight_multiplier']
    weights = np.r_[oldweights, np.full(receipt['n_new'], receipt['weight_per_bright_atom'])]
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-9)
    np.testing.assert_allclose(weights[:len(oldweights)][bright].sum()+weights[len(oldweights):].sum(), receipt['bright_mass'], rtol=1e-12)
    prob = np.concatenate([np.load(RUN/'disk_assembled20m/probability.npy', mmap_mode='r')[0]]+
                          [np.load(r/'probability_zero.npy') for r in roots])
    proposal = DefensiveLocalProposal(coords, weights, local_base_weights=prob, score_dtype=torch.float64)
    oldproposal = DefensiveLocalProposal(baseline, baselineweights,
        local_base_weights=prob[:len(oldweights)], score_dtype=torch.float64)
    observations = pd.read_parquet(RUN/'input/measurements.parquet')
    out = ROOT/'comparison' if output is None else Path(output)
    out.mkdir(exist_ok=False)
    reports = []
    for row in ROWS:
        oldterms_path = MAPS/f'results_16696310/row_{row}_log_terms.npy'
        previous = read(MAPS/f'results_16696310/row_{row}_report.json')
        assert file_hash(oldterms_path) == previous['terms_sha256']
        oldterms = np.load(oldterms_path).copy()
        oldterms[bright] += np.log(receipt['old_bright_weight_multiplier'])
        terms = np.concatenate([oldterms]+[np.load(r/f'row_{row}_log_terms.npy') for r in roots])
        total = logsumexp(terms)
        mass = np.exp(terms-total)
        observed = observations.iloc[[row]].reset_index(drop=True)
        oldcandidates = oldproposal.candidates(observed, n_candidates=n_candidates,
            prefilter_candidates=131072, torch_device='cuda')
        oldmass = np.exp(np.load(oldterms_path)-previous['log_numerator'])
        oldtrials = []
        for seed in range(8701, 8717):
            olddraw = oldproposal.draw_tilted(oldcandidates, observed, n_draws=n_draws,
                delta=.1, temperature=1., seed=seed, object_ids=np.array([row]), device='cuda')
            oldids = olddraw.indices[0]
            oldexact = oldcandidates.indices[0]
            outside = ~np.isin(oldids, oldexact)
            ratio = oldmass[oldexact].sum() + np.sum(oldmass[oldids[outside]]/olddraw.probability[0, outside])/n_draws
            oldtrials.append(dict(seed=seed, likelihood_ratio=float(ratio),
                posterior_reached=float(oldmass[np.union1d(oldexact, oldids[outside])].sum())))
            if seed == 8701:
                np.savez_compressed(out/f'row_{row}_baseline_draws.npz', exact=oldexact,
                                    draws=oldids, probability=olddraw.probability[0])
        if (n_candidates, n_draws) == (1024, 16384):
            np.testing.assert_allclose(oldtrials[0]['likelihood_ratio'], previous['likelihood_ratio'], rtol=1e-10)
        candidates = proposal.candidates(observed, n_candidates=n_candidates, prefilter_candidates=131072, torch_device='cuda')
        exact = candidates.indices[0]
        trials = []
        # Fixed predeclared repeated seeds, including the original production
        # seed. Report all; no seed-based acceptance or favourable selection.
        for seed in range(8701, 8717):
            draw = proposal.draw_tilted(candidates, observed, n_draws=n_draws, delta=.1,
                temperature=1., seed=seed, object_ids=np.array([row]), device='cuda')
            ids = draw.indices[0]
            inside = np.isin(ids, exact)
            reached = np.union1d(exact, ids[~inside])
            weighted = np.zeros(len(ids))
            weighted[~inside] = mass[ids[~inside]]/draw.probability[0, ~inside]
            trials.append(dict(seed=seed, likelihood_ratio=float(mass[exact].sum()+weighted.mean()),
                               posterior_reached=float(mass[reached].sum()),
                               unique_complement=int(len(np.unique(ids[~inside])))))
            if seed == 8701:
                np.savez_compressed(out/f'row_{row}_draws.npz', exact=exact, draws=ids,
                                    probability=draw.probability[0])
        top = np.argpartition(mass, -64)[-64:]
        report = dict(row=row, baseline_seed8701=previous, baseline_trials=oldtrials, expanded_trials=trials,
                      expanded_log_numerator=float(total), baseline_log_numerator=previous['log_numerator'],
                      new_atoms_posterior_mass=float(mass[len(oldterms):].sum()),
                      expanded_max_atom_mass=float(mass.max()), expanded_top64_mass=float(mass[top].sum()),
                      expanded_posterior_ess=float(1/np.square(mass).sum()))
        np.save(out/f'row_{row}_log_terms.npy', terms)
        write_json(out/f'row_{row}_report.json', report)
        reports.append(report)
        print(f'COMPARED row={row} ratio={trials[0]["likelihood_ratio"]:.6f} reached={trials[0]["posterior_reached"]:.6f} new_mass={report["new_atoms_posterior_mass"]:.6f}', flush=True)
    write_json(out/'summary.json', dict(status='complete', prior=receipt, rows=reports,
        n_candidates=n_candidates, n_draws=n_draws, prefilter_candidates=131072, seeds=list(range(8701, 8717)),
        script_sha256=file_hash(__file__), frozen_floor=floors.tolist(), frozen_fractional_flux_floor=float(flux_floor),
        frozen_metric_center=coords._metric_center.tolist(), frozen_metric_scale=coords._metric_scale.tolist(),
        metric_identity_check='exact equality on every 10000th baseline coordinate',
        limitations=['Three previously inspected rows, not representative validation.',
                     'Centre numerator and sampler coverage only; no selected normalizer or shear estimate.',
                     'Same parent population intended; the finite empirical approximation and generator-domain support differ.',
                     'Neighbour distribution remains that of the original randomly placed faint-complete scenes.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'evaluate', 'compare'])
    parser.add_argument('--shard', type=int)
    parser.add_argument('--n-candidates', type=int, default=1024)
    parser.add_argument('--n-draws', type=int, default=16384)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'evaluate':
        evaluate(args.shard)
    elif args.stage == 'compare':
        if not 1 <= args.n_candidates <= 131072 or args.n_draws < 1:
            parser.error('positive budgets and candidates <= prefilter131072 required')
        compare(args.n_candidates, args.n_draws, args.output)
    else:
        build()
