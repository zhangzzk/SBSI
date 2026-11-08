"""Small paired replay and whole-prior-ranking probe at two saved centres."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['SBSI_RECENTER_ROOT'])
OUT = Path(os.environ['SBSI_RANKING_PROBE_ROOT'])
sys.path.insert(0, str(ROOT))
import run_v39_recenter as run  # noqa: E402

np, torch, study = run.np, run.torch, run.study


def main():
    execution = run.identity()
    protocol = study.read(OUT/'protocol.json')
    study.verify(__file__, protocol['driver_sha256'])
    audit = ROOT/'audit_20260926_v1/result.json'
    study.verify(audit, protocol['audit_sha256'])
    diagnostic = study.read(audit)
    random_groups = np.random.default_rng(202609291).choice(5000, size=32, replace=False)
    stress_groups = sorted({row['row']//4 for arm in diagnostic['arms'].values()
                            for row in arm['worst_nonlinearity_rows']})
    groups = np.unique(np.concatenate([random_groups, stress_groups]))
    ids = (groups[:, None]*4+np.arange(4)[None, :]).ravel()
    run.save(OUT/'selection.json', dict(object_ids=ids.tolist(),
        random_groups=random_groups.tolist(), stress_groups=stress_groups,
        interpretation='Fixed random groups plus explicitly post-hoc stress groups; no population bias estimate.'))
    for pass_index in (0, 1):
        likelihood, proposal, mock = run.load_pass(pass_index)
        center = study.read(ROOT/f'pass_{pass_index:02d}/center.json')['center']
        window = study.MockCatalogue(mock.measurements.iloc[ids].reset_index(drop=True),
                                     mock.truth.iloc[ids].reset_index(drop=True))
        folder = OUT/f'pass_{pass_index:02d}'
        folder.mkdir()
        # Replay original four-row groups exactly before testing another ranking.
        original = run.evaluate(likelihood, proposal, mock, ids, center, 'gaussian', 'historical_probe_replay')
        saved_parts = {int(part): dict(np.load(ROOT/f'pass_{pass_index:02d}/part_{part:02d}/gaussian.npz'))
                       for part in np.unique(ids//1000)}
        for key in run.KEYS:
            axis = 1 if key.startswith(('ladder_', 'weight_')) else 0
            saved = np.stack([saved_parts[int(i//1000)][key][:, i%1000] if axis else
                              saved_parts[int(i//1000)][key][i%1000] for i in ids], axis=axis)
            np.testing.assert_array_equal(original[key], saved, err_msg=f'saved replay: {key}')
        np.savez_compressed(folder/'prefilter.npz', object_rows=ids, **original)
        print(f'EXACT_REPLAY_PASS pass={pass_index} rows={len(ids)}', flush=True)
        started = time.monotonic()
        direct = study.run_adaptive_section5(likelihood, window, proposal,
            center=center, h=.001, draw_ladder=run.LADDER, n_candidates=1024,
            proposal_seed=8701, candidate_backend='torch',
            candidate_source='whole_catalogue_gaussian_proxy', estimator_mode='tilted_stratified',
            tilt_delta=.1, retain_full_ladder=True, full_information=True,
            object_ids=ids, object_chunk=4, atom_chunk=4096,
            progress=lambda done, total, elapsed: print(
                f'WHOLE_PRIOR pass={pass_index} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
        direct = {k: getattr(direct, k) for k in run.KEYS}
        np.savez_compressed(folder/'whole_prior.npz', object_rows=ids, **direct)
        overlap, probability_max_difference, same_indices = [], 0., True
        for lo in range(0, len(ids), 4):
            frame = window.measurements.iloc[lo:lo+4].reset_index(drop=True)
            pre = proposal.candidates(frame, n_candidates=1024, prefilter_candidates=131072, torch_device='cuda')
            top, draw = proposal.whole_catalogue_proxy_candidates_and_tilted_draw(frame,
                n_candidates=1024, n_draws=16384, delta=.1, seed=8701,
                object_ids=ids[lo:lo+4], device='cuda', object_chunk=4)
            ref = proposal.draw_tilted(pre, frame, n_draws=16384, delta=.1, seed=8701,
                object_ids=ids[lo:lo+4], device='cuda', object_chunk=4)
            same_indices &= bool(np.array_equal(draw.indices, ref.indices))
            probability_max_difference = max(probability_max_difference,
                float(np.max(np.abs(draw.probability-ref.probability))))
            overlap.extend([len(np.intersect1d(a, b))/1024 for a, b in zip(pre.indices, top.indices)])
        # Existing fused whole-prior path can reorder floating-point operations;
        # report its exact draw comparison rather than asserting bitwise q equality.
        record = dict(status='complete', pass_index=pass_index, center=center, n_rows=len(ids),
            execution=execution, protocol_sha256=study.file_hash(OUT/'protocol.json'),
            exact_saved_prefilter_replay=True, same_global_draw_indices=same_indices,
            global_probability_max_abs_difference=probability_max_difference,
            overlap_quantiles=np.quantile(overlap, [0, .1, .5, .9, 1]).tolist(),
            elapsed_direct_seconds=time.monotonic()-started,
            output_sha256={p.name: study.file_hash(p) for p in folder.glob('*.npz')}, groups={})
        for name, mask in [('random', np.isin(ids//4, random_groups)), ('stress', np.isin(ids//4, stress_groups))]:
            record['groups'][name] = dict(n_rows=int(mask.sum()), overlap_median=float(np.median(np.asarray(overlap)[mask])))
            for arm, values in [('prefilter', original), ('whole_prior', direct)]:
                rel = values['weight_relative_error'][-1, mask]
                ds = values['ladder_score'][-1, mask]-values['ladder_score'][-2, mask]
                record['groups'][name][arm] = dict(
                    relative_error_quantiles=np.quantile(rel, [.5, .9, .99]).tolist(),
                    g1_score_8k_to16k_sum=float(ds[:, 0].sum()),
                    g1_score_8k_to16k_absolute_sum=float(abs(ds[:, 0]).sum()),
                    score_sum=values['score'][mask].sum(0).tolist(),
                    information_sum=values['information'][mask].sum(0).tolist())
        run.save(folder/'result.json', record)
        print(json.dumps({k: v for k, v in record.items() if k != 'execution'}), flush=True)
        del likelihood, proposal, mock, original, direct, saved_parts
        gc.collect()
        torch.cuda.empty_cache()
    run.save(OUT/'result.json', dict(status='complete', passes=[f'pass_{i:02d}/result.json' for i in (0, 1)],
        limitation='Small paired numerical probe; no bias acceptance and no new optimizer chain.'))


if __name__ == '__main__':
    torch.set_num_threads(4)
    try:
        main()
    except Exception:
        import traceback
        run.save(OUT/'failure.json', dict(traceback=traceback.format_exc()))
        raise
